# -*- coding: utf-8 -*-
"""TaiwanChat 資料處理的「單一事實來源」(single source of truth)。

這個模組同時被兩邊使用:
  1. scripts/preview_dataset.py —— 本機 CPU 驗證(plain transformers AutoTokenizer)
  2. train_colab.ipynb          —— Colab GPU 訓練(unsloth 修補過的 tokenizer)

notebook 裡的副本由 scripts/sync_notebook.py 自動注入;改完本檔請執行:
    python scripts/sync_notebook.py
用 `python scripts/sync_notebook.py --check` 可偵測兩邊是否漂移。

刻意只用 Python 標準函式庫(不 import torch / transformers / datasets / unsloth):
tokenizer 與 dataset 都由呼叫端注入,因此在沒有 GPU、沒裝 unsloth 的環境也能執行。
"""

import hashlib
from collections import Counter

DATASET_ID = "yentinglin/TaiwanChat"

# Llama-3.x chat template 會在 system header 塞入「Today Date」;固定日期
# (meta 官方 template 的預設值)讓本機與 Colab 的輸出完全可重現、可比對。
FIXED_DATE = "26 Jul 2024"

# ShareGPT 欄位("from")→ OpenAI 欄位("role")的對應。
ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    # 已是 OpenAI 風格時原樣通過
    "user": "user",
    "assistant": "assistant",
}

# train_on_responses_only 依賴的 Llama-3.x header 標記(結尾的 \n\n 是必要的)。
LLAMA3_USER_MARKER = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# 微調前後對照用的 5 個台灣情境問題(notebook 與 model card 共用)。
TAIWAN_TEST_QUESTIONS = [
    "請用台灣人習慣的說法介紹珍珠奶茶:它的由來、基本做法,以及為什麼能代表台灣的手搖飲文化。",
    "我是剛來台灣的交換學生,想辦手機門號。請比較預付卡和月租方案的差別,以及辦理時要準備哪些證件。",
    "請推薦台南有名的夜市,並列出五樣必吃的小吃,每樣用一兩句話介紹。",
    "「土豆」這個詞在台灣和中國大陸分別指什麼?請再舉三組兩岸說法不同的日常用語,整理成表格。",
    "請解釋台灣的「垃圾不落地」政策,以及為什麼垃圾車會播放音樂?",
]


def count_sources(dataset):
    """回傳 {id 資料源: 筆數},由多到少。TaiwanChat 有 9 種 id(flan/sharegpt/self/…)。"""
    return dict(Counter(dataset["id"]).most_common())


def filter_by_sources(dataset, include_sources):
    """依 `id` 欄過濾資料源(消融實驗用)。include_sources=None → 原樣返回。

    回傳 (dataset, stats dict)。與 notebook、preview 共用,確保兩邊過濾語意一致。
    """
    if include_sources is None:
        return dataset, {"filtered": False}
    include = set(include_sources)
    n_before = len(dataset)
    ds = dataset.filter(lambda ex: ex["id"] in include)
    return ds, {
        "filtered": True,
        "include": sorted(include),
        "n_before": n_before,
        "n_after": len(ds),
    }


def normalize_text(text):
    """統一換行(資料裡混有 Windows \\r\\n)並去除頭尾空白。"""
    return (text or "").replace("\r\n", "\n").strip()


def standardize_sharegpt(conversation):
    """把 ShareGPT 格式 [{"from": "human"/"gpt", "value": ...}] 轉成
    OpenAI 格式 [{"role": "user"/"assistant", "content": ...}]。

    語意等同 unsloth.chat_templates.standardize_sharegpt,但不依賴 unsloth,
    因此本機 CPU 也能跑同一套邏輯。未知 role 會丟 ValueError(呼叫端決定丟棄)。
    """
    messages = []
    for turn in conversation:
        src_role = turn.get("from", turn.get("role"))
        role = ROLE_MAP.get(src_role)
        if role is None:
            raise ValueError("unknown role: %r" % (src_role,))
        content = normalize_text(turn.get("value", turn.get("content")))
        if not content:
            raise ValueError("empty message content")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("empty conversation")
    return messages


def render_conversation(conversation, tokenizer):
    """standardize + 套用 tokenizer 的 Llama-3.x chat template,回傳純文字。

    date_string 固定為 FIXED_DATE,消除 template 內建「今天日期」造成的
    跨日期/跨環境非決定性。轉換失敗回傳 ""(由 prepare_dataset 過濾並計數)。
    """
    try:
        messages = standardize_sharegpt(conversation)
    except ValueError:
        return ""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        date_string=FIXED_DATE,
    )


def formatting_prompts_func(examples, tokenizer):
    """datasets.map 用的 batched 函式:conversations → {"text": [...]}"""
    texts = [render_conversation(convo, tokenizer) for convo in examples["conversations"]]
    return {"text": texts}


def add_token_counts(examples, tokenizer):
    """datasets.map 用的 batched 函式:計算每筆 text 的 token 數。

    template 輸出已含 <|begin_of_text|>,所以 add_special_tokens=False,
    避免重複計入 BOS 導致過濾邊界偏移。
    """
    ids = tokenizer(examples["text"], add_special_tokens=False)["input_ids"]
    return {"n_tokens": [len(x) for x in ids]}


def prepare_dataset(dataset, tokenizer, max_seq_length):
    """完整資料處理管線:格式化 → 丟無效列 → 計 token → 過濾過長樣本。

    dataset:已由呼叫端取樣完成的 datasets.Dataset
      (notebook 端:shuffle(seed).select(range(N));preview 端:streaming 前 N 筆)。
    回傳 (只剩 "text" 欄的 dataset, 統計 dict)。
    """
    n_input = len(dataset)

    ds = dataset.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[c for c in dataset.column_names if c != "text"],
    )
    ds = ds.filter(lambda ex: ex["text"] != "")
    n_valid = len(ds)

    ds = ds.map(add_token_counts, batched=True, fn_kwargs={"tokenizer": tokenizer})
    all_lens = sorted(ds["n_tokens"])
    ds = ds.filter(lambda ex: ex["n_tokens"] <= max_seq_length)
    n_kept = len(ds)
    ds = ds.remove_columns(["n_tokens"])

    stats = {
        "n_input": n_input,
        "n_dropped_invalid": n_input - n_valid,
        "n_dropped_overlong": n_valid - n_kept,
        "n_kept": n_kept,
        "max_seq_length": max_seq_length,
    }
    if all_lens:
        stats.update({
            "token_len_min": all_lens[0],
            "token_len_mean": round(sum(all_lens) / len(all_lens), 1),
            "token_len_p95": all_lens[min(len(all_lens) - 1, int(len(all_lens) * 0.95))],
            "token_len_max": all_lens[-1],
        })
    return ds, stats


def describe_dataset(dataset, n_rows=2, max_chars=300):
    """印出欄位 schema 與前 n_rows 筆(長字串截斷)。兩邊輸出格式一致,方便比對。"""

    def _trunc(value):
        if isinstance(value, str):
            return value if len(value) <= max_chars else value[:max_chars] + "…[截斷]"
        if isinstance(value, list):
            return [_trunc(v) for v in value]
        if isinstance(value, dict):
            return {k: _trunc(v) for k, v in value.items()}
        return value

    lines = ["欄位:%s" % (dataset.column_names,)]
    for i in range(min(n_rows, len(dataset))):
        lines.append("--- 第 %d 筆 ---" % i)
        for key, value in dataset[i].items():
            lines.append("%s = %r" % (key, _trunc(value)))
    return "\n".join(lines)


def crosscheck_messages_column(row):
    """用資料集內建的平行欄位 `messages`(OpenAI 格式)驗證我們的
    standardize_sharegpt(conversations) 轉換結果。回傳不一致描述清單(空 = OK)。"""
    problems = []
    try:
        ours = standardize_sharegpt(row["conversations"])
    except ValueError as e:
        return ["standardize failed: %s" % e]
    theirs = [
        {"role": m["role"], "content": normalize_text(m["content"])}
        for m in row["messages"]
    ]
    if len(ours) != len(theirs):
        return ["turn count mismatch: ours=%d dataset=%d" % (len(ours), len(theirs))]
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a["role"] != b["role"]:
            problems.append("turn %d role mismatch: %r vs %r" % (i, a["role"], b["role"]))
        if a["content"] != b["content"]:
            problems.append("turn %d content mismatch" % i)
    return problems


def check_llama3_markers(text):
    """確認 render 結果含有 response-only loss masking 依賴的 Llama-3 header 標記。"""
    missing = [m for m in (LLAMA3_USER_MARKER, LLAMA3_ASSISTANT_MARKER) if m not in text]
    if missing:
        raise AssertionError(
            "chat template 輸出缺少 Llama-3 header 標記 %r —— "
            "train_on_responses_only 會把所有 label 遮罩掉,等於白練!" % (missing,)
        )
    return True


def format_parity_sha256(tokenizer):
    """對固定探針對話 render 後取 SHA-256。

    本機 preview 與 Colab notebook 各印一次:相同 → 兩邊格式化 100% 一致;
    不同 → template 來源有差(例如 unsloth get_chat_template 與 stock template
    的日期行差異),需人工確認差異僅限於此。
    """
    probe = [
        {"role": "user", "content": "測試"},
        {"role": "assistant", "content": "回覆"},
    ]
    text = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False, date_string=FIXED_DATE
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
