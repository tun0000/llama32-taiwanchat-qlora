# -*- coding: utf-8 -*-
"""本機(CPU)驗證:實際下載 TaiwanChat 前 N 筆,套用與 llama32-taiwanchat-qlora.ipynb
完全相同的格式化邏輯(scripts/data_prep.py),印出轉換結果供人工檢查。

用法(Windows / uv):
    uv run python scripts/preview_dataset.py
    uv run python scripts/preview_dataset.py --n 50 --max-seq-length 2048

只需要 CPU 與網路:用 streaming 抓前 N 筆(KB 級流量,不會下載整包 678MB),
tokenizer 用 ungated 的 unsloth 鏡像(僅 ~10MB,不需要 HF token)。
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

# Windows 主控台預設 cp950,印繁中/emoji 會炸;強制 UTF-8。
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="預覽筆數(預設 50)")
    parser.add_argument(
        "--tokenizer",
        default="unsloth/Llama-3.2-3B-Instruct",
        help="tokenizer repo(預設用 ungated 鏡像,免 HF token)",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--include-sources",
        default=None,
        help="逗號分隔的 id 資料源清單(消融實驗用),例:'n/a,sharegpt,gpt4';不給 = 全部",
    )
    args = parser.parse_args()
    include = set(s.strip() for s in args.include_sources.split(",")) if args.include_sources else None

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    print("=" * 88)
    print(f"[1/5] streaming 下載 {data_prep.DATASET_ID} 前 {args.n} 筆"
          + (f"(僅 {sorted(include)} 源)" if include else "") + " …")
    stream = load_dataset(data_prep.DATASET_ID, split="train", streaming=True)
    picked = (row for row in stream if include is None or row["id"] in include)
    rows = list(islice(picked, args.n))
    ds = Dataset.from_list(rows)
    print(f"取得 {len(ds)} 筆;資料源分佈:{data_prep.count_sources(ds)}")
    print()
    print(data_prep.describe_dataset(ds, n_rows=2))

    print("=" * 88)
    print(f"[2/5] 載入 tokenizer:{args.tokenizer}(僅 tokenizer 檔,~10MB)…")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("=" * 88)
    print(f"[3/5] crosscheck:用資料集平行欄位 `messages` 驗證 standardize_sharegpt() …")
    n_bad = 0
    for i in range(len(ds)):
        problems = data_prep.crosscheck_messages_column(ds[i])
        if problems:
            n_bad += 1
            if n_bad <= 3:
                print(f"  第 {i} 筆不一致:{problems}")
    print(f"crosscheck 結果:{len(ds) - n_bad}/{len(ds)} 筆一致" + (" ✅" if n_bad == 0 else " ⚠️"))

    print("=" * 88)
    print(f"[4/5] 套用與 notebook 相同的 prepare_dataset()(max_seq_length={args.max_seq_length})…")
    prepared, stats = data_prep.prepare_dataset(ds, tokenizer, args.max_seq_length)
    for key, value in stats.items():
        print(f"  {key} = {value}")

    print("=" * 88)
    print("[5/5] 轉換後樣本(Llama-3.x chat template 格式):")
    for i in range(min(3, len(prepared))):
        text = prepared[i]["text"]
        shown = text if i == 0 else (text[:800] + ("…[截斷]" if len(text) > 800 else ""))
        print(f"--- 樣本 {i}{'(完整)' if i == 0 else '(截 800 字)'} ---")
        print(shown)
        print()

    data_prep.check_llama3_markers(prepared[0]["text"])
    print("Llama-3 header 標記檢查:✅ user/assistant markers 都在(masking 依賴成立)")
    print(f"FORMAT_PARITY_SHA256 = {data_prep.format_parity_sha256(tokenizer)}")
    print("(此值應與 Colab notebook C8 印出的值一致;不一致見 README 的說明)")


if __name__ == "__main__":
    main()
