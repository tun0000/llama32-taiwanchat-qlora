# 🧋 llama32-taiwanchat-qlora

**Built with Llama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tun0000/llama32-taiwanchat-qlora/blob/main/train_colab.ipynb)

用 [Unsloth](https://github.com/unslothai/unsloth) 以 **QLoRA** 微調 **Llama-3.2-3B-Instruct**,資料集為 [yentinglin/TaiwanChat](https://huggingface.co/datasets/yentinglin/TaiwanChat) 子集(預設 15,000 筆),讓模型的**繁體中文與台灣在地語感**更自然。訓練全程在 Google Colab(L4/T4)完成,一張免費/Pro 等級的 GPU 就能複現。

**成果模型**(訓練完成後出現):

| Repo | 內容 |
|---|---|
| [`<HF_USERNAME>/llama-3.2-3b-taiwan-chat-lora`](https://huggingface.co/models?search=llama-3.2-3b-taiwan-chat-lora) | LoRA adapter(數十 MB,搭配 base model 載入) |
| [`<HF_USERNAME>/llama-3.2-3b-taiwan-chat`](https://huggingface.co/models?search=llama-3.2-3b-taiwan-chat) | 合併後完整 fp16 模型(下載即用) |

## 為什麼做這個?

Llama-3.2-3B-Instruct 用中文提問時,常出現簡體字、中國大陸用語(「土豆」當馬鈴薯、「視頻」、「打車」),對台灣特有情境(悠遊卡、垃圾不落地、辦門號)也不熟。TaiwanChat 是台大 [Taiwan LLM 計畫](https://arxiv.org/abs/2311.17487)釋出的 48 萬筆繁中對話資料,拿它的子集做 QLoRA 微調,是在單張消費級 GPU 上改善「語言在地化」最划算的路徑。

## 展示了哪些技術

- **QLoRA**:4-bit NF4 量化載入 + LoRA(r=16, α=16),3B 模型在 T4(16GB)也訓得動
- **Unsloth**:比原生 transformers 快 ~2 倍、省 ~30% VRAM 的微調框架
- **Llama-3 chat template 資料工程**:ShareGPT → OpenAI 格式標準化、token 長度過濾、固定 `date_string` 消除非決定性
- **Response-only loss masking**:`train_on_responses_only`,只對 assistant 回覆計 loss,並附遮罩正確性驗證 cell
- **微調前後對照評測**:5 個台灣情境問題、greedy decoding,差異可完全歸因於權重
- **雙格式發佈 + 自動 model card**:adapter 與 merged 各一個 HF repo,卡片(繁中+英文摘要)由訓練結果自動生成,含 Llama 3.2 授權合規(Built with Llama、NOTICE、LICENSE.txt)
- **notebook ↔ 本機腳本單一事實來源**:資料處理邏輯只有一份(`scripts/data_prep.py`),機械同步進 notebook,漂移可偵測

## Repo 結構

```
llama32-taiwanchat-qlora/
├── train_colab.ipynb          # 上傳 Colab 直接從頭跑到尾
├── scripts/
│   ├── data_prep.py           # 資料處理「正本」(stdlib-only,CPU 可跑)
│   ├── preview_dataset.py     # 本機驗證:下載前 50 筆實跑轉換
│   └── sync_notebook.py       # 把 data_prep.py 注入 notebook;--check 偵測漂移
├── requirements.txt           # 本機驗證用(刻意不含 torch/unsloth)
├── LICENSE                    # MIT(只涵蓋本 repo 程式碼,見下方授權說明)
└── .gitignore
```

## Colab 執行步驟

1. **開啟 notebook**:點上方 Colab 徽章,或自行上傳 `train_colab.ipynb` 到 [colab.research.google.com](https://colab.research.google.com)。
2. **選 GPU**:選單 Runtime(執行階段)→ Change runtime type → **L4**(建議,Colab Pro)或 T4(較慢)。
3. **設定 HF_TOKEN**:左側 **🔑 Secrets → Add new secret**
   - Name:`HF_TOKEN`
   - Value:HuggingFace **write** token(到 [hf.co/settings/tokens](https://huggingface.co/settings/tokens) 建立)
   - 把該 secret 的「Notebook access」開關打開
4. **先跑 SMOKE_TEST**(強烈建議):參數 cell 預設 `SMOKE_TEST = True`(只取 200 筆、訓練 20 步、不推送)。Runtime → **Run all**,約 10 分鐘可驗證整條流程(安裝→資料→訓練→遮罩驗證→推論)都能跑通。
5. **正式訓練**:把 `SMOKE_TEST = False`,再 Run all。結束後 adapter 與 merged 模型會自動推上你的 HF 帳號,並附自動生成的 model card。

### 預估運算單元(Compute Units)

社群實測值,會浮動;實際費率請看 Colab 的 Runtime → View resources。

| 執行 | GPU | 預估時間 | 預估 CU |
|---|---|---|---|
| SMOKE_TEST | L4 / T4 | ~10 分鐘 | < 1 |
| 正式(15k 筆、1 epoch) | **L4(建議)** | ~1–2 小時 | ~2–6 |
| 正式(15k 筆、1 epoch) | T4 | ~2.5–4.5 小時 | ~3–9 |

Colab Pro 每月 100 CU,跑好幾次完整訓練都夠。

### Base model 說明(gated vs 鏡像)

notebook 預設 `BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"`——這是 meta-llama 官方權重的 **ungated 鏡像**(權重相同、免申請、下載較快)。若你已在 HF 網頁申請並獲准存取 [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),把參數 cell 的註解那行換上即可,notebook 會在第 4 個 cell 先驗證你的 token 有權存取(而不是下載到一半才爆)。

## 本機驗證(不需要 GPU)

訓練前可先在本機確認資料轉換邏輯正確(與 notebook 共用同一套程式碼)。Windows(PowerShell)+ [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python scripts/preview_dataset.py        # 下載前 50 筆(streaming、KB 級流量)實跑轉換
```

輸出會包含:資料 schema 與前 2 筆、`messages` 平行欄位 crosscheck、token 長度統計、3 筆 chat template 轉換結果、Llama-3 header 標記檢查,以及 `FORMAT_PARITY_SHA256`。

> **FORMAT_PARITY_SHA256 是什麼?** 本機與 Colab 各自對同一段固定對話 render 後取的雜湊。兩邊一致代表格式化 100% 相同;若不一致,唯一合法的差異來源是 template 出處(本機用 stock tokenizer template、Colab 經過 unsloth `get_chat_template`)的日期行,masking 依賴的 header 標記則兩邊都有 assert 硬檢查。

## 開發流程(改資料處理邏輯時)

資料處理只有一份正本:`scripts/data_prep.py`。改完執行:

```powershell
python scripts/sync_notebook.py          # 注入 notebook 的共用模組 cell
python scripts/sync_notebook.py --check  # 驗證兩邊一致(CI / commit 前建議跑)
```

不要直接改 notebook 裡的那個 cell(開頭有哨兵註解),會被下次同步覆寫。

## 已知限制

- 事實型內容(電信資費、店家資訊)仍會過時或幻覺——微調改善的是**語感與在地知識的表達**,不是即時資訊。
- TaiwanChat 含翻譯/合成語料,微調後偶爾仍可能滲出簡體或英文片段。
- 未做安全對齊,安全行為同 base model。
- 資料集為 CC BY-NC 4.0 → **產出模型僅供研究/非商業用途**。

## 授權

| 部分 | 授權 |
|---|---|
| 本 repo 的程式碼與 notebook | [MIT](LICENSE) |
| 模型權重(base 與微調產物) | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/)(模型 repo 內附 LICENSE.txt 與 NOTICE;遵守 [AUP](https://www.llama.com/llama3_2/use-policy)) |
| 訓練資料 TaiwanChat | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

## 引用

```bibtex
@misc{lin2023taiwanllm,
  title={Taiwan LLM: Bridging the Linguistic Divide with a Culturally Aligned Language Model},
  author={Yen-Ting Lin and Yun-Nung Chen},
  year={2023},
  eprint={2311.17487},
  archivePrefix={arXiv}
}
```
