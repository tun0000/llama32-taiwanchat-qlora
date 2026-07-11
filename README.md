# 🧋 llama32-taiwanchat-qlora

**Built with Llama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tun0000/llama32-taiwanchat-qlora/blob/main/llama32-taiwanchat-qlora.ipynb)

用 [Unsloth](https://github.com/unslothai/unsloth) 以 **QLoRA** 微調 **Llama-3.2-3B-Instruct**,資料集為 [yentinglin/TaiwanChat](https://huggingface.co/datasets/yentinglin/TaiwanChat) 子集(預設 15,000 筆),讓模型的**繁體中文與台灣在地語感**更自然。訓練全程在 Google Colab(L4/T4)完成,一張免費/Pro 等級的 GPU 就能複現。

**成果模型**(在 Colab A100 上以 15,000 筆訓練 1 epoch、~1,840 步完成):

| Repo | 內容 |
|---|---|
| [`steven0226/llama-3.2-3b-taiwan-chat-lora`](https://huggingface.co/steven0226/llama-3.2-3b-taiwan-chat-lora) | LoRA adapter(97 MB,搭配 base model 載入) |
| [`steven0226/llama-3.2-3b-taiwan-chat`](https://huggingface.co/steven0226/llama-3.2-3b-taiwan-chat) | 合併後完整 fp16 模型(6.4 GB,下載即用) |

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

## 成果:微調前後對照(節錄)

正式訓練後,同一題、同 greedy 參數(差異 100% 來自權重)的實際輸出:

**Q:我是剛來台灣的交換學生,想辦手機門號。請比較預付卡和月租方案的差別,以及辦理時要準備哪些證件。**

> **微調前(base)**:「Congratulations on your arrival in Taiwan! 當你想辦手機門號時…**优點**:可以自行充值…在 **Taiwanese 手機業界**中,預付卡和月租方案都很流行…」
> (冒出英文開場、簡體字「优點」、中英夾雜)

> **微調後**:「當然可以幫助您!以下是預付卡和月租方案的比較:…為了辦理手機門號,您可能需要提供以下證件:護照或國民身份證…建議您與手機供應商聯繫,以確定他們的具體要求。」
> (全程通順繁中,零英文/簡體滲漏)

**誠實評估**:微調明顯改善的是**語言純度與語感**——簡體字與英文夾雜幾乎消失、語氣更接近台灣口語(五題皆然);**事實性沒有改善**(資費數字、店家名稱仍會幻覺),最難的題目(兩岸用語對照表)仍會輸出退化——這符合 3B 模型 + 15k 樣本 + 1 epoch 的合理預期,詳見「已知限制」。完整五題對照收錄在兩個模型 repo 的 model card 內。

## 消融實驗 v2:資料源品質(clean mix)

v1 之後,我們對 TaiwanChat 的 9 種 `id` 資料源做了稽核:機械指標掃描 6,000 筆(簡體字滲漏、大陸詞彙命中、外文比例)+ 每源抽 40 筆交由 LLM 鑑定小組逐筆評分。

**主要發現:9 個源全部是英文資料機翻成繁體「字形」的產物,資料集中沒有台灣人原生書寫的語料**——大陸用語(視頻、軟件、信息、質量…)以繁體字形遍布各源、台灣在地語境為零。「篩掉翻譯源、留原生源」的直覺假設不成立;真正能切的軸是**回覆品質**:

| | 資料源 | 狀況 |
|---|---|---|
| ✅ 保留(品質 ≥3/5) | `n/a`(43%)、`sharegpt`(16%,唯一多輪來源)、`gpt4`(5%) | 回覆完整、結構良好 |
| ❌ 丟棄(品質 ≤2/5) | `flan`、`self`、`stanford`、`unnatural`、`super`、`code` | 指令與答案錯位、數學算錯、單詞標籤式答案、**整句非中文輸出**(日文/印地文/波斯文)、Alpaca 模板殘留 |

假設:v1 在困難題(兩岸用語對照)出現的退化重複輸出,部分源自低品質「單詞標籤」模板源;丟棄後輸出穩定性應改善。

**執行 v2**——notebook 參數 cell 只改三行,其他全部不動(15,000 筆、1 epoch、seed 3407,控制變因):

```python
INCLUDE_SOURCES = ["n/a", "sharegpt", "gpt4"]
RUN_TAG         = "-clean"     # 成果推到 llama-3.2-3b-taiwan-chat-clean(-lora)
SMOKE_TEST      = False        # 照慣例:先 True 跑通,再 False 正式訓練
```

本機先驗證過濾邏輯:`uv run python scripts/preview_dataset.py --include-sources "n/a,sharegpt,gpt4"`

**成果模型**(在 Colab A100 上完成,14,633/15,000 筆通過長度過濾、1 epoch、1,830 步、36.8 分鐘、最終 train_loss 1.2069):

| Repo | 內容 |
|---|---|
| [`steven0226/llama-3.2-3b-taiwan-chat-clean-lora`](https://huggingface.co/steven0226/llama-3.2-3b-taiwan-chat-clean-lora) | v2 LoRA adapter |
| [`steven0226/llama-3.2-3b-taiwan-chat-clean`](https://huggingface.co/steven0226/llama-3.2-3b-taiwan-chat-clean) | v2 完整合併模型 |

> v1、v2 訓練資料分佈不同,train_loss 數值不能跨資料集直接比較模型優劣(不同資料子集的困惑度基準不同)。本消融實驗真正的判準是下面的實際輸出品質評測,而非 loss 曲線高低。

### 5 題盲測結果

同一組 5 題 Taiwan-context 問題、同 greedy 解碼參數,比較 v1(全 9 源)與 v2(clean mix)微調後的輸出。為避免憑感覺打分,交由獨立 LLM judge 逐句核對(大陸用語滲漏、語感自然度、事實/文化幻覺、輸出穩定性):

| 題目 | Verdict | 關鍵發現 |
|---|---|---|
| Q1 珍珠奶茶 | v2 較優 | v1 出現簡體字殘留「玉米**淀**粉」(應為澱粉);v2 完整無截斷,但年代幻覺更誇張(早了近 80 年) |
| Q2 辦門號 | v2 較優 | v1 有「短信」「視頻」;v2 僅「信息」一詞,語感更口語、針對交換學生情境客製化 |
| Q3 台南夜市 | 平手 | 兩者都嚴重幻覺(v1 編造夜市名;v2 編出「**鯨魚麵**」——台灣沒有鯨魚料理、鯨豚是保育類,文化上更離譜) |
| Q4 兩岸用語表 | v2 較優 | v1 陷入**病態無限重複迴圈**;v2 沒有迴圈了,但仍未答對花生 vs 馬鈴薯的核心差異 |
| Q5 垃圾車音樂 | **v1 較優** | v2 反而冒出更多大陸用語:「計劃」×2(應為計畫)、「通過」當連接詞(應為透過)、「運行」×3 |

**戰績:v2 三勝一平一負**,不是壓倒性勝利。

### 誠實的結論:品質篩選 ≠ 用語純度

這次消融實驗最有價值的發現,不是「v2 全面完勝」,而是一個反直覺的教訓。我們用「答案品質」當篩選軸,留下 `n/a`、`sharegpt`、`gpt4` 三源。但用稽核時算出的各源大陸用語命中率加權計算,**v2 的資料組成大陸用語密度(≈9.9%)反而比 v1 全源(≈6.8%)更高**——因為 `sharegpt`(9 源中大陸用語命中率最高,22.2%)恰好也在「答案品質」名單裡,且在 v2 的資料中佔比從 16% 拉高到 25%。Q5 的實測結果印證了這點:v2 該題冒出比 v1 更多的大陸用語痕跡。

v2 真正實質的進步是**結構穩定性**:Q4 那種無限重複迴圈的病態輸出消失了,代表濾掉低品質模板源(`unnatural`/`super` 這類單詞標籤式資料)確實減少了退化性重複——但這跟「台灣語感更純正」是兩件不同的事,不能混為一談。

若要真正降低陸語滲漏,篩選軸應該直接用「每源大陸用語命中率」,或把「答案品質」與「用語純度」兩個指標加權組合,而非只看其中一個——這是留給下一輪消融實驗的方向。

## 8B 升級實驗:更大不等於更好

用 v1 的全源配方(15,000 筆、1 epoch、其他參數不變),把 base model 從 Llama-3.2-3B 換成 **Llama-3.1-8B**(Llama-3.2 沒有 8B 版本,3.1-8B 用同一套 chat template),測試加大模型能不能改善繁中語感與推理品質。

**成果模型**(A100 標準 40GB、1,830 步、無錯誤):

| Repo | 內容 |
|---|---|
| [`steven0226/llama-3.1-8b-taiwan-chat-lora`](https://huggingface.co/steven0226/llama-3.1-8b-taiwan-chat-lora) | 8B LoRA adapter |
| [`steven0226/llama-3.1-8b-taiwan-chat`](https://huggingface.co/steven0226/llama-3.1-8b-taiwan-chat) | 8B 完整合併模型(~16GB) |

### 5 題盲測結果(3B v1 vs 8B)

同 5 題、同 greedy 解碼,獨立 LLM judge 逐字掃描——這次特別加強偵測**真正的簡體字**(不只是繁體字形寫的陸味詞彙):

| 題目 | Verdict | 關鍵發現 |
|---|---|---|
| Q1 珍珠奶茶 | **3B 較優** | 8B 編出完整的「清朝福建泉州」起源偽史,把台灣代表性飲品的發明權歸給中國大陸,還跟後段「代表台灣創新精神」自相矛盾 |
| Q2 辦門號 | 8B 較優 | 8B 正確點出「是否需簽約」這個預付卡/月租的核心差異;3B 編造不合理的預付卡月費(比月租還貴) |
| Q3 台南夜市 | 8B 較優 | 8B 列出肉圓、鹽酥雞、豆花、蚵仔煎等真實台灣小吃;3B 編出不存在的「鷹嘴豆腐」且結尾亂碼崩潰 |
| Q4 兩岸用語表 | 平手 | 3B 陷入無限重複、完全答非所問;8B 格式對題,但**「中國大陸」欄位直接輸出超過 20 處真正的簡體字**(鸡/电脑/网络/数字…),還失控列出一長串數位經濟術語 |
| Q5 垃圾車音樂 | **3B 較優** | 8B 對台灣人盡皆知的常識,同一段話裡自相矛盾三次(不知道→沒有→可能有但少見) |

**戰績:3B 2 勝、8B 2 勝、1 平——幾乎打平,不是 8B 全面勝出。**

### 誠實的結論:更大的模型犯了 3B 從沒犯過的錯誤等級

這次實驗最重要的發現:8B 在「語言純度」這個微調核心目標上,出現了兩次 3B 訓練(v1、v2)都沒有的失敗模式:

1. **真正的簡體字滲漏**:v1、v2 頂多是用繁體字形寫大陸慣用詞(視頻、信息…);8B 在 Q4 直接印出**真正的簡體字形**(鸡、电、网、数、济、艺、术、娱…),等級完全不同。
2. **更自信、更完整、但更離譜的文化幻覺**:3B 的錯誤通常是搞錯年代或原料細節;8B 在 Q1 編出**一整條完整偽歷史**(朝代、省份、行政區、古稱一應俱全),把珍珠奶茶——這個微調本該強化的台灣文化符號——的發明權歸給中國大陸清朝。
3. Q5 出現「拒答→否認→又承認」的自我矛盾,誤判台灣人盡皆知的常識。

同時 8B 確實在**可列舉的事實性知識**上優於 3B(Q2 電信方案的本質差異、Q3 真實存在的台灣小吃),符合「大模型知識量較大」的預期。

**推測(非定論)**:Llama-3.1 與 Llama-3.2 的預訓練語料組成不同,可能讓 8B 的 base model 對中國大陸相關內容有更強、更「自信」的潛在關聯;微調沒能蓋掉這個更強的先驗,反而讓它用更完整、更有說服力的方式輸出錯誤內容。這比 3B「答非所問但至少沒說錯話」的失敗模式更危險——因為讀起來像是「認真且有把握地說錯」,而非「明顯答錯」。想驗證與改善的話,下一步可以試更長訓練(目前僅 1 epoch)、加大 LoRA rank,或針對「兩岸對比」類問題額外補強訓練資料。

## GGUF 匯出

用獨立的 [`export-to-gguf.ipynb`](export-to-gguf.ipynb) 把已合併的 3B 與 8B 模型轉成 GGUF(`q4_k_m` 量化),供 [Ollama](https://ollama.com)、[LM Studio](https://lmstudio.ai) 本機執行。這個 notebook**不訓練**,只載入已在 HF 上的合併模型轉檔上傳,單一模型約 10–20 分鐘,兩個模型不用改任何參數,上傳後直接 Run all。

| Repo | 內容 |
|---|---|
| `<HF_USERNAME>/llama-3.2-3b-taiwan-chat-gguf` | 3B GGUF(q4_k_m) |
| `<HF_USERNAME>/llama-3.1-8b-taiwan-chat-gguf` | 8B GGUF(q4_k_m) |

## Repo 結構

```
llama32-taiwanchat-qlora/
├── llama32-taiwanchat-qlora.ipynb  # 上傳 Colab 直接從頭跑到尾(訓練)
├── export-to-gguf.ipynb       # 把已合併模型轉成 GGUF(不訓練,純轉檔)
├── scripts/
│   ├── data_prep.py           # 資料處理「正本」(stdlib-only,CPU 可跑)
│   ├── preview_dataset.py     # 本機驗證:下載前 50 筆實跑轉換
│   └── sync_notebook.py       # 把 data_prep.py 注入 notebook;--check 偵測漂移
├── requirements.txt           # 本機驗證用(刻意不含 torch/unsloth)
├── LICENSE                    # MIT(只涵蓋本 repo 程式碼,見下方授權說明)
└── .gitignore
```

## Colab 執行步驟

1. **開啟 notebook**:點上方 Colab 徽章,或自行上傳 `llama32-taiwanchat-qlora.ipynb` 到 [colab.research.google.com](https://colab.research.google.com)。
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
