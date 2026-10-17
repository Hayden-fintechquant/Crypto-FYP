# Student A 程式包：因子驗證 → 訊號組合 → Black-Litterman 先驗

你只要：**① 把 B 的兩個檔案放進 `data/`　② 填好 `config.yaml` 並鎖定　③ 執行 `python run_all.py`**。
其餘（因子計算、篩選、IC/ICIR、五分位、Fama-MacBeth、Newey-West、多重檢定、三種訊號組合、預期報酬校準、BL 反向最佳化、前視稽核）都已寫好，並用合成資料測試過。

## 安裝
```
pip install -r requirements.txt
```

## 三步驟
1. **放資料**：`data/panel.csv` 與 `data/universe.csv`（欄位見下）。
2. **預先登記**：打開 `config.yaml`，在「看到任何真實因子結果之前」確認所有門檻，然後
   ```
   python run_all.py --lock
   ```
3. **正式執行**：
   ```
   python run_all.py
   ```
   結果在 `results/`。`config.yaml` 鎖定後如果被改過，正式執行會拒絕執行。

## 其他指令
| 指令 | 用途 |
|---|---|
| `python run_all.py --synthetic` | 用假資料跑完整流程（內建一個已知訊號: `liq_28d`），隨時可以檢查程式沒壞 |
| `python run_all.py --dev` | 用真實資料（例如 B 的 v0.1）**測試程式有沒有報錯**。盲測：不輸出因子績效，避免鎖定前偷看結果 |
| `python run_all.py --lock` | 鎖定 `config.yaml` |
| `python -m pytest tests` | 跑 14 個自動測試（約 1～2 分鐘） |

## 輸入檔格式
- `panel.csv`：`date, coin_id, close, volume_usd, mcap`（其他欄位會被忽略）。日期是 UTC 日，`close` 是該日收盤。
- `universe.csv`：`rebalance_date, coin_id, is_member`（`rank, mcap` 可有可無）。`is_member` 用 1/0 或 True/False。

## 時間規則（全組統一）
`rebalance_date` 若是週一（v0 的情況），程式用「週日收盤前」的資料算因子（`info_cutoff_offset_days: 1`），預測的是「週日收盤 → 下週日收盤」的報酬。如果 B 的 v1 把 `rebalance_date` 改成週日，把這個值改成 0。

## 輸出檔（`results/`）
| 檔案 | 內容 | 給誰 |
|---|---|---|
| `scores.csv` | `date, coin_id, factor, score`（標準化因子分數） | C |
| `mu.csv` | `date, coin_id, mu_weekly, method`（ew / icir / fm 三種預期報酬，週報酬單位） | C |
| `bl_prior.csv` | `date, coin_id, w_mkt, pi_weekly, delta`（市值權重與隱含均衡報酬） | C |
| `coverage.csv` | 各因子覆蓋率 | 報告 |
| `redundancy_factor_values.csv` / `_returns.csv` | 因子值相關 / 因子報酬序列相關（兩個層次） | 報告 |
| `ic_table.csv` | IC、ICIR、NW t 值、滾動 IC 範圍 | 報告 |
| `quintile_table.csv` | 五分位各組報酬、多空價差、單調性 | 報告 |
| `fm_univariate.csv` / `fm_multivariate.csv` / `fm_spanning.csv` | Fama-MacBeth 溢酬表 | 報告 |
| `multiple_testing.csv` | 全部檢定的原始 p 值、Holm、BH 修正 p 值（總數已宣告） | 報告 |
| `verdicts.csv` | 依事前門檻的 PASS / FAIL 與原因 | 報告 |
| `lookahead_audit.csv` | 前視稽核表 | 報告附錄 |
| `summary.txt` | 以上重點的文字摘要 | 你 |

## 程式模組對應你的任務
| 你的任務 | 模組 |
|---|---|
| 1. 因子規格與篩選 | `src/factors.py`、`src/screen.py`、`config.yaml` |
| 2. IC / ICIR / 五分位 | `src/validate.py` |
| 3. Fama-MacBeth + NW + 多重檢定 | `src/fama_macbeth.py`、`src/stats.py`、`src/pipeline.py` |
| 4. 訊號組合與預期報酬校準 | `src/combine.py`、`src/audit.py`（前視稽核） |
| 5. Black-Litterman | `src/bl.py`（反向最佳化、後驗、兩個極限情況測試） |
| 預先登記 | `src/prereg.py`、`run_all.py --lock` |

## 這些事程式「不會」幫你做（要你自己做）
1. **決定門檻並寫成文件。** `config.yaml` 裡的數字是我建議的預設值，不是定論；鎖定前請你確認。
2. **因子樣本起點。** 預設 2020-09-21（舊 universe 達 30 個幣的第一週）。B 的 v1 出來後，用 B 的每週成員數表重新決定，再鎖定。
3. **解讀結果與寫報告。** 程式只給數字與 PASS/FAIL；為什麼、經濟意義、限制，要你寫。
4. **用已發表結果驗證 Fama-MacBeth。** 程式已用 statsmodels 驗證 Newey-West，並用已知答案驗證 FM 迴歸；但「重現論文單變量結果」需要真實資料與論文資料，要你自己做一次。
5. **FYP II 的 BL views。** `bl.py` 已有 `relative_views()` 與 `omega_from_trust()`，但第一學期不使用（無 views 時，後驗 = 先驗）。

## 已知限制（請在報告中誠實說明）
- **篩選決定使用全樣本**：覆蓋率與冗餘剔除用整個樣本的因子值判斷（不涉及報酬，所以不是績效前視），並在事前登記的門檻下一次完成。前視稽核對這一步採「沿用既定因子清單」。
- **完整樣本法**：每週只保留所有留下因子都有值的幣，確保各因子在同一批幣上比較。
- **上線日** 用 panel 第一筆資料日推算；如果 B 能提供真正上市日，請改用。
- **`fm` 預期報酬** 是各因子的滾動 FM 溢酬 × 因子分數（不含截距），代表「相對」預期報酬，適合 BL 的相對 view。
- **下市處理** 由 `config.yaml` 的 `delist_policy` 決定；建議正式結果同時跑 `last_price` 與 `minus_100` 兩種並比較。
- 在合成資料上測過；**還沒在 B 的真實資料上跑過**。第一次跑真實資料時，先用 `--dev` 看有沒有報錯。

## 常見錯誤訊息
| 訊息 | 意思 |
|---|---|
| `panel 缺少欄位` / `universe 缺少欄位` | B 的檔案欄位名稱不對，請 B 照介面規格修正 |
| `panel 的 (date, coin_id) 有重複` | B 的資料有重複列 |
| `樣本期間內沒有可用的再平衡日` | `sample.start/end` 與資料日期對不上 |
| `篩選後沒有任何因子留下` | 覆蓋率或每週最少幣數太嚴；先看 `coverage.csv`（不要為了讓因子留下而改門檻，要記錄原因） |
| `config.yaml 在鎖定之後被修改過` | 預先登記鎖偵測到改動。如果真的必須改，重新 `--lock` 並在報告中說明 |
