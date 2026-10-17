# FYP II 擴充：預測層 → Black-Litterman 整合 → 風險模型 → 評估套件

和 FYP I 同一個資料夾、同一套介面檔 (`scores`、`mu`、`bl_prior`)。FYP II 的程式**可以現在就用合成資料測試**；
真實資料要等 FYP I 的因子結果出來、B 的 v1 資料與 C 的回測都就緒後才有意義。

## 指令
| 指令 | 用途 |
|---|---|
| `python run_fyp2.py all --synthetic` | 用假資料跑完 forecast + bl + risk + evaluate (示範) |
| `python run_fyp2.py --lock` | 鎖定 `config_fyp2.yaml` (預先登記)。要先 `python run_fyp1.py --lock` 鎖好 FYP I |
| `python run_fyp2.py all --dev` | 真實資料測試程式 (盲測：不輸出樣本外績效) |
| `python run_fyp2.py all` | 正式：預測 → BL → 風險模型 (兩個鎖都要通過) |
| `python run_fyp2.py evaluate --returns C的weekly_log.csv` | 評估 C 跑出來的回測 |
階段也可單獨跑: `forecast` / `bl` / `risk`。

## 五個模組
| 模組 | 做什麼 | 對應報告 |
|---|---|---|
| `src/cv.py` | 清除與禁運的 expanding walk-forward 切分，自帶「目標視窗不重疊」檢查 | 4.6 |
| `src/forecast.py` | XGBoost 分位數模型 (10/50/90%)；訓練視窗內調參；**保形校準 (CQR)** 讓 80% 區間真的涵蓋約 80%；診斷 (涵蓋率、各分位數比例、pinball 技巧、OOS IC、寬度是否有資訊) | 階段 6、4.5 |
| `src/bl_integration.py` | q50 → view 數值；區間寬度 → view 不確定性 Omega；`trust` 控制整體信任；產出 `mu_bl` 與信任參數敏感度 | 階段 7 |
| `src/risk.py` | 每個幣 GARCH(1,1) 週波動預測 (少於 250 天用 EWMA)；相關結構：樣本 / Ledoit-Wolf / PCA / 常數；用 QLIKE、偏誤統計、最小變異組合的實現波動比較 | 階段 8 |
| `src/evaluate.py` | 夏普、最大回撤、Expected Shortfall、**Deflated Sharpe**、配對區塊 bootstrap、年度與 COVID/Luna/FTX 壓力期間表 | 階段 10、2.4 |

## 輸出檔
| 檔案 | 內容 | 給誰 |
|---|---|---|
| `oos_forecasts.csv` | `date, coin_id, q10, q50, q90, fold, cqr_adj, sigma` 樣本外預測 | C、報告 |
| `mu_bl.csv` | `date, coin_id, trust, pi_weekly, mu_bl_weekly, mu_naive_weekly` (主要結果使用 `base_trust`) | **C 的最佳化器** |
| `trust_sensitivity.csv` / `bl_distance_to_prior.csv` | 各 trust 下的後驗；離先驗多遠 | 報告 |
| `risk_cov_selected.pkl` | `dict(method, cov={date: (幣清單, Sigma)})`，**週共變異數**。讀法: `pd.read_pickle(...)` | **C 的最佳化器** |
| `vol_forecast.csv` / `garch_params.csv` | 每幣每週的 GARCH 波動預測；各次重估的參數 | 報告 |
| `forecast_params.csv` / `forecast_diagnostics.csv` | 每個 fold 選的超參數；樣本外診斷 (校準前 vs 後) | 報告 |
| `risk_comparison.csv` | 各共變異數方法的比較 (共同樣本) | 報告 |
| `performance_full.csv` / `performance_by_period.csv` | 績效表 (全樣本、各年、壓力期間) | 報告 |
| `deflated_sharpe.csv` / `sharpe_bootstrap_ci.csv` / `sharpe_diff_vs_benchmark.csv` | 多重檢定後的夏普、信賴區間、相對基準的差距 | 報告 |

## 設計重點 (請在報告中說明)
1. **fold 數約 6，不是 9。** 313 個因子週、3 年最短訓練、26 週測試 → 6 個 fold。`embargo_weeks` 在「只往前走」的 expanding 窗口裡不會移除任何資料 (訓練永遠在測試之前)，保留是為了一般化；真正起作用的是 `purge_weeks`。
2. **預測目標預設是「相對同週平均」的報酬 (`target: demeaned`)**，因為 BL 的 view 是相對於市值組合的相對看法。
3. **區間為什麼要校準：** 梯度提升的分位數模型在尾端常常欠擬合，區間偏窄。程式把每個 fold 訓練視窗最後 `calib_weeks` 週留作校準集 (前面再清除 purge 週)，用 CQR 調整區間，**測試區塊只評估一次**。合成資料上，80% 區間校準前/後的樣本外涵蓋率約 79% / 80%。注意：同一週的幣彼此相關，嚴格的可交換性假設只是近似。
4. **view 不確定性的設計：** `Omega_ii = diag(P·tau·Sigma·P')_ii × (sigma_i / 平均sigma)² / trust`。區間「相對」較寬的幣，view 被打折；`trust` 決定整體尺度 (trust→0 回到先驗)。這是我的設計選擇，不是唯一做法，請在報告中說明並做 trust 敏感度分析。
5. **風險模型比較只在「所有方法都算得出來的同一批週」上進行**，否則 QLIKE 不能比較。選方法的規則 (`selection_metric`) 寫在 config 裡，事前固定。
6. **Deflated Sharpe 的 `n_trials` 要如實宣告：** = weekly_log 裡的設定數 + `extra_trials` (你在 log 以外試過的設定，例如預測層超參數組合數 8)。少報會讓結果看起來比實際顯著。

## 給 C 的介面 (BL 核心比較)
報告 4.8 的比較要「先驗、共變異數、最佳化器、限制式、樣本、成本都相同，只有 view 不同」：
- **BL 基準 (無 view)**：`bl_prior.csv` 的 `pi_weekly`，或 `mu_bl.csv` 的 `pi_weekly` 欄。
- **BL + 預測 view**：`mu_bl.csv` 的 `mu_bl_weekly` (固定一個 `trust`)。
- **樸素均值變異 (同一份預測)**：`mu_bl.csv` 的 `mu_naive_weekly` (直接把 q50 當預期報酬)。
- 共變異數：三者都用 `risk_cov_selected.pkl` 的同一份，才是公平比較。

## 這些事程式「不會」幫你做
1. **決定特徵與格網並預先登記。** `forecast.include_mu_features`、超參數格網 (預設 8 組)、`trust_grid` 都是我的建議值。
2. **加入鏈上因子。** 需要 B 的覆蓋報告與資料。做法：在 `src/factors.py` 的 `raw_factor_wide` 新增因子名稱的解析規則，再把名稱加進 `config_fyp1.yaml` 的 `factors` 清單 (注意：候選因子清單一旦鎖定就不能再增加；多一個因子，多重檢定的 `n_specs` 也會增加)。
3. **決定風險模型的負責人。** 你們的計畫表沒有寫誰負責。
4. **與 C 的最佳化器整合與 FYP II 的完整回測**，以及所有結果的解讀與寫作。

## 已知限制
- 只在合成資料上測過 (22 個自動測試，包含「改動未來資料，過去的預測必須完全不變」的洩漏測試，以及會故意失敗的對照組)；**真實資料上預期效果很小**：樣本外 IC 很低、相對無條件基準的 pinball 技巧接近 0 是正常的，請如實報告。
- 每個幣的 GARCH 各自估計，沒有 DCC 等動態相關模型；相關結構是 120～180 天窗口內的標準化報酬估計。
- 風險模型用「下一週實現報酬」評估，單週雜訊大，必須平均很多週才有意義。
- Deflated Sharpe 的 `var_sr_trials` 只用 weekly_log 內的設定估計。
- 預設假設 C 的 `weekly_log.csv` 欄位為 `date, allocator, net_ret`，`date` 是再平衡日。
