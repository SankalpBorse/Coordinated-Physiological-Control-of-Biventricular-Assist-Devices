# Table: Reward Component Ablation — Impact of $R_{\text{balance}}$

Comparison of Bilateral-Branch SAC trained with vs. without the flow-conservation penalty ($R_{\text{balance}} = -w_2 |Q_{\text{LVAD}} - Q_{\text{RVAD}}|$, $w_2 = 0.012261$ vs. $w_2 = 0.0$, Seed 42, 500,000 steps).

## 1. Standalone Ablation Results (Without $R_{\text{balance}}$)

| Scenario | FBE Without $R_{\text{balance}}$ (mL/s) | MAP (mmHg) | CO (L/min) | Suction (%) | Congestion (%) |
|---|---|---|---|---|---|
| **Rest** | 0.54 | 117.7 | 6.41 | 0.0% | 0.0% |
| **Exercise** | 3.14 | 108.3 | 6.83 | 0.0% | 0.0% |
| **SVR_down** | **18.92** | 95.6 | 7.19 | 0.0% | 0.0% |
| **Hypovolemia** | 0.63 | 117.6 | 6.39 | 0.0% | 0.0% |
| **Suction_Test** | 0.77 | 115.6 | 6.33 | 0.0% | **12.1%** |

---

## 2. Comparative Impact: With vs. Without $R_{\text{balance}}$

| Scenario | FBE With $R_{\text{balance}}$ (mL/s) | FBE Without $R_{\text{balance}}$ (mL/s) | FBE Impact / Divergence | MAP With / Without (mmHg) | Suction With / Without (%) | Congestion With / Without (%) |
|---|---|---|---|---|---|---|
| **Rest** | 5.26 | 0.54 | Steady | 95.2 / 117.7 | 0.0% / 0.0% | 0.0% / 0.0% |
| **Exercise** | 3.31 | 3.14 | Comparable | 109.4 / 108.3 | 0.0% / 0.0% | 12.2% / 0.0% |
| **SVR_down** | **1.63** | **18.92** | **+17.29 mL/s (+1060.7%)** | 92.0 / 95.6 | 0.0% / 0.0% | 18.6% / 0.0% |
| **Hypovolemia** | 0.21 | 0.63 | +0.42 mL/s | 98.1 / 117.6 | 0.0% / 0.0% | 12.5% / 0.0% |
| **Suction_Test** | 4.36 | 0.77 | Stable | 81.2 / 115.6 | 22.0% / 0.0% | 0.0% / **12.1%** |
