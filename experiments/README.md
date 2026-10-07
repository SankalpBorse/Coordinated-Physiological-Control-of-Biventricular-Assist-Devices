# In Silico Validation & Ablation Experiments

This directory contains standalone clinical validation and causal ablation suites complementing the core 9-scenario benchmark:

## 1. Virtual Patient Cohort Robustness (`virtual_patients/`)
- **Objective**: Evaluates zero-shot generalization across 5 pathological phenotypes under both Rest and Exercise conditions ($n = 5$ random seeds, 100 evaluation episodes).
- **Phenotypes**:
  - `VP-1`: Nominal Dilated Cardiomyopathy reference
  - `VP-2`: Depressed Ischemic Contractility
  - `VP-3`: Hypertensive & Stiff (impaired diastolic compliance)
  - `VP-4`: Secondary Pulmonary Hypertension (remodelled pulmonary resistance)
  - `VP-5`: Hyperdynamic / Tachycardic stress
- **Execution Script**: [`evaluate_virtual_patients.py`](virtual_patients/evaluate_virtual_patients.py)
- **Summary Results**: [`table_virtual_patients.csv`](virtual_patients/table_virtual_patients.csv) & [`table_virtual_patients.md`](virtual_patients/table_virtual_patients.md)

---

## 2. Reward Component Ablation (`reward_ablation/`)
- **Objective**: Establishes the causal necessity of the flow-conservation penalty ($R_{\text{balance}} = -w_2 |Q_{\text{LVAD}} - Q_{\text{RVAD}}|$, $w_2 = 0.012261$).
- **Protocol**: Trains Bilateral-Branch SAC from scratch for 500,000 steps with $w_2 = 0.0$ and evaluates across 5 benchmark scenarios (Rest, Exercise, SVR_down, Hypovolemia, Suction_Test).
- **Execution Script**: [`train_ablation_no_balance.py`](reward_ablation/train_ablation_no_balance.py)
- **Summary Results**: [`table_reward_ablation.csv`](reward_ablation/table_reward_ablation.csv) & [`table_reward_ablation.md`](reward_ablation/table_reward_ablation.md)
