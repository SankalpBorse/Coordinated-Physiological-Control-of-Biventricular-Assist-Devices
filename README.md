# Coordinated Physiological Control of Biventricular Assist Devices (BiVAD) Using Bilateral-Branch Soft Actor-Critic (SAC)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Solo M.Tech Thesis](https://img.shields.io/badge/Status-Journal%20Manuscript%20(25%20pp)-purple.svg)](#13-license--citation)

---

## Overview

Biventricular heart failure carries severe morbidity and near-total mortality without mechanical circulatory support. While continuous-flow Left Ventricular Assist Devices (LVADs) are clinical mainstays for isolated left ventricular failure, total biventricular assistance (BiVAD) requires coordinated dual-pump actuation to balance the systemic and pulmonary circulations in real time. Independent decentralized controllers inevitably produce dangerous interventricular flow imbalances, precipitating fatal pulmonary edema (over-pumping) or ventricular suction collapse (under-pumping). Meanwhile, standard centralized deep reinforcement learning (DRL) policies suffer from an **Arterial Pressure Priority Trap**, where unpartitioned networks prioritize systemic arterial pressure at the expense of venous congestion and pulmonary vascular overload.

This repository provides the official implementation of **Bilateral-Branch Soft Actor-Critic (SAC)**, a biologically grounded deep reinforcement learning framework for closed-loop physiological BiVAD control. By incorporating an anatomical inductive bias—segregating systemic and pulmonary observation subspaces into dedicated left and right feature extraction branches before intermediate cross-ventricular coordination—the policy decouples dual-pump dynamics while preserving synchronized flow regulation. Benchmarked against clinical standard-of-care Constant Speed, decentralized Dual-PI feedback, physiological Dual Starling-like control, Centralized Multi-Objective Model Predictive Control (CMO-MPC), Centralized Flat SAC, and Independent Dual-Agent SAC across 9 clinical scenarios and 5 multi-phenotype virtual patients, Bilateral-Branch SAC eliminates ventricular suction, maintains normotension, and enforces strict flow conservation with sub-millisecond inference suitable for embedded clinical hardware.

---

## Reproduction Guide & Quick Start

### 1. Environment Installation
Clone the repository and install dependencies in a clean Python 3.10+ virtual environment:
```bash
git clone https://github.com/<your-username>/bivad-bilateral-sac.git
cd bivad-bilateral-sac
python -m venv myenv

# Windows
myenv\Scripts\activate

# Linux / macOS
source myenv/bin/activate

pip install -r requirements.txt
```

### 2. Train DRL Policies
All models support Behavioral Cloning (BC) warm-start followed by full SAC training across multiple CPU cores:
```bash
# 1. Train Proposed Bilateral-Branch SAC (Seed 42)
python train_bilateral.py --mode bc --seed 42
python train_bilateral.py --mode full --seed 42 --full_steps 500000 --n_envs 4

# 2. Train Centralized Flat SAC Baseline (Seed 42)
python train_centralized.py --mode bc --seed 42
python train_centralized.py --mode full --seed 42 --full_steps 500000 --n_envs 4

# 3. Train Independent Dual-Agent SAC Baseline (Seed 42)
python train_independent.py --mode bc --seed 42
python train_independent.py --mode full --seed 42 --steps 500000 --n_envs 4
```
*Note: Ready-to-run Jupyter notebooks for free Kaggle 4-CPU / GPU execution are provided in [`kaggle_notebooks/`](kaggle_notebooks/).*

### 3. Evaluation of All Controllers
Benchmark all 7 controllers across the 9 clinical perturbation scenarios:
```bash
# Evaluate classical baselines (Constant Speed, Dual-PI, Starling, CMO-MPC)
python evaluate_controllers.py

# Run comprehensive 9-scenario benchmark across all controllers & seeds
python evaluate_all.py --controllers Constant_Speed,PI,Starling,MPC --scenarios 1,2,3,4,5,6,7,8,9 --parallel

# Evaluate Virtual Patient Cohort (5 phenotypes across Rest and Exercise)
python experiments/virtual_patients/evaluate_virtual_patients.py

# Train and evaluate Reward Component Ablation (w2 = 0 vs baseline)
python experiments/reward_ablation/train_ablation_no_balance.py
```

---

## 5. Repository Directory Structure

```
This Repository
├── controllers/                  # Classical and model-based benchmark controllers
│   ├── __init__.py              # Module exports
│   ├── constant_speed.py        # Clinical standard-of-care fixed-speed controller
│   ├── pi_controller.py         # Decentralized Dual-PI feedback controller (Perra et al. 2025)
│   ├── starling_controller.py   # Preload-sensitive Dual Starling-like controller (Stephens 2017)
│   └── mpc_controller.py        # Centralized Multi-Objective Model Predictive Control (Koh 2020)
│
├── simulator/                    # Coupled lumped-parameter cardiovascular ODE engine
│   ├── __init__.py              # Module exports
│   ├── circulatory_system.py    # 12-state coupled cardiovascular dynamics & valves
│   ├── parameters.py            # Clinical patient parameters (resistances, compliances)
│   └── vads.py                  # HeartWare HVAD rotary blood pump hydraulic equations
│
├── trained_models/               # Pre-trained policy checkpoints (Seed 42, 500k steps)
│   ├── README.md                # Checkpoint loading and inference documentation
│   ├── sac_bilateral.zip        # Proposed Bilateral-Branch SAC policy checkpoint
│   ├── sac_centralized.zip      # Centralized Flat SAC baseline policy checkpoint
│   └── sac_independent/         # Decentralized Independent SAC PyTorch state dicts
│       ├── actor_l_final.pth    # Left pump (LVAD) decentralized actor network
│       └── actor_r_final.pth    # Right pump (RVAD) decentralized actor network
│
├── experiments/                  # Specialized validation and ablation study suites
│   ├── virtual_patients/        # Multi-phenotype virtual patient cohort evaluation
│   │   ├── evaluate_virtual_patients.py # Cohort evaluation script across 5 phenotypes
│   │   ├── table_virtual_patients.csv   # Raw evaluation metrics across 100 episodes
│   │   └── table_virtual_patients.md    # Summary table formatted for manuscript
│   └── reward_ablation/         # Causal reward component ablation (R_balance analysis)
│       ├── train_ablation_no_balance.py # 500k-step ablation training script (w2 = 0)
│       ├── table_reward_ablation.csv    # Comparative metrics across 5 clinical scenarios
│       └── table_reward_ablation.md     # Summary comparison table
│
├── kaggle_notebooks/             # Self-contained Kaggle runner notebooks for cloud training
│   ├── collect_expert_data_colab.ipynb # Generates behavioral cloning demonstrations
│   ├── kaggle_bilateral_runner.ipynb   # 4-core parallel runner for Bilateral SAC
│   ├── kaggle_centralized_runner.ipynb # 4-core parallel runner for Centralized SAC
│   ├── kaggle_independent_runner.ipynb # 4-core parallel runner for Independent SAC
│   └── kaggle_mpc_suction_runner.ipynb # Standalone runner for CMO-MPC Scenario 9
│
├── Docs/                         # High-resolution publication figures (300 DPI)
│   ├── SystemArchitecture.png               # System schematic & network architecture
│   ├── Figure_2_Learning_Curves.png         # Multi-seed training curves with 95% CIs
│   ├── Figure_4_FBE_IAE_Grouped_Bars.png    # Flow balance error & IAE grouped bar charts
│   └── Figure_5_Ablation_Distributions.png  # Performance distribution boxplots across tiers
│
├── bilateral_extractor.py        # PyTorch Bilateral-Branch feature extractor architecture
├── bivad_env.py                  # Farama Gymnasium environment modeling CVS-BiVAD closed-loop
├── reward.py                     # Multi-objective physiological reward function & weights
├── train_bilateral.py            # Training pipeline for proposed Bilateral-Branch SAC
├── train_centralized.py          # Training pipeline for Centralized Flat SAC baseline
├── train_independent.py          # Training pipeline for Independent Dual-Agent SAC baseline
├── evaluate_all.py               # Master multi-controller, multi-scenario evaluation harness
├── evaluate_controllers.py       # Classical controller evaluation benchmark
├── evaluate_independent.py       # Independent dual-agent SAC evaluation pipeline
├── post_training_evaluation.py   # Automated post-training evaluation and trajectory plotting
├── calibrate_rewards.py          # Latin Hypercube sampling reward calibration script
├── compare_all_controllers.py    # Cross-controller comparison summary generator
├── run_mpc_suction.py            # Standalone runner for CMO-MPC under Scenario 9
├── create_kaggle_zip.py          # Packaging script for Kaggle dataset preparation
├── expert_data_pi.npz            # 400,000 expert Dual-PI transitions for BC warm-start
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Git tracking rules excluding local cache and phase files
└── README.md                     # Comprehensive manuscript documentation
```


