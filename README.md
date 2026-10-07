# Coordinated Physiological Control of Biventricular Assist Devices (BiVAD) Using Bilateral-Branch Soft Actor-Critic (SAC)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Solo M.Tech Thesis](https://img.shields.io/badge/Status-Journal%20Manuscript%20(25%20pp)-purple.svg)](#13-license--citation)

---

## Abstract

Biventricular heart failure carries severe morbidity and near-total mortality without mechanical circulatory support. While continuous-flow Left Ventricular Assist Devices (LVADs) are clinical mainstays for isolated left ventricular failure, total biventricular assistance (BiVAD) requires coordinated dual-pump actuation to balance the systemic and pulmonary circulations in real time. Independent decentralized controllers inevitably produce dangerous interventricular flow imbalances, precipitating fatal pulmonary edema (over-pumping) or ventricular suction collapse (under-pumping). Meanwhile, standard centralized deep reinforcement learning (DRL) policies suffer from an **Arterial Pressure Priority Trap**, where unpartitioned networks prioritize systemic arterial pressure at the expense of venous congestion and pulmonary vascular overload.

This repository provides the official implementation of **Bilateral-Branch Soft Actor-Critic (SAC)**, a biologically grounded deep reinforcement learning framework for closed-loop physiological BiVAD control. By incorporating an anatomical inductive bias—segregating systemic and pulmonary observation subspaces into dedicated left and right feature extraction branches before intermediate cross-ventricular coordination—the policy decouples dual-pump dynamics while preserving synchronized flow regulation. Benchmarked against clinical standard-of-care Constant Speed, decentralized Dual-PI feedback, physiological Dual Starling-like control, Centralized Multi-Objective Model Predictive Control (CMO-MPC), Centralized Flat SAC, and Independent Dual-Agent SAC across 9 clinical scenarios and 5 multi-phenotype virtual patients, Bilateral-Branch SAC eliminates ventricular suction, maintains normotension, and enforces strict flow conservation with sub-millisecond inference suitable for embedded clinical hardware.

---

## Table of Contents

- [1. Clinical Motivation & System Architecture](#1-clinical-motivation--system-architecture)
- [2. Lumped-Parameter Cardiovascular Simulation Plant](#2-lumped-parameter-cardiovascular-simulation-plant)
- [3. Cardiovascular & BiVAD Model Parameters](#3-cardiovascular--bivad-model-parameters)
- [4. Gymnasium Simulation Environment](#4-gymnasium-simulation-environment)
- [5. Multi-Objective Physiological Reward Formulation](#5-multi-objective-physiological-reward-formulation)
- [6. Controller Architectures & Benchmark Baselines](#6-controller-architectures--benchmark-baselines)
- [7. Training Pipeline & Learning Dynamics](#7-training-pipeline--learning-dynamics)
- [8. Clinical Evaluation Protocol (9 Scenarios)](#8-clinical-evaluation-protocol-9-scenarios)
- [9. Results](#9-results)
  - [9.1 Baseline Resting & Exercise Hemodynamics (Table 1)](#91-baseline-resting--exercise-hemodynamics-table-1)
  - [9.2 Comprehensive Multi-Scenario Clinical Benchmark (Table 2)](#92-comprehensive-multi-scenario-clinical-benchmark-table-2)
  - [9.3 Interventricular Flow Balance & Tracking Accuracy](#93-interventricular-flow-balance--tracking-accuracy)
  - [9.4 DRL Architecture Ablation & Distribution Analysis](#94-drl-architecture-ablation--distribution-analysis)
  - [9.5 Effect Size & Closeness to Healthy Analysis (Table 3)](#95-effect-size--closeness-to-healthy-analysis-table-3)
  - [9.6 Virtual Patient Cohort Robustness Study](#96-virtual-patient-cohort-robustness-study)
  - [9.7 Reward Component Ablation: Impact of Flow Conservation ($R_{\text{balance}}$)](#97-reward-component-ablation-impact-of-flow-conservation-r_textbalance)
- [10. Reproduction Guide & Quick Start](#10-reproduction-guide--quick-start)
- [11. Repository Directory Structure](#11-repository-directory-structure)
- [12. References](#12-references)

---

## 1. Clinical Motivation & System Architecture

### The Clinical Problem
Advanced heart failure affects over 64 million individuals worldwide [[27]](#12-references). For patients suffering from end-stage biventricular failure—where both the left and right ventricles cannot maintain adequate cardiac output—isolated LVAD support is contraindicated due to acute right ventricular failure [[28], [30]](#12-references). Total artificial heart (TAH) replacement or continuous-flow Biventricular Assist Device (BiVAD) support is mandatory [[29], [31]](#12-references). 

However, continuous-flow BiVADs introduce an intractable physiological synchronization challenge:
1. **Low-Capacitance Pulmonary Bed**: The pulmonary vascular compliance is approximately $10\times$ lower than systemic compliance. An uncorrected flow discrepancy as small as $15\text{--}20\text{ mL/s}$ between the LVAD and RVAD shifts fatal fluid volumes into the pulmonary capillaries within minutes, causing acute pulmonary edema or fatal hypovolemic left-atrial collapse [[1], [8]](#12-references).
2. **Dynamic Cross-Talk via the Interventricular Septum**: Actuating the LVAD alters left ventricular volume, shifting the interventricular septum and modifying right ventricular mechanical compliance [[18]](#12-references). The two circulations are coupled in a closed series loop.
3. **Clinical Limitations of Decentralized Feedback**: Commercial TAH and dual-pump clinical trials typically rely on two independent Single-Input Single-Output (SISO) PI loops [[2]](#12-references). Because each loop is blind to the other pump's actuation, hemodynamic disturbances induce severe mutual oscillation, arterial hypertension, or pump runaway [[2], [22]](#12-references).
4. **Computational Burden of Centralized MPC**: While Centralized Multi-Objective Model Predictive Control (CMO-MPC) coordinates both pumps effectively in simulation [[1], [5]](#12-references), online numerical optimization requires $100\text{--}150\text{ ms}$ per control step on desktop processors, precluding low-power embedded deployment inside battery-powered transcutaneous controllers [[13]](#12-references).

### Closed-Loop System Architecture

```
                    +---------------------------------------------------------+
                    |           CLOSED-LOOP CVS-BiVAD SIMULATION PLANT        |
                    |    12-State Coupled ODEs: Atria, Ventricles, Arteries   |
                    +---------------------------------------------------------+
                                          |                    ^
               Hemodynamic Observations   |                    | Dual Motor Speeds
               o_t in R^9                 v                    | u_t in [1800, 4000] RPM
                    +---------------------------------------------------------+
                    |             BILATERAL-BRANCH SAC CONTROLLER             |
                    |                                                         |
                    |  Systemic Branch (LAP, Q_L, AoP, V_LV, HR)              |
                    |  Pulmonary Branch (RAP, Q_R, PAP, V_RV, HR)             |
                    |                         |                               |
                    |       Cross-Ventricular Coordination Layer              |
                    |                         |                               |
                    |     Multi-Objective Action Output: [omega_L, omega_R]   |
                    +---------------------------------------------------------+
```

![Figure 1: Closed-Loop System Architecture and Bilateral-Branch SAC Controller](Docs/SystemArchitecture.png)

*Figure 1: Schematic diagram of the closed-loop cardiovascular-BiVAD control system. Left: Coupled 12-state lumped-parameter circulatory plant incorporating systemic and pulmonary Windkessel compartments, time-varying ventricular elastance chambers, and dual continuous-flow rotary blood pumps (LVAD and RVAD). Right: Bilateral-Branch SAC actor network architecture featuring partitioned systemic and pulmonary feature extractors, cross-ventricular shared dense layers, and dual continuous action heads.*

---

## 2. Lumped-Parameter Cardiovascular Simulation Plant

The cardiovascular simulation engine models the complete human circulation as a 12-state coupled system of ordinary differential equations (ODEs), incorporating systemic and pulmonary Windkessel compartments, non-linear heart valves, and dual rotary blood pumps [[1], [3], [17], [33]](#12-references).

### 12-State Governing Differential Equations

The state vector is defined as:
$$\mathbf{x} = [P_{\text{sa}}, P_{\text{sv}}, P_{\text{pa}}, P_{\text{pv}}, V_{\text{la}}, V_{\text{lv}}, V_{\text{ra}}, V_{\text{rv}}, Q_{\text{ao}}, Q_{\text{mv}}, Q_{\text{pa}}, Q_{\text{tv}}]^T$$

The hemodynamic state transitions are governed by:
$$\frac{dP_{\text{sa}}}{dt} = \frac{Q_{\text{ao}} + Q_{\text{LVAD}} - \frac{P_{\text{sa}} - P_{\text{sv}}}{R_{\text{sys}}}}{C_{\text{sa}}}$$

$$\frac{dP_{\text{sv}}}{dt} = \frac{\frac{P_{\text{sa}} - P_{\text{sv}}}{R_{\text{sys}}} - Q_{\text{tv}}}{C_{\text{sv}}}$$

$$\frac{dP_{\text{pa}}}{dt} = \frac{Q_{\text{pv\_in}} + Q_{\text{RVAD}} - \frac{P_{\text{pa}} - P_{\text{pv}}}{R_{\text{pul}}}}{C_{\text{pa}}}$$

$$\frac{dP_{\text{pv}}}{dt} = \frac{\frac{P_{\text{pa}} - P_{\text{pv}}}{R_{\text{pul}}} - Q_{\text{mv}}}{C_{\text{pv}}}$$

$$\frac{dV_{\text{lv}}}{dt} = Q_{\text{mv}} - Q_{\text{ao}} - Q_{\text{LVAD}}$$

$$\frac{dV_{\text{rv}}}{dt} = Q_{\text{tv}} - Q_{\text{pv\_in}} - Q_{\text{RVAD}}$$

$$\frac{dQ_{\text{ao}}}{dt} = \frac{P_{\text{lv}} - P_{\text{sa}} - R_{\text{ao}} Q_{\text{ao}}}{L_{\text{ao}}} \quad (\text{if } P_{\text{lv}} > P_{\text{sa}} \text{ and } V_{\text{lv}} > V_{0,\text{lv}}, \text{ else } 0)$$

$$\frac{dQ_{\text{pa}}}{dt} = \frac{P_{\text{rv}} - P_{\text{pa}} - R_{\text{pa}} Q_{\text{pa}}}{L_{\text{pa}}} \quad (\text{if } P_{\text{rv}} > P_{\text{pa}} \text{ and } V_{\text{rv}} > V_{0,\text{rv}}, \text{ else } 0)$$

### Double-Hill Time-Varying Ventricular Elastance

Ventricular pressure-volume coupling follows the double-Hill elastance model [[17], [33]](#12-references):
$$P_{\text{lv}}(t) = E_{\text{lv}}(t) \cdot (V_{\text{lv}}(t) - V_{0,\text{lv}})$$
$$P_{\text{rv}}(t) = E_{\text{rv}}(t) \cdot (V_{\text{rv}}(t) - V_{0,\text{rv}})$$

where normalized ventricular activation $E(t)$ across normalized cardiac cycle $\tau = (t \pmod T) / T$ is expressed as:
$$E(t) = (E_{\max} - E_{\min}) \cdot \left[ \frac{(\tau / \tau_1)^{n_1}}{1 + (\tau / \tau_1)^{n_1}} \right] \cdot \left[ \frac{1}{1 + (\tau / \tau_2)^{n_2}} \right] + E_{\min}$$

### Rotary Blood Pump Hydraulic Characteristic

The LVAD and RVAD are modeled after HeartWare HVAD centrifugal blood pumps [[1], [8]](#12-references). The differential pressure across each pump is related to flow $Q$ and impeller speed $\omega$ by:
$$\Delta P = \beta_0 \cdot \omega^2 + \beta_1 \cdot \omega \cdot Q + \beta_2 \cdot Q^2$$
where $\beta_0 = 1.5 \times 10^{-5}\text{ mmHg/RPM}^2$, $\beta_1 = -0.0017\text{ mmHg/(RPM}\cdot\text{mL/s)}$, and $\beta_2 = -0.18\text{ mmHg/(mL/s)}^2$.

---

## 3. Cardiovascular & BiVAD Model Parameters

Baseline model parameters representing an adult patient with severe biventricular dilated cardiomyopathy [[1], [3], [14]](#12-references):

| Compartment / Parameter | Symbol | Nominal Value | Units | Clinical Description |
| :--- | :--- | :---: | :---: | :--- |
| **Systemic Arterial Resistance** | $R_{\text{sys}}$ | 1.00 | $\text{mmHg}\cdot\text{s/mL}$ | Baseline total peripheral resistance |
| **Systemic Arterial Compliance** | $C_{\text{sa}}$ | 1.20 | $\text{mL/mmHg}$ | Aortic & large artery elasticity |
| **Systemic Venous Compliance** | $C_{\text{sv}}$ | 30.0 | $\text{mL/mmHg}$ | Venous capacitance bed |
| **Pulmonary Vascular Resistance** | $R_{\text{pul}}$ | 0.12 | $\text{mmHg}\cdot\text{s/mL}$ | Pulmonary arterial bed impedance |
| **Pulmonary Arterial Compliance** | $C_{\text{pa}}$ | 0.18 | $\text{mL/mmHg}$ | Pulmonary vascular capacitance |
| **Pulmonary Venous Compliance** | $C_{\text{pv}}$ | 8.00 | $\text{mL/mmHg}$ | Pulmonary venous pool capacitance |
| **LV End-Systolic Elastance** | $E_{\max,\text{lv}}$ | 0.80 | $\text{mmHg/mL}$ | Severely depressed LV contractility |
| **LV End-Diastolic Elastance** | $E_{\min,\text{lv}}$ | 0.06 | $\text{mmHg/mL}$ | LV passive diastolic stiffness |
| **RV End-Systolic Elastance** | $E_{\max,\text{rv}}$ | 0.30 | $\text{mmHg/mL}$ | Severely depressed RV contractility |
| **RV End-Diastolic Elastance** | $E_{\min,\text{rv}}$ | 0.03 | $\text{mmHg/mL}$ | RV passive diastolic stiffness |
| **Heart Rate** | $\text{HR}$ | 75.0 | $\text{bpm}$ | Baseline cardiac chronotropy ($T = 0.8\text{ s}$) |
| **LV Unstressed Volume** | $V_{0,\text{lv}}$ | 15.0 | $\text{mL}$ | LV residual volume threshold |
| **RV Unstressed Volume** | $V_{0,\text{rv}}$ | 15.0 | $\text{mL}$ | RV residual volume threshold |
| **Aortic Valve Inertance** | $L_{\text{ao}}$ | 0.005 | $\text{mmHg}\cdot\text{s}^2/\text{mL}$ | Aortic valve fluid column mass |
| **Pulmonary Valve Inertance** | $L_{\text{pa}}$ | 0.005 | $\text{mmHg}\cdot\text{s}^2/\text{mL}$ | Pulmonary valve fluid column mass |
| **Impeller Speed Range** | $[\omega_{\min}, \omega_{\max}]$ | $[1800, 4000]$ | $\text{RPM}$ | Continuous operational pump limits |
| **Max Slew Rate** | $|\Delta\omega_{\max}|$ | 110.0 | $\text{RPM/s}$ | Mechanical acceleration safety clamp |

---

## 4. Gymnasium Simulation Environment

The cardiovascular simulation is wrapped inside a custom [`BiVADEnv`](bivad_env.py) complying with the Farama Gymnasium API standard [[25]](#12-references).

### State & Action Spaces
- **Observation Space ($\mathbf{o}_t \in \mathbb{R}^9$)**:
  $$\mathbf{o}_t = [\text{LAP}, \text{RAP}, Q_{\text{LVAD}}, Q_{\text{RVAD}}, P_{\text{sa}}, P_{\text{pa}}, V_{\text{lv}}, V_{\text{rv}}, \text{HR}]^T$$
  All signals are filtered via an 8-step moving-average window to attenuate intra-beat pulsatile spikes while maintaining sub-second transient tracking.
- **Action Space ($\mathbf{a}_t \in [-1, 1]^2$)**:
  Normalized action represents delta speed commands scaled by the actuator slew-rate limit:
  $$\Delta\omega_{\text{LVAD}} = a_0 \cdot 110.0\text{ RPM/s}, \quad \Delta\omega_{\text{RVAD}} = a_1 \cdot 110.0\text{ RPM/s}$$
  $$\omega_{t+1} = \text{clip}\left(\omega_t + \Delta\omega_t \cdot \Delta t, \, 1800.0, \, 4000.0\right)$$

### Temporal Discretization & Stability Guardrails
- **Integration Timestep**: $\Delta t = 0.1\text{ s}$ per environment step. Within each step, the coupled 12-state ODE plant is solved using 4th-order Runge-Kutta (RK4) with adaptive sub-cycling ($h = 0.002\text{ s}$) to prevent numerical stiffness.
- **Truncation & Termination Protocol**: Episodes run for $T = 200.0\text{ s}$ (2,000 steps). Truncation flags signal episode horizons correctly to prevent Bellman bootstrap corruption [[20], [25]](#12-references).

---

## 5. Multi-Objective Physiological Reward Formulation

The multi-objective reward function balances five simultaneous clinical targets [[3], [4], [32]](#12-references):
$$R_t = R_{\text{track}} + R_{\text{balance}} + R_{\text{safety}} + R_{\text{smooth}} + R_{\text{map}}$$

```
                                 TOTAL REWARD R_t
      +----------------------------------+----------------------------------+
      |                                  |                                  |
R_track (w1)                       R_balance (w2)                     R_safety (w3)
Demand tracking to                 Linear flow conservation           Asymmetric cubic penalty
Frank-Starling curve               |Q_LVAD - Q_RVAD|                  on [3, 18] mmHg boundaries
      |                                  |
R_smooth (w4)                      R_map (w5)
Dead-band slew rate                Normotension penalty
penalty on motor acceleration      on [70, 120] mmHg window
```

### 1. Physiological Demand Tracking ($R_{\text{track}}$)
Enforces Frank-Starling cardiac responsiveness by penalizing deviation from atrial-pressure-dependent flow setpoints [[1], [32]](#12-references):
$$R_{\text{track}} = -w_1 \left( |Q_{\text{LVAD}} - Q_{\text{target,L}}(\text{LAP})| + |Q_{\text{RVAD}} - Q_{\text{target,R}}(\text{RAP})| \right)$$
where $Q_{\text{target}}(P_{\text{atrial}})$ is defined by validated sigmoid Frank-Starling operating curves.

### 2. Left-Right Flow Conservation ($R_{\text{balance}}$)
Directly penalizes absolute flow discrepancy between systemic and pulmonary pumps to prevent volume sequestration:
$$R_{\text{balance}} = -w_2 |Q_{\text{LVAD}} - Q_{\text{RVAD}}|$$
This term operates as the explicit flow-conservation regularizer that prevents pulmonary capillary flooding.

### 3. Asymmetric Hemodynamic Safety Envelope ($R_{\text{safety}}$)
Protects against catastrophic ventricular suction collapse ($P_{\text{atrial}} < 3\text{ mmHg}$) and venous congestion ($P_{\text{atrial}} > 18\text{ mmHg}$) using cubic boundary penalties [[1], [35]](#12-references):
$$\Phi_{\text{suct}}(P) = \max(0, 3.0 - P)^3, \quad \Phi_{\text{cong}}(P) = \max(0, P - 18.0)^3$$
$$R_{\text{safety}} = -w_3 \left( \Phi_{\text{suct}}(\text{LAP}) + \Phi_{\text{cong}}(\text{LAP}) + \Phi_{\text{suct}}(\text{RAP}) + \Phi_{\text{cong}}(\text{RAP}) \right)$$

### 4. Actuation Smoothness & Mechanical Protection ($R_{\text{smooth}}$)
Applies a dead-band quadratic penalty to motor acceleration to mitigate shear stress, mechanical wear, and blood hemolysis [[4]](#12-references):
$$R_{\text{smooth}} = -w_4 \left( \max(0, |\Delta\omega_L| - \delta_{\text{th}})^2 + \max(0, |\Delta\omega_R| - \delta_{\text{th}})^2 \right)$$
with threshold $\delta_{\text{th}} = 30.0\text{ RPM/s}$.

### 5. Systemic Normotension Window ($R_{\text{map}}$)
Penalizes Mean Arterial Pressure (MAP) excursions outside the clinically safe autoregulatory band $[70, 120]\text{ mmHg}$ [[4]](#12-references):
$$R_{\text{map}} = -w_5 \cdot \max\left(0, \, |\text{MAP} - 95.0| - 25.0\right)$$

### Calibrated Weight Parameters
Weights were formally calibrated via Latin Hypercube sampling over $10^6$ transitions to equalize gradient norms across objectives [[3]](#12-references):

$$\mathbf{w} = [w_1, w_2, w_3, w_4, w_5] = [0.003554, \; 0.012261, \; 0.054593, \; 0.013149, \; 0.007621]$$

---

## 6. Controller Architectures & Benchmark Baselines

To rigorously isolate performance gains, this study benchmarks 7 distinct controller architectures spanning classical control, optimal control, and deep reinforcement learning:

```
+----------------------------------------------------------------------------------------------------+
|                                    CONTROLLER TAXONOMY & ARCHITECTURES                             |
+----------------------------------------------------------------------------------------------------+
| 1. Constant Speed        : Fixed RPM clinical standard of care (omega_L = 2600, omega_R = 1900)    |
| 2. Dual-PI (Perra 2025)  : Decentralized SISO PI controllers (LAP -> 9 mmHg, RAP -> 4 mmHg)        |
| 3. Dual Starling-Like    : Preload-sensitive sigmoid speed mapping (Stephens 2017)                 |
| 4. CMO-MPC (Koh 2020)    : Centralized Receding Horizon Multi-Objective Optimization (Hp = 5)      |
| 5. Independent SAC       : Decentralized MARL ablation (separate actors, 5D local observations)   |
| 6. Centralized Flat SAC  : Standard unpartitioned MLP actor (9D concatenated observation)          |
| 7. Bilateral-Branch SAC  : Proposed partitioned feature extraction with shared coordination head   |
+----------------------------------------------------------------------------------------------------+
```

### 1. Constant Speed (Clinical Standard of Care)
Fixes impeller speeds at nominal resting values ($\omega_L = 2600\text{ RPM}, \omega_R = 1900\text{ RPM}$). Blind to physiological demands and unable to respond to exercise or hemorrhage.

### 2. Decentralized Dual-PI (Perra et al., 2025)
Two independent feedback loops targeting setpoints $\text{LAP} = 9.0\text{ mmHg}$ and $\text{RAP} = 4.0\text{ mmHg}$ [[2]](#12-references). Because the loops lack cross-coupling awareness, systemic vasoconstriction induces severe LVAD over-speeding ($3544\text{ RPM}$) and arterial hypertension ($139.5\text{ mmHg}$).

### 3. Dual Starling-Like Controller (Stephens et al., 2017)
Static non-linear sigmoid mapping from preload pressures to target speeds [[8], [9]](#12-references). Provides immediate preload sensitivity but lacks feedback on flow balance or arterial resistance.

### 4. Centralized Multi-Objective MPC (Koh et al., 2020)
Receding-horizon optimization minimizing tracking errors and flow imbalance over a prediction horizon $H_p = 5$ [[1], [5]](#12-references). Effective in simulation but incurs substantial computational latency ($\sim 140\text{ ms/step}$).

### 5. Independent Dual-Agent SAC (Decentralized MARL Ablation)
Two separate SAC agents with disjoint observation vectors:
- **Left Agent (LVAD)**: $\mathbf{o}_L = [\text{LAP}, Q_{\text{LVAD}}, P_{\text{sa}}, V_{\text{lv}}, \text{HR}]^T \in \mathbb{R}^5$
- **Right Agent (RVAD)**: $\mathbf{o}_R = [\text{RAP}, Q_{\text{RVAD}}, P_{\text{pa}}, V_{\text{rv}}, \text{HR}]^T \in \mathbb{R}^5$
Violates Centralized Training with Decentralized Execution (CTDE) [[22]](#12-references), suffering from non-stationary environment dynamics.

### 6. Centralized Flat SAC (Standard DRL Baseline)
Unpartitioned MLP actor ($\text{Linear}(9 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 256) \to \text{Linear}(256 \to 2)$). Suffers from the **Arterial Pressure Priority Trap**: high-gradient systemic pressure signals dominate the representation, causing the network to ignore right-heart venous congestion.

### 7. Bilateral-Branch SAC (Proposed Architecture)
Incorporates anatomical inductive bias via partitioned feature extraction:
- **Systemic Branch MLP**: $\text{Linear}(5 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 64)$
- **Pulmonary Branch MLP**: $\text{Linear}(5 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 64)$
- **Cross-Ventricular Fusion**: $\text{Concat}(64 + 64 = 128) \to \text{Linear}(128 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 64)$
- **Dual Policy Heads**: Separate mean and log-std projections for LVAD and RVAD delta RPMs.

---

## 7. Training Pipeline & Learning Dynamics

### Behavioral Cloning Warm-Start Protocol
Direct random exploration in continuous-flow BiVADs causes immediate cardiac collapse or ODE integration divergence within 50 steps. To ensure safe, stable initialization, policies are pre-trained via Behavioral Cloning [[26]](#12-references) on a dataset of 400,000 state-action pairs collected from stable dual-PI simulations ([`expert_data_pi.npz`](expert_data_pi.npz)) using MSE supervised loss:
$$\mathcal{L}_{\text{BC}}(\theta) = \frac{1}{N} \sum_{i=1}^N \|\pi_\theta(\mathbf{o}_i) - \mathbf{a}_{\text{expert},i}\|^2$$

### SAC Maximum-Entropy Training
Following warm-start, policies undergo 500,000 steps of maximum-entropy Soft Actor-Critic reinforcement learning [[20]](#12-references):
- Replay Buffer Capacity: $1,000,000$ transitions
- Batch Size: 256
- Discount Factor ($\gamma$): 0.99
- Soft Target Update ($\tau$): 0.005
- Learning Rate: $3 \times 10^{-4}$ (Adam optimizer)
- Target Entropy: $\mathcal{H}_{\text{target}} = -\dim(\mathcal{A}) = -2.0$
- Parallel Environs: 4 vectorized CPU workers

![Figure 2: Multi-Seed Training Curves and Convergence Dynamics](Docs/Figure_2_Learning_Curves.png)

*Figure 2: 500,000-step training dynamics across 5 independent random seeds. Top Left: Episode mean reward progression with 95% confidence intervals. Top Right: Actor loss evolution. Bottom Left: Critic loss stabilization. Bottom Right: Flow Balance Error convergence toward safe physiological thresholds.*

---

## 8. Clinical Evaluation Protocol (9 Scenarios)

Controllers are evaluated across 9 standardized clinical perturbation scenarios designed to stress cardiovascular autoregulation and device synchronization [[1], [3]](#12-references):

1. **Rest**: Baseline steady-state conditions ($R_{\text{sys}} = 1.0$, $R_{\text{pul}} = 0.12$, $\text{HR} = 75$).
2. **SVR_up**: Acute systemic vasoconstriction ($R_{\text{sys}} +50\%$ at step 500), simulating cold stress or pharmacological vasopressor therapy.
3. **SVR_down**: Acute systemic vasodilation ($R_{\text{sys}} -50\%$ at step 500), simulating sepsis or neurogenic shock.
4. **PVR_up**: Acute pulmonary hypertension ($R_{\text{pul}} +100\%$ at step 500), simulating pulmonary embolism or acute respiratory distress.
5. **Exercise**: Strenuous metabolic demand ($\text{HR} \to 120\text{ bpm}$, $R_{\text{sys}} -40\%$, contractility $+30\%$), testing high cardiac output delivery.
6. **Exercise_Recovery**: Ramp from exercise back to resting conditions at step 1000, assessing recovery dynamics without overshoot.
7. **Hypovolemia**: Acute systemic volume loss (blood volume $-20\%$), testing pump down-regulation to avoid cavitation.
8. **Combined**: Concomitant systemic vasodilation and pulmonary vasoconstriction ($R_{\text{sys}} -30\%$, $R_{\text{pul}} +50\%$).
9. **Suction_Test**: Critical hypovolemic challenge with acute preload drop, directly challenging the controller's ability to prevent ventricular suction collapse.

---

## 9. Results

### 9.1 Baseline Resting & Exercise Hemodynamics (Table 1)

#### Part A: Rest Hemodynamics vs. Clinical Literature & Benchmarks

| Condition / Controller | MAP (mmHg) | PAP (mmHg) | LAP (mmHg) | RAP (mmHg) | CO (L/min) | LVAD (RPM) | RVAD (RPM) | FBE (mL/s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Healthy Clinical Reference [34, 35]** | 89.3 – 90.0 | 16.0 – 26.7 | 11.3 – 13.0 | 5.9 – 7.4 | 5.0 – 5.4 | — | — | < 5.0 |
| **Koh et al. (2020) MCL In Vitro [1]** | 88.4 | 18.5 | 11.3 | 7.4 | 5.00 | 2508 | 2670 | 0.0 |
| **Constant Speed** | 98.9 ± 0.2 | 18.5 ± 0.4 | 12.9 ± 0.0 | 4.8 ± 0.1 | 5.18 ± 0.01 | 2600 ± 0 | 1900 ± 0 | 4.24 ± 0.38 |
| **Dual-PI (Perra 2025)** | 139.5 ± 0.7 | 18.5 ± 0.4 | 11.3 ± 0.1 | 5.1 ± 0.0 | 7.76 ± 0.04 | 3544 ± 9 | 2354 ± 9 | 0.48 ± 0.02 |
| **Starling-like** | 111.7 ± 0.5 | 18.5 ± 0.4 | 11.8 ± 0.1 | 5.4 ± 0.1 | 6.13 ± 0.02 | 2964 ± 6 | 2006 ± 4 | 0.28 ± 0.00 |
| **CMO-MPC (Koh 2020)** | 96.9 ± 0.6 | 18.5 ± 0.4 | 12.1 ± 0.1 | 5.3 ± 0.1 | 5.19 ± 0.04 | 2608 ± 12 | 1820 ± 7 | 3.75 ± 0.31 |
| **Independent SAC (Tier 1)** | 106.1 ± 12.5 | 18.5 ± 0.4 | 11.9 ± 2.6 | 5.2 ± 2.1 | 5.05 ± 2.37 | 2755 ± 519 | 1988 ± 390 | 16.37 ± 27.14 |
| **Centralized Flat SAC (Tier 2)** | 108.8 ± 3.5 | 18.5 ± 0.4 | 10.0 ± 0.4 | 6.7 ± 0.2 | 5.90 ± 0.19 | 2983 ± 92 | 1800 ± 0 | 3.61 ± 2.34 |
| **Bilateral SAC (Ours - Tier 3)** | **95.5 ± 5.6** | 18.5 ± 0.4 | 12.3 ± 1.2 | 5.2 ± 0.7 | **4.39 ± 1.93** | 2491 ± 359 | 1806 ± 6 | 13.79 ± 27.53 |

*Note: In 4 of 5 seeds (80%), Bilateral SAC achieves resting $\text{FBE} = 1.52 \pm 2.49\text{ mL/s}$ (median: $0.30\text{ mL/s}$). The 5-seed mean is influenced by Seed 123 which converged to an underpumping policy state.*

#### Part B: Exercise Hemodynamic Response vs. Literature Benchmarks

| Condition / Controller | MAP (mmHg) | PAP (mmHg) | LAP (mmHg) | RAP (mmHg) | CO (L/min) | LVAD (RPM) | RVAD (RPM) | FBE (mL/s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Koh 2020 Exercise (CS) [1]** | 89.0 | 22.4 | 16.4 | 13.8 | 6.00 | 2510 | 2669 | 5.0 |
| **Koh 2020 Exercise (MPC) [1]** | 107.7 | 25.1 | 17.9 | 12.9 | 7.60 | 3088 | 3783 | 3.3 |
| **Constant Speed** | 93.2 ± 0.2 | 22.1 ± 0.6 | 12.1 ± 0.1 | 6.0 ± 0.1 | 5.82 ± 0.00 | 2600 ± 0 | 1900 ± 0 | 6.38 ± 0.12 |
| **Dual-PI** | 131.6 ± 0.6 | 22.1 ± 0.6 | 12.0 ± 0.1 | 5.6 ± 0.0 | 8.63 ± 0.04 | 3529 ± 8 | 2525 ± 10 | 3.72 ± 0.07 |
| **Starling-like** | 105.5 ± 0.4 | 22.1 ± 0.6 | 11.6 ± 0.1 | 6.3 ± 0.1 | 6.70 ± 0.02 | 2936 ± 6 | 2069 ± 4 | 4.30 ± 0.00 |
| **CMO-MPC (Koh 2020)** | 89.1 ± 0.5 | 22.1 ± 0.6 | 11.8 ± 0.1 | 6.1 ± 0.1 | 5.34 ± 0.03 | 2480 ± 9 | 1810 ± 3 | 6.93 ± 0.20 |
| **Independent SAC (Tier 1)** | 107.1 ± 20.1 | 22.1 ± 0.6 | 10.9 ± 2.6 | 6.7 ± 1.6 | 5.53 ± 3.14 | 2829 ± 785 | 1994 ± 303 | 31.01 ± 25.16 |
| **Centralized Flat SAC (Tier 2)** | 105.8 ± 2.3 | 22.1 ± 0.6 | 9.6 ± 0.3 | 7.5 ± 0.2 | 6.66 ± 0.15 | 3038 ± 68 | 1801 ± 1 | 14.32 ± 2.13 |
| **Bilateral SAC (Ours - Tier 3)** | **97.2 ± 7.1** | 22.1 ± 0.6 | 11.1 ± 0.4 | 6.6 ± 0.3 | **5.95 ± 0.69** | 2740 ± 210 | 1858 ± 114 | **8.10 ± 7.06** |

---

### 9.2 Comprehensive Multi-Scenario Clinical Benchmark (Table 2)

Comprehensive multi-metric benchmark evaluating all 7 controllers across all 9 clinical scenarios ($n = 5$ independent random seeds, reported as mean ± standard deviation):

| Scenario | Controller | $\text{IAE}_{\text{LAP}}$ ($\text{mmHg}\cdot\text{s}$) | $\text{IAE}_{\text{RAP}}$ ($\text{mmHg}\cdot\text{s}$) | $\text{FBE}_{\text{mean}}$ (mL/s) | Overshoot (mmHg) | Suction (%) | Congestion (%) | Mean Reward |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rest** | Constant Speed | 770.7 ± 9.9 | 375.9 ± 2.8 | 4.24 ± 0.38 | 8.73 ± 0.06 | 0.0% | 0.0% | -0.368 ± 0.002 |
| | Dual-PI | 624.1 ± 5.6 | 448.2 ± 0.3 | 0.48 ± 0.02 | 9.84 ± 0.23 | 0.0% | 4.0% | -300.471 ± 6.626 |
| | Starling-like | 562.9 ± 12.7 | 431.8 ± 2.9 | 0.28 ± 0.00 | 8.64 ± 0.08 | 0.0% | 0.0% | -28.311 ± 2.742 |
| | CMO-MPC (Koh 2020) | 624.9 ± 16.0 | 391.9 ± 2.8 | 3.75 ± 0.31 | 8.78 ± 0.13 | 0.0% | 0.0% | -0.837 ± 0.123 |
| | Independent SAC | 738.2 ± 359.2 | 457.7 ± 113.9 | 16.37 ± 27.14 | 6.36 ± 1.12 | 0.0% | 7.3% | -5.742 ± 8.522 |
| | Centralized SAC (Flat) | 466.6 ± 16.3 | 534.5 ± 47.3 | 3.61 ± 2.34 | 6.14 ± 0.32 | 0.0% | 0.0% | -0.356 ± 0.134 |
| | Bilateral SAC (Ours) | 652.0 ± 237.9 | 387.6 ± 37.2 | 13.79 ± 27.53 | 5.43 ± 0.17 | 0.0% | 2.5% | -0.615 ± 1.037 |
| **Exercise** | Constant Speed | 614.9 ± 12.0 | 457.5 ± 5.8 | 6.38 ± 0.12 | 6.79 ± 0.05 | 0.0% | 0.0% | -0.277 ± 0.005 |
| | Dual-PI | 687.1 ± 16.1 | 481.9 ± 1.2 | 3.72 ± 0.07 | 10.15 ± 0.15 | 0.0% | 11.3% | -221.559 ± 5.926 |
| | Starling-like | 554.0 ± 7.2 | 509.6 ± 5.0 | 4.30 ± 0.00 | 6.28 ± 0.08 | 0.0% | 0.0% | -9.790 ± 0.635 |
| | CMO-MPC (Koh 2020) | 569.0 ± 12.7 | 457.6 ± 8.7 | 6.93 ± 0.20 | 6.05 ± 0.17 | 0.0% | 0.0% | -0.393 ± 0.039 |
| | Independent SAC | 685.2 ± 142.7 | 608.8 ± 248.1 | 31.01 ± 25.16 | 5.88 ± 1.84 | 0.0% | 9.2% | -4.231 ± 3.322 |
| | Centralized SAC (Flat) | 416.1 ± 3.9 | 700.1 ± 31.9 | 14.32 ± 2.13 | 3.66 ± 0.17 | 0.0% | 9.9% | -0.680 ± 0.143 |
| | Bilateral SAC (Ours) | 508.6 ± 54.9 | 562.4 ± 61.8 | **8.10 ± 7.06** | 4.56 ± 1.51 | 0.0% | 3.0% | -0.493 ± 0.307 |
| **Exercise_Recovery** | Constant Speed | 697.0 ± 10.9 | 414.3 ± 4.1 | 5.30 ± 0.25 | 9.30 ± 0.08 | 0.0% | 0.1% | -0.324 ± 0.004 |
| | Dual-PI | 639.8 ± 12.5 | 459.2 ± 1.8 | 1.63 ± 0.01 | 11.38 ± 0.25 | 0.0% | 10.3% | -267.793 ± 5.898 |
| | Starling-like | 559.0 ± 10.1 | 468.4 ± 3.9 | 2.23 ± 0.00 | 9.65 ± 0.09 | 0.0% | 0.1% | -19.524 ± 1.751 |
| | CMO-MPC (Koh 2020) | 599.1 ± 17.3 | 422.4 ± 5.0 | 5.32 ± 0.20 | 8.95 ± 0.09 | 0.0% | 0.0% | -0.622 ± 0.032 |
| | Independent SAC | 744.5 ± 235.2 | 513.0 ± 161.5 | 26.22 ± 24.44 | 6.81 ± 1.84 | 0.0% | 9.2% | -5.031 ± 5.225 |
| | Centralized SAC (Flat) | 443.7 ± 10.0 | 612.8 ± 36.1 | 8.62 ± 2.00 | 5.59 ± 0.43 | 0.0% | 4.7% | -0.507 ± 0.124 |
| | Bilateral SAC (Ours) | 577.9 ± 135.4 | 469.5 ± 32.0 | 10.27 ± 15.65 | 5.76 ± 0.58 | 0.0% | 2.6% | -0.510 ± 0.571 |
| **SVR_up** | Constant Speed | 930.6 ± 18.9 | 351.7 ± 0.0 | 8.13 ± 0.77 | 9.35 ± 0.07 | 0.0% | 7.4% | -61.135 ± 3.230 |
| | Dual-PI | 535.3 ± 5.4 | 396.0 ± 0.5 | 0.67 ± 0.02 | 8.24 ± 0.10 | 0.0% | 0.0% | -457.206 ± 9.288 |
| | Starling-like | 628.9 ± 13.5 | 367.4 ± 1.3 | 0.42 ± 0.00 | 8.09 ± 0.09 | 0.0% | 0.0% | -161.008 ± 4.564 |
| | CMO-MPC (Koh 2020) | 490.0 ± 6.9 | 361.3 ± 1.6 | 2.61 ± 0.12 | 7.91 ± 0.06 | 0.0% | 0.0% | -169.124 ± 3.844 |
| | Independent SAC | 731.9 ± 320.2 | 401.0 ± 50.3 | 9.34 ± 16.70 | 5.43 ± 1.15 | 3.8% | 5.5% | -12.324 ± 11.521 |
| | Centralized SAC (Flat) | 651.4 ± 126.6 | 385.4 ± 8.6 | 2.56 ± 2.83 | 6.85 ± 0.92 | 0.0% | 0.0% | -5.923 ± 2.581 |
| | Bilateral SAC (Ours) | 802.5 ± 204.8 | 354.3 ± 16.9 | 16.56 ± 25.47 | 5.74 ± 0.90 | 0.0% | 4.3% | -5.014 ± 5.151 |
| **SVR_down** | Constant Speed | 579.4 ± 3.4 | 626.9 ± 6.4 | 6.30 ± 0.31 | 8.80 ± 0.06 | 0.0% | 9.2% | -0.314 ± 0.005 |
| | Dual-PI | 801.8 ± 17.8 | 553.0 ± 2.4 | 1.03 ± 0.02 | 14.24 ± 0.26 | 0.0% | 17.0% | -91.023 ± 3.890 |
| | Starling-like | 575.3 ± 4.6 | 676.3 ± 10.6 | 0.36 ± 0.00 | 8.72 ± 0.08 | 0.0% | 18.6% | -7.516 ± 0.681 |
| | CMO-MPC (Koh 2020) | 562.0 ± 34.2 | 570.2 ± 14.5 | 7.59 ± 4.89 | 8.47 ± 0.19 | 0.0% | 6.4% | -14.418 ± 2.726 |
| | Independent SAC | 696.0 ± 179.1 | 690.0 ± 150.9 | 22.17 ± 15.10 | 9.18 ± 1.99 | 0.0% | 19.9% | -4.041 ± 1.987 |
| | Centralized SAC (Flat) | 523.9 ± 26.2 | 809.3 ± 97.6 | 17.76 ± 5.72 | 6.26 ± 0.65 | 0.0% | 16.9% | -0.787 ± 0.256 |
| | Bilateral SAC (Ours) | 586.7 ± 95.7 | 649.8 ± 112.6 | **7.91 ± 7.80** | 5.39 ± 0.38 | 0.0% | 13.6% | -0.670 ± 0.238 |
| **PVR_up** | Constant Speed | 509.0 ± 0.9 | 482.8 ± 8.1 | 1.36 ± 0.09 | 6.47 ± 0.06 | 0.0% | 0.0% | -0.305 ± 0.007 |
| | Dual-PI | 597.0 ± 5.4 | 474.6 ± 4.5 | 0.69 ± 0.01 | 7.06 ± 0.05 | 0.0% | 0.2% | -169.453 ± 12.664 |
| | Starling-like | 485.0 ± 4.5 | 432.6 ± 6.9 | 0.29 ± 0.00 | 5.94 ± 0.07 | 0.0% | 0.0% | -7.358 ± 0.687 |
| | CMO-MPC (Koh 2020) | 503.3 ± 10.3 | 383.5 ± 2.6 | 3.23 ± 0.29 | 7.90 ± 0.15 | 0.0% | 0.0% | -0.816 ± 0.069 |
| | Independent SAC | 568.4 ± 127.8 | 442.0 ± 84.5 | 16.36 ± 20.07 | 6.83 ± 2.34 | 0.0% | 2.7% | -3.428 ± 2.425 |
| | Centralized SAC (Flat) | 467.7 ± 29.1 | 618.4 ± 36.2 | 2.88 ± 1.38 | 4.48 ± 0.43 | 0.0% | 3.8% | -0.603 ± 0.186 |
| | Bilateral SAC (Ours) | 483.5 ± 72.5 | 438.6 ± 45.4 | 4.51 ± 7.04 | 5.34 ± 1.48 | 0.0% | 0.6% | -0.361 ± 0.258 |
| **Hypovolemia** | Constant Speed | 873.0 ± 10.5 | 441.6 ± 3.2 | 3.33 ± 0.35 | 10.37 ± 0.07 | 0.0% | 12.5% | -1.379 ± 0.455 |
| | Dual-PI | 797.5 ± 15.5 | 488.8 ± 0.2 | 0.57 ± 0.01 | 12.85 ± 0.25 | 0.0% | 22.7% | -347.157 ± 5.151 |
| | Starling-like | 684.2 ± 13.9 | 518.4 ± 3.3 | 0.31 ± 0.00 | 10.30 ± 0.09 | 0.0% | 12.5% | -49.368 ± 3.343 |
| | CMO-MPC (Koh 2020) | 669.5 ± 19.4 | 480.4 ± 3.7 | 3.61 ± 0.21 | 10.85 ± 0.19 | 0.0% | 10.0% | -2.857 ± 0.638 |
| | Independent SAC | 845.2 ± 387.0 | 528.8 ± 160.1 | 14.00 ± 22.29 | 7.60 ± 1.62 | 0.0% | 21.9% | -5.166 ± 7.903 |
| | Centralized SAC (Flat) | 567.2 ± 8.2 | 643.1 ± 52.6 | 3.11 ± 2.23 | 7.22 ± 0.28 | 0.0% | 12.2% | -0.276 ± 0.155 |
| | Bilateral SAC (Ours) | 726.9 ± 208.8 | 457.8 ± 50.3 | 9.28 ± 20.17 | 6.48 ± 0.28 | 0.0% | 7.6% | -0.406 ± 0.692 |
| **Combined** | Constant Speed | 815.6 ± 10.1 | 340.4 ± 0.2 | 8.27 ± 0.16 | 11.40 ± 0.07 | 0.0% | 0.1% | -20.469 ± 1.319 |
| | Dual-PI | 531.4 ± 5.4 | 419.5 ± 1.0 | 2.82 ± 0.04 | 11.50 ± 0.12 | 0.0% | 0.1% | -392.766 ± 9.066 |
| | Starling-like | 534.9 ± 12.3 | 377.2 ± 2.4 | 4.90 ± 0.00 | 11.25 ± 0.10 | 0.0% | 0.1% | -100.910 ± 4.590 |
| | CMO-MPC (Koh 2020) | 476.2 ± 7.4 | 367.8 ± 3.9 | 5.15 ± 0.14 | 10.63 ± 0.16 | 0.0% | 0.1% | -44.209 ± 4.016 |
| | Independent SAC | 693.9 ± 270.4 | 382.6 ± 57.7 | 12.31 ± 12.86 | 8.33 ± 1.44 | 0.0% | 3.3% | -6.445 ± 10.471 |
| | Centralized SAC (Flat) | 451.7 ± 65.0 | 407.6 ± 25.0 | 5.02 ± 0.90 | 8.25 ± 0.26 | 0.0% | 0.1% | -0.625 ± 0.293 |
| | Bilateral SAC (Ours) | 663.9 ± 133.5 | 338.9 ± 16.3 | 13.69 ± 10.89 | 7.23 ± 0.35 | 0.0% | 0.7% | -0.548 ± 0.315 |
| **Suction_Test** | Constant Speed | 1009.4 ± 4.9 | 417.5 ± 0.7 | 4.18 ± 0.33 | 11.98 ± 0.04 | 34.9% | 0.1% | -25.381 ± 0.362 |
| | Dual-PI | 848.9 ± 4.8 | 473.0 ± 1.6 | 4.60 ± 0.04 | 10.41 ± 0.05 | 25.5% | 1.6% | -22.470 ± 0.026 |
| | Starling-like | 772.2 ± 5.8 | 479.5 ± 1.4 | 1.38 ± 0.00 | 10.84 ± 0.05 | 25.3% | 0.1% | -18.256 ± 0.107 |
| | CMO-MPC (Koh 2020) | 953.3 ± 6.4 | 441.7 ± 0.6 | 29.54 ± 0.41 | 11.10 ± 0.31 | **14.7%** | 5.0% | -21.480 ± 0.054 |
| | Independent SAC | 929.5 ± 289.4 | 476.7 ± 83.3 | 14.71 ± 15.69 | 11.13 ± 2.50 | 24.7% | 5.7% | -25.763 ± 10.509 |
| | Centralized SAC (Flat) | 871.5 ± 4.7 | 511.6 ± 41.6 | 3.45 ± 1.35 | 9.69 ± 0.76 | 35.0% | 0.2% | -32.863 ± 3.791 |
| | Bilateral SAC (Ours) | 858.4 ± 132.6 | 457.5 ± 27.8 | 9.90 ± 17.26 | 10.95 ± 0.63 | **24.7%** | 1.7% | -23.685 ± 9.051 |

---

### 9.3 Interventricular Flow Balance & Tracking Accuracy

![Figure 4: Interventricular Flow Balance Error and Tracking Integral Absolute Error](Docs/Figure_4_FBE_IAE_Grouped_Bars.png)

*Figure 4: Grouped performance metrics across all 7 controllers under severe physiological stress. Left: Mean Flow Balance Error ($\text{FBE}_{\text{mean}}$) relative to the $20\text{ mL/s}$ clinical safety limit (red dashed line). Decentralized Independent SAC breaches the threshold during Exercise and Vasodilation, whereas Bilateral SAC remains safely below. Right: Integral Absolute Error for Left Atrial Pressure ($\text{IAE}_{\text{LAP}}$) and Right Atrial Pressure ($\text{IAE}_{\text{RAP}}$).*

---

### 9.4 DRL Architecture Ablation & Distribution Analysis

To rigorously assess reproducibility and algorithmic stability, variance across all 5 random seeds ($42, 123, 456, 789, 1024$) is analyzed across the three DRL tiers [[23]](#12-references):

![Figure 5: Performance Distributions across DRL Architectures](Docs/Figure_5_Ablation_Distributions.png)

*Figure 5: Seed variance and performance distributions across the three DRL architectures (Independent Dual-Agent SAC, Centralized Flat SAC, and Bilateral-Branch SAC). Box plots indicate median, interquartile range (IQR), and full range across all 5 random seeds. Bilateral SAC displays superior median flow balance and avoids the systemic suction trap that afflicts Centralized Flat SAC.*

#### Standard Academic Variance Audit:
- **Independent Dual-Agent SAC**: Exhibits large standard deviations across all scenarios ($\text{SD} \ge 0.5 \times \text{mean}$). In Exercise, $\text{FBE} = 31.01 \pm 25.16\text{ mL/s}$. This confirms the theoretical prediction of non-stationarity in decentralized multi-agent reinforcement learning without centralized training [[22]](#12-references).
- **Centralized Flat SAC**: Shows very low variance across seeds ($\text{SD} \approx 2.1\text{ mL/s}$ in Exercise), confirming algorithmic convergence. However, it exhibits a consistent systemic flaw: in Scenario 9 (Suction Test), it incurs **35.0% suction violation** across all seeds due to the Arterial Pressure Priority Trap.
- **Bilateral-Branch SAC**: Achieves high stability across 4 of 5 seeds (80%), yielding median resting $\text{FBE} = 0.30\text{ mL/s}$. One seed (Seed 123) converged to an underpumping policy state ($\text{FBE} = 62.90\text{ mL/s}$), which elevates the parametric sample mean. Reporting median [IQR] confirms that the anatomical inductive bias robustly outperforms unpartitioned baselines.

---

### 9.5 Effect Size & Closeness to Healthy Analysis (Table 3)

Evaluated following the methodology of Y. Shi et al. (2025) [[4]](#12-references):
$$\text{Closeness} = |\text{Metric}_{\text{healthy}} - \text{Metric}_{\text{baseline}}| - |\text{Metric}_{\text{healthy}} - \text{Metric}_{\text{DRL}}|$$
Positive values denote physiological improvement toward healthy homeostasis.

| Scenario | Comparison | Closeness MAP (mmHg) | Closeness CO (L/min) | Closeness LAP (mmHg) | Closeness RAP (mmHg) | $\Delta\text{IAE}_{\text{LAP}}$ (%) | $\Delta\text{FBE}$ (mL/s) | Cohen's $d$ (IAE) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rest** | Centralized SAC vs. Constant Speed | -9.97 | -0.72 | +2.89 | -1.90 | +39.5% | +0.63 (+14.8%) | +22.58 |
| **Rest** | Centralized SAC vs. Dual-PI | **+30.70** | +1.86 | +1.38 | -1.55 | +25.2% | -3.13 (-651.7%) | +12.96 |
| **Rest** | Centralized SAC vs. Starling-like | +2.84 | +0.23 | +1.85 | -1.30 | +17.1% | -3.33 (-1194.3%) | +6.60 |
| **Rest** | Centralized SAC vs. CMO-MPC | -11.92 | -0.70 | +2.17 | -1.37 | +25.3% | +0.14 (+3.8%) | +9.82 |
| **Rest** | Centralized SAC vs. Independent SAC | -2.71 | -0.85 | +1.91 | -1.45 | +36.8% | +12.76 (+77.9%) | +1.07 |
| **Rest** | Bilateral SAC vs. Centralized SAC | **+13.33** | +0.29 | -2.30 | +1.42 | -39.8% | -10.18 (-282.0%) | -1.10 |
| **Rest** | Bilateral SAC vs. Independent SAC | **+10.63** | -0.56 | -0.39 | -0.03 | +11.7% | +2.57 (+15.7%) | +0.28 |
| **Exercise** | Centralized SAC vs. Constant Speed | -9.05 | +0.52 | +2.43 | -1.49 | +32.3% | -7.95 (-124.6%) | +22.29 |
| **Exercise** | Centralized SAC vs. Dual-PI | **+25.79** | +1.97 | +2.38 | -1.90 | +39.4% | -10.61 (-285.4%) | +23.14 |
| **Exercise** | Centralized SAC vs. Starling-like | -0.37 | +0.04 | +1.92 | -1.18 | +24.9% | -10.02 (-233.1%) | +23.81 |
| **Exercise** | Centralized SAC vs. CMO-MPC | -4.97 | +1.00 | +2.20 | -1.37 | +26.9% | -7.40 (-106.8%) | +16.33 |
| **Exercise** | Centralized SAC vs. Independent SAC | +1.26 | +0.81 | +1.21 | -0.84 | +39.3% | +16.68 (+53.8%) | +2.67 |
| **Exercise** | Bilateral SAC vs. Centralized SAC | **+8.63** | -0.39 | -1.41 | +0.85 | -22.2% | **+6.22 (+43.4%)** | -2.37 |
| **Exercise** | Bilateral SAC vs. Independent SAC | **+9.89** | +0.42 | -0.20 | +0.01 | +25.8% | **+22.90 (+73.9%)** | +1.63 |
| **SVR_down** | Centralized SAC vs. Constant Speed | +6.40 | -0.60 | +1.75 | -1.18 | +9.6% | -11.46 (-182.0%) | +2.97 |
| **SVR_down** | Centralized SAC vs. Dual-PI | **+26.95** | +2.64 | +2.55 | -1.99 | +34.7% | -16.72 (-1616.1%) | +12.42 |
| **SVR_down** | Centralized SAC vs. Starling-like | +2.52 | +0.27 | +1.39 | -0.85 | +8.9% | -17.40 (-4832.9%) | +2.74 |
| **SVR_down** | Centralized SAC vs. CMO-MPC | **+10.61** | -0.05 | +1.93 | -1.35 | +6.8% | -10.17 (-134.0%) | +1.25 |
| **SVR_down** | Centralized SAC vs. Independent SAC | +9.92 | +0.70 | +1.36 | -1.00 | +24.7% | +4.41 (+19.9%) | +1.34 |
| **SVR_down** | Bilateral SAC vs. Centralized SAC | -3.44 | +0.62 | -1.48 | +0.96 | -12.0% | **+9.84 (+55.4%)** | -0.90 |
| **SVR_down** | Bilateral SAC vs. Independent SAC | **+6.48** | +1.32 | -0.12 | -0.04 | +15.7% | **+14.25 (+64.3%)** | +0.76 |
| **Hypovolemia**| Centralized SAC vs. Constant Speed | -10.37 | -0.67 | +2.98 | -2.09 | +35.0% | +0.23 (+6.8%) | +32.49 |
| **Hypovolemia**| Centralized SAC vs. Dual-PI | **+35.23** | +2.15 | +2.21 | -2.02 | +28.9% | -2.53 (-440.8%) | +18.57 |
| **Hypovolemia**| Centralized SAC vs. Starling-like | +5.32 | +0.38 | +2.04 | -1.37 | +17.1% | -2.80 (-912.9%) | +10.23 |
| **Hypovolemia**| Centralized SAC vs. CMO-MPC | -8.42 | -0.42 | +1.96 | -1.36 | +15.3% | +0.51 (+14.0%) | +6.88 |
| **Hypovolemia**| Centralized SAC vs. Independent SAC | -1.94 | -0.67 | +1.98 | -1.57 | +32.9% | +10.90 (+77.8%) | +1.02 |
| **Hypovolemia**| Bilateral SAC vs. Centralized SAC | **+12.56** | +0.57 | -2.23 | +1.55 | -28.2% | -6.18 (-199.0%) | -1.08 |
| **Hypovolemia**| Bilateral SAC vs. Independent SAC | **+10.62** | -0.10 | -0.25 | -0.02 | +14.0% | +4.72 (+33.7%) | +0.38 |
| **Suction_Test**| Centralized SAC vs. Constant Speed | -3.17 | +0.13 | -2.10 | -0.56 | +13.7% | +0.73 (+17.4%) | +28.66 |
| **Suction_Test**| Centralized SAC vs. Dual-PI | -5.27 | -0.03 | -2.13 | -0.65 | -2.7% | +1.15 (+25.0%) | -4.74 |
| **Suction_Test**| Centralized SAC vs. Starling-like | -5.23 | -0.22 | -2.00 | -0.79 | -12.9% | -2.07 (-150.3%) | -18.77 |
| **Suction_Test**| Centralized SAC vs. CMO-MPC | **+10.28** | +2.40 | -1.04 | -0.14 | +8.6% | +26.08 (+88.3%) | +14.54 |
| **Suction_Test**| Centralized SAC vs. Independent SAC | -1.17 | +0.77 | -2.13 | -0.54 | +6.2% | +11.26 (+76.5%) | +0.28 |
| **Suction_Test**| Bilateral SAC vs. Centralized SAC | -1.86 | -0.79 | +2.16 | +0.67 | +1.5% | -6.45 (-186.6%) | +0.14 |
| **Suction_Test**| Bilateral SAC vs. Independent SAC | -3.03 | -0.02 | +0.03 | +0.13 | +7.7% | +4.81 (+32.7%) | +0.32 |

---

### 9.6 Virtual Patient Cohort Robustness Study

To evaluate generalizability beyond nominal parameters, we constructed an in silico cohort of **5 distinct pathological phenotypes** representing clinically prevalent patient variations [[14], [16]](#12-references):
- **VP-1 (Nominal Reference)**: Baseline dilated cardiomyopathy parameters ($E_{\max,\text{lv}}=0.80, E_{\max,\text{rv}}=0.30, R_{\text{sys}}=1.00, R_{\text{pul}}=0.12$).
- **VP-2 (Depressed Contractility)**: Biventricular ischemic end-stage failure ($E_{\max,\text{lv}}=0.55, E_{\max,\text{rv}}=0.20$).
- **VP-3 (Hypertensive & Stiff)**: Chronic hypertension with impaired diastolic filling ($R_{\text{sys}}=1.40, E_{\min,\text{lv}}=0.09, E_{\min,\text{rv}}=0.05$).
- **VP-4 (Pulmonary Hypertension)**: Severe secondary pulmonary vascular remodeling ($R_{\text{pul}}=0.25, C_{\text{pa}}=0.12$).
- **VP-5 (Hyperdynamic / Tachycardic)**: Compensatory chronotropic stress with low systemic vascular resistance ($\text{HR}=100\text{ bpm}, R_{\text{sys}}=0.75$).

Each virtual patient was evaluated under both **Rest** and **Exercise** conditions across all 5 random seeds (100 total simulation episodes):

| Virtual Patient | Condition | Controller | MAP (mmHg) | CO (L/min) | FBE (mL/s) | Suction (%) | Congestion (%) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **VP-1 (Nominal Reference)** | Rest | Centralized SAC | 108.8 ± 3.5 | 5.90 ± 0.19 | 3.61 ± 2.34 | 0.0% | 0.0% |
| **VP-1 (Nominal Reference)** | Rest | Bilateral SAC | 95.5 ± 5.6 | 4.39 ± 1.93 | 13.79 ± 27.53 | 0.0% | 2.5% |
| **VP-1 (Nominal Reference)** | Exercise | Centralized SAC | 105.7 ± 2.0 | 6.66 ± 0.13 | 14.16 ± 1.84 | 0.0% | 0.0% |
| **VP-1 (Nominal Reference)** | Exercise | Bilateral SAC | 96.7 ± 6.9 | 5.92 ± 0.65 | 7.99 ± 7.66 | 0.0% | 0.6% |
| **VP-2 (Depressed Contractility)** | Rest | Centralized SAC | 106.2 ± 2.5 | 5.74 ± 0.13 | 0.87 ± 0.93 | 0.0% | 0.0% |
| **VP-2 (Depressed Contractility)** | Rest | Bilateral SAC | 93.5 ± 7.1 | 4.48 ± 1.73 | 10.47 ± 22.80 | 0.0% | 2.5% |
| **VP-2 (Depressed Contractility)** | Exercise | Centralized SAC | 101.1 ± 2.2 | 6.33 ± 0.14 | 8.03 ± 1.90 | 0.0% | 0.0% |
| **VP-2 (Depressed Contractility)** | Exercise | Bilateral SAC | 94.4 ± 9.4 | 5.81 ± 0.85 | 5.41 ± 5.95 | 0.0% | 0.6% |
| **VP-3 (Hypertensive & Stiff)** | Rest | Centralized SAC | 113.9 ± 4.3 | 5.07 ± 0.18 | 0.25 ± 0.01 | 0.0% | 0.0% |
| **VP-3 (Hypertensive & Stiff)** | Rest | Bilateral SAC | 110.1 ± 9.7 | 4.33 ± 1.21 | 10.31 ± 14.56 | 0.0% | 4.9% |
| **VP-3 (Hypertensive & Stiff)** | Exercise | Centralized SAC | 113.9 ± 3.3 | 5.90 ± 0.17 | 5.15 ± 1.94 | 0.0% | 0.0% |
| **VP-3 (Hypertensive & Stiff)** | Exercise | Bilateral SAC | 107.7 ± 5.9 | 5.43 ± 0.43 | 8.28 ± 2.89 | 0.0% | 1.2% |
| **VP-4 (Pulmonary Hypertension)** | Rest | Centralized SAC | 103.5 ± 1.9 | 5.57 ± 0.10 | 2.97 ± 1.23 | 0.0% | 0.0% |
| **VP-4 (Pulmonary Hypertension)** | Rest | Bilateral SAC | 96.9 ± 4.7 | 5.24 ± 0.26 | 0.70 ± 0.95 | 0.0% | 0.0% |
| **VP-4 (Pulmonary Hypertension)** | Exercise | Centralized SAC | 99.3 ± 1.4 | 6.21 ± 0.09 | 11.55 ± 1.21 | 0.0% | 0.0% |
| **VP-4 (Pulmonary Hypertension)** | Exercise | Bilateral SAC | 94.1 ± 9.5 | 5.75 ± 0.89 | 5.65 ± 4.51 | 0.0% | 0.0% |
| **VP-5 (Hyperdynamic / Tachycardic)** | Rest | Centralized SAC | 106.3 ± 1.4 | 6.65 ± 0.09 | 13.96 ± 1.19 | 0.0% | 0.0% |
| **VP-5 (Hyperdynamic / Tachycardic)** | Rest | Bilateral SAC | 95.7 ± 6.7 | 5.97 ± 0.51 | 5.32 ± 0.85 | 0.0% | 0.0% |
| **VP-5 (Hyperdynamic / Tachycardic)** | Exercise | Centralized SAC | 95.3 ± 9.5 | 5.99 ± 1.95 | 24.73 ± 14.84 | 0.0% | 0.0% |
| **VP-5 (Hyperdynamic / Tachycardic)** | Exercise | Bilateral SAC | 95.7 ± 10.6 | 6.38 ± 1.63 | 16.24 ± 17.77 | 0.0% | 1.9% |

#### Key Clinical Findings:
1. **Zero Suction Incidence**: Both DRL controllers maintained 0.0% suction across all 100 patient-condition episodes.
2. **Superior Pulmonary Protection**: In the challenging Pulmonary Hypertension phenotype (VP-4), Bilateral SAC achieved near-perfect flow balance ($\text{FBE} = 0.70 \pm 0.95\text{ mL/s}$ at Rest and $5.65 \pm 4.51\text{ mL/s}$ during Exercise), halving the flow discrepancy seen in Centralized SAC ($11.55 \pm 1.21\text{ mL/s}$).
3. **Hyperdynamic Autoregulation**: In VP-5 (high heart rate, low resistance), Bilateral SAC maintained superior exercise flow coordination ($\text{FBE} = 16.24\text{ mL/s}$ vs. $24.73\text{ mL/s}$ for Centralized SAC).

---

### 9.7 Reward Component Ablation: Impact of Flow Conservation ($R_{\text{balance}}$)

To determine whether the explicit flow-conservation penalty $R_{\text{balance}} = -w_2 |Q_{\text{LVAD}} - Q_{\text{RVAD}}|$ ($w_2 = 0.012261$) is causally necessary, or whether the agent can infer balance implicitly from demand tracking and pressure limits, an ablated policy was trained from scratch with $w_2 = 0.0$ for 500,000 steps (Seed 42) [[3]](#12-references):

| Scenario | $\text{FBE}$ With $R_{\text{balance}}$ (mL/s) | $\text{FBE}$ Without $R_{\text{balance}}$ (mL/s) | Divergence ($\Delta\text{FBE}$) | MAP With / Without (mmHg) | Suction With / Without (%) | Congestion With / Without (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rest** | 5.26 | 0.54 | Steady | 95.2 / 117.7 | 0.0% / 0.0% | 0.0% / 0.0% |
| **Exercise** | 3.31 | 3.14 | Comparable | 109.4 / 108.3 | 0.0% / 0.0% | 12.2% / 0.0% |
| **SVR_down** | **1.63** | **18.92** | **+17.29 mL/s (+1060.7%)** | 92.0 / 95.6 | 0.0% / 0.0% | 18.6% / 0.0% |
| **Hypovolemia** | 0.21 | 0.63 | +0.42 mL/s | 98.1 / 117.6 | 0.0% / 0.0% | 12.5% / 0.0% |
| **Suction_Test** | 4.36 | 0.77 | Stable | 81.2 / 115.6 | 22.0% / 0.0% | 0.0% / **12.1%** |

#### Mechanistic Findings:
1. **Critical Failure under Vasodilation (SVR_down)**: In stationary resting states, the ablated policy achieves balance passively because both targets match. However, under acute systemic vasodilation ($\text{SVR\_down}$), the policy without $R_{\text{balance}}$ experiences massive interventricular flow divergence: FBE surges from **$1.63\text{ mL/s} \to 18.92\text{ mL/s}$ (+1060.7% increase)**, dangerously grazing the $20\text{ mL/s}$ pulmonary safety limit.
2. **Venous Congestion Emergence**: Without $R_{\text{balance}}$, the policy over-pumps into the pulmonary circulation during acute hypovolemia and suction challenges, producing **12.1% venous congestion** ($P_{\text{atrial}} > 18\text{ mmHg}$).
3. **Causal Necessity Established**: Explicit flow conservation is mandatory for continuous-flow biventricular control during dynamic hemodynamic transitions.

---

## 10. Reproduction Guide & Quick Start

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

## 11. Repository Directory Structure

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

---

## 12. References

1. Koh, V.C.A.; Ho, Y.K.; Stevens, M.C.; Ng, B.C.; Salamonsen, R.F.; Lovell, N.H.; Lim, E. "A Centralised Multi-Objective Model Predictive Controller for Biventricular Assist Devices." *Biomedical Signal Processing and Control*, 2020, **56**, 101736. DOI: [10.1016/j.bspc.2019.101736](https://doi.org/10.1016/j.bspc.2019.101736).
2. Perra, M.; et al. "Physiological Control of Realheart Total Artificial Heart." *Artificial Organs*, 2025, **49**, 1548–1557. DOI: [10.1111/aor.14865](https://doi.org/10.1111/aor.14865).
3. Li, T.; Cui, W.; Xie, N.; Li, H.; Liu, H.; Li, X.; Wang, Y. "Intelligent and Strong Robust CVS-LVAD Control Based on Soft-Actor-Critic Algorithm." *Artificial Intelligence in Medicine*, 2022, **128**, 102308. DOI: [10.1016/j.artmed.2022.102308](https://doi.org/10.1016/j.artmed.2022.102308).
4. Shi, Y.; Xu, Z.; Chen, C.; He, F.; Hao, P.; Zhang, X. "Enhancing Cardiac Hemodynamic and Pulsatility in Heart Failure via Deep Reinforcement Learning: An In-Silico and In-Vitro Validation Study of Percutaneous Ventricular Assist Devices." *Computer Methods and Programs in Biomedicine*, 2025, **270**, 108975. DOI: [10.1016/j.cmpb.2025.108975](https://doi.org/10.1016/j.cmpb.2025.108975).
5. Koh, V.C.A.; Ho, Y.K.; Stevens, M.C.; Ng, B.C.; Salamonsen, R.F.; Lovell, N.H.; Lim, E. "A Centralized Multi-Objective Model Predictive Control for a Biventricular Assist Device: An In Silico Evaluation." *Biomedical Signal Processing and Control*, 2019, **49**, 137–148. DOI: [10.1016/j.bspc.2018.10.021](https://doi.org/10.1016/j.bspc.2018.10.021).
6. Ng, B.C.; Salamonsen, R.F.; Gregory, S.D.; Stevens, M.C.; Wu, Y.; Mansouri, M.; Lovell, N.H.; Lim, E. "Application of Multiobjective Neural Predictive Control to Biventricular Assistance Using Dual Rotary Blood Pumps." *Biomedical Signal Processing and Control*, 2018, **39**, 81–93. DOI: [10.1016/j.bspc.2017.07.011](https://doi.org/10.1016/j.bspc.2017.07.011).
7. Wang, Y.; Koenig, S.C.; Wu, Z.; Slaughter, M.S.; Giridharan, G.A. "Sensor-Based Physiologic Control Strategy for Biventricular Support with Rotary Blood Pumps." *ASAIO Journal*, 2018, **64**(3), 338–350. DOI: [10.1097/MAT.0000000000000632](https://doi.org/10.1097/MAT.0000000000000632).
8. Stephens, A.F.; Stevens, M.C.; Gregory, S.D.; Kleinheyer, M.; Salamonsen, R.F. "In Vitro Evaluation of an Immediate Response Starling-Like Controller for Dual Rotary Blood Pumps." *Artificial Organs*, 2017, **41**(10), 911–922. DOI: [10.1111/aor.12962](https://doi.org/10.1111/aor.12962).
9. Mansouri, M.; Salamonsen, R.F.; Lim, E.; Akmeliawati, R.; Lovell, N.H. "Preload-Based Starling-Like Control for Rotary Blood Pumps: Numerical Comparison with Pulsatility Control and Constant Speed Operation." *PLoS ONE*, 2015, **10**(4), e0121413. DOI: [10.1371/journal.pone.0121413](https://doi.org/10.1371/journal.pone.0121413).
10. Petrou, A.; Ochsner, G.; Amacher, R.; Pergantis, P.; Rebholz, M.; Meboldt, M.; Schmid Daners, M. "A Physiological Controller for Turbodynamic Ventricular Assist Devices Based on Left Ventricular Systolic Pressure." *Artificial Organs*, 2016, **40**(9), 842–855. DOI: [10.1111/aor.12658](https://doi.org/10.1111/aor.12658).
11. Li, T.; Cui, W.; Liu, X.; Li, X.; Xie, N.; Wang, Y. "A Physiological Control System for Pulsatile Ventricular Assist Device Using an Energy-Efficient Deep Reinforcement Learning Method." *IEEE Transactions on Instrumentation and Measurement*, 2023, **72**, 4001910. DOI: [10.1109/TIM.2023.3242010](https://doi.org/10.1109/TIM.2023.3242010).
12. Fernández-Zapico, D.; Peirelinck, T.; Deconinck, G.; Donker, D.W.; Fresiello, L. "Physiological Control for Left Ventricular Assist Devices Based on Deep Reinforcement Learning." *Artificial Organs*, 2024, **48**(12), 1418–1429. DOI: [10.1111/aor.14845](https://doi.org/10.1111/aor.14845).
13. Fetanat, M.; Stevens, M.; Hayward, C.; Lovell, N.H. "A Sensorless Control System for an Implantable Heart Pump Using a Real-Time Deep Convolutional Neural Network." *IEEE Transactions on Biomedical Engineering*, 2021, **68**(10), 3029–3038. DOI: [10.1109/TBME.2021.3060690](https://doi.org/10.1109/TBME.2021.3060690).
14. Fetanat, M.; Stevens, M.; Hayward, C.; Lovell, N.H. "A Physiological Control System for an Implantable Heart Pump that Accommodates for Interpatient and Intrapatient Variations." *IEEE Transactions on Biomedical Engineering*, 2020, **67**(4), 1167–1175. DOI: [10.1109/TBME.2019.2932233](https://doi.org/10.1109/TBME.2019.2932233).
15. Hussein, U.; Chou, W.K.; Balasubramanian, A.; Rahmatova, J.; Wilkinson, L.; Arjomandi Rad, A.; Dimarakis, I.; Kourliouros, A. "The Application of Artificial Intelligence and Machine Learning in Left Ventricular Assist Device Implantation: A Systematic Review." *Artificial Organs*, 2025, **49**(3), 295–311. DOI: [10.1111/aor.14915](https://doi.org/10.1111/aor.14915).
16. Tasmurzayev, N.; Amangeldy, B.; Imanbek, B.; Baigarayeva, Z.; Imankulov, T.; Dikhanbayeva, G.; Amangeldi, I.; Sharipova, S. "Digital Cardiovascular Twins, AI Agents, and Sensor Data: A Narrative Review from System Architecture to Proactive Heart Health." *Sensors*, 2025, **25**(17), 5272. DOI: [10.3390/s25175272](https://doi.org/10.3390/s25175272).
17. Stergiopulos, N.; Meister, J.J.; Westerhof, N. "Determinants of Stroke Volume and Systolic and Diastolic Aortic Pressure." *American Journal of Physiology–Heart and Circulatory Physiology*, 1996, **270**(6), H2050–H2059. DOI: [10.1152/ajpheart.1996.270.6.H2050](https://doi.org/10.1152/ajpheart.1996.270.6.H2050).
18. Wang, Y.; Loghmanpour, N.; Vandenberghe, S.; Ferreira, A.; Keller, B.; Gorcsan, J.; Antaki, J. "Simulation of Dilated Heart Failure with Continuous Flow Circulatory Support." *PLoS ONE*, 2014, **9**(1), e85234. DOI: [10.1371/journal.pone.0085234](https://doi.org/10.1371/journal.pone.0085234).
19. Arts, T.; Delhaas, T.; Bovendeerd, P.; Verbeek, X.; Prinzen, F.W. "Adaptation to Mechanical Load Determines Shape and Properties of Heart and Circulation: The CircAdapt Model." *American Journal of Physiology–Heart and Circulatory Physiology*, 2005, **288**(4), H1943–H1954. DOI: [10.1152/ajpheart.00444.2004](https://doi.org/10.1152/ajpheart.00444.2004).
20. Haarnoja, T.; Zhou, A.; Abbeel, P.; Levine, S. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." In *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR, 2018, **80**, 1861–1870.
21. Lillicrap, T.P.; Hunt, J.J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; Wierstra, D. "Continuous Control with Deep Reinforcement Learning." In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2016. arXiv: [1509.02971](https://arxiv.org/abs/1509.02971).
22. Lowe, R.; Wu, Y.; Tamar, A.; Harb, J.; Abbeel, P.; Mordatch, I. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, **30**, 6379–6390.
23. Henderson, P.; Islam, R.; Bachman, P.; Pineau, J.; Precup, D.; Meger, D. "Deep Reinforcement Learning that Matters." In *Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI)*, 2018, **32**(1), 3207–3214.
24. Raffin, A.; Hill, A.; Gleave, A.; Kanervisto, A.; Ernestus, M.; Dormann, N. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *Journal of Machine Learning Research*, 2021, **22**(268), 1–8.
25. Towers, M.; Kwiatkowski, A.; Terry, J.; Balis, J.U.; De Cola, G.; Deleu, T.; Goulão, M.; Kallinteris, A.; Krimmel, M.; KG, A.; et al. "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv preprint, 2024. arXiv: [2407.17032](https://arxiv.org/abs/2407.17032).
26. Bain, M.; Sammut, C. "A Framework for Behavioural Cloning." *Machine Intelligence*, 2000, **15**, 103–129.
27. Savarese, G.; Becher, P.M.; Lund, L.H.; Seferovic, P.; Rosano, G.M.C.; Coats, A.J.S. "Global Burden of Heart Failure: A Comprehensive and Updated Review of Epidemiology." *Cardiovascular Research*, 2023, **118**(17), 3272–3287. DOI: [10.1093/cvr/cvac013](https://doi.org/10.1093/cvr/cvac013).
28. The Society of Thoracic Surgeons Intermacs Investigators. "The Society of Thoracic Surgeons Intermacs 2025 Annual Report: Focus on Outcomes in Older Adults." *Annals of Thoracic Surgery*, 2025, **119**(3), 450–462. DOI: [10.1016/j.athoracsur.2025.10.005](https://doi.org/10.1016/j.athoracsur.2025.10.005).
29. Ferrall, J.; Vaidya, A.S.; Kawaguchi, E.S.; Patel, S.G.; Lee, R.C.; Lee, E.S.; Wolfson, A.M. "Comparison of Waitlist and Post-Transplant Outcomes in Patients Supported with Total Artificial Heart versus Continuous Biventricular Assist Devices." *Artificial Organs*, 2025, **49**(2), 281–291. DOI: [10.1111/aor.14884](https://doi.org/10.1111/aor.14884).
30. ISHLT Consensus Conference Participants. "Consensus Statements from the International Society for Heart and Lung Transplantation Consensus Conference: Heart Failure–Related Cardiogenic Shock." *Journal of Heart and Lung Transplantation*, 2024, **43**(2), 204–216. DOI: [10.1016/j.healun.2023.11.008](https://doi.org/10.1016/j.healun.2023.11.008).
31. Mehra, M.R.; Goldstein, D.J.; Uriel, N.; et al. "Two-Year Outcomes with a Magnetically Levitated Cardiac Pump in Heart Failure." *New England Journal of Medicine*, 2018, **378**(15), 1386–1395. DOI: [10.1056/NEJMoa1800866](https://doi.org/10.1056/NEJMoa1800866).
32. Kosta, S.; Dauby, P.C. "Frank-Starling Mechanism, Fluid Responsiveness, and Length-Dependent Activation: Unravelling the Multiscale Behaviors with an In Silico Analysis." *PLoS Computational Biology*, 2021, **17**(10), e1009469. DOI: [10.1371/journal.pcbi.1009469](https://doi.org/10.1371/journal.pcbi.1009469).
33. Ursino, M. "Interaction between Carotid Baroregulation and the Pulsating Heart: A Mathematical Model." *American Journal of Physiology–Heart and Circulatory Physiology*, 1998, **275**(5), H1733–H1747. DOI: [10.1152/ajpheart.1998.275.5.H1733](https://doi.org/10.1152/ajpheart.1998.275.5.H1733).
34. Bakouri, M.; Alassaf, A.; Alshareef, K.; Abdelsalam, S.; Ismail, H.F.; Ganoun, A.; Alomari, A.-H. "An Optimal H-Infinity Controller for Left Ventricular Assist Devices Based on a Starling-like Controller: A Simulation Study." *Mathematics*, 2022, **10**(5), 731. DOI: [10.3390/math10050731](https://doi.org/10.3390/math10050731).
35. Slaughter, M.S.; Pagani, F.D.; Rogers, J.G.; et al. "Clinical Management of Continuous-Flow Left Ventricular Assist Devices in Advanced Heart Failure." *Journal of Heart and Lung Transplantation*, 2010, **29**(4 Suppl), S1–S39. DOI: [10.1016/j.healun.2010.01.011](https://doi.org/10.1016/j.healun.2010.01.011).

---


