"""
Virtual Patient Robustness Evaluation Harness
==============================================
Evaluates trained Centralized-Flat SAC and Bilateral-Branch SAC controllers
across 5 clinically grounded Virtual Patient (VP) phenotypes under Rest and Exercise.

Virtual Patient Archetypes:
  VP-1: Nominal Reference (Baseline Severe HF: E_max,lv=0.80, R_sys=1.00, HR=75)
  VP-2: Depressed Contractility / Severe Dilated HF (-20% E_max,lv, -20% E_max,rv)
  VP-3: Hypertensive & Stiff Vasculature (+25% R_sys, -20% C_sys)
  VP-4: Secondary Pulmonary Hypertension (+50% R_pulm, -20% C_pulm)
  VP-5: Hyperdynamic / Tachycardic HF (+20% HR, -15% R_sys)

Produces a concise, publication-ready comparison table (CSV & Markdown).
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Set project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bivad_env import BiVADEnv
from bilateral_extractor import BilateralFeaturesExtractor

# Define 5 standardized Virtual Patient profiles
VIRTUAL_PATIENTS = {
    'VP-1 (Nominal Reference)': {
        'description': 'Baseline severe HF profile',
        'overrides': {}
    },
    'VP-2 (Depressed Contractility)': {
        'description': 'Severe dilated HF: -20% LV/RV elastance',
        'overrides': {'E_max_lv': 0.64, 'E_max_rv': 0.24}
    },
    'VP-3 (Hypertensive & Stiff)': {
        'description': 'Stiff arterial tree: +25% R_sys, -20% C_sys',
        'overrides': {'R_sys': 1.25, 'C_sys': 1.06}
    },
    'VP-4 (Pulmonary Hypertension)': {
        'description': 'Elevated pulmonary vascular resistance: +50% R_pulm',
        'overrides': {'R_pulm': 0.15, 'C_pulm': 1.06}
    },
    'VP-5 (Hyperdynamic / Tachycardic)': {
        'description': 'Chronotropic elevation: HR=90 bpm, -15% R_sys',
        'overrides': {'HR': 90.0, 'R_sys': 0.85}
    }
}

CONDITIONS = {
    'Rest': {
        'description': 'Resting baseline (200s steady operation)',
        'perturb_step': 0,
        'changes': {}
    },
    'Exercise': {
        'description': 'Physical exertion at t=50s (HR +33%, R_sys -20%)',
        'perturb_step': 500,
        'changes': {500: lambda p: p.update({'HR': min(p['HR'] * 1.33, 130.0), 'R_sys': p['R_sys'] * 0.80})}
    }
}

SEEDS = [42, 123, 456, 789, 1024]


def load_model(controller_type, seed):
    """Loads trained checkpoint from Results/ directory."""
    if controller_type == 'Bilateral SAC':
        model_path = os.path.join(PROJECT_ROOT, "Results", f"bilateral_sac_seed_{seed}", "sac_bilateral_final.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "Results", f"bilateral_sac_seed_{seed}", f"sac_bilateral_500000_steps.zip")
        return SAC.load(model_path, device='cpu', custom_objects={'features_extractor_class': BilateralFeaturesExtractor})
    elif controller_type == 'Centralized SAC':
        model_path = os.path.join(PROJECT_ROOT, "Results", f"centralized_sac_seed_{seed}", "sac_centralized_final.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "Results", f"centralized_sac_seed_{seed}", f"sac_centralized_seed{seed}_500000_steps.zip")
        return SAC.load(model_path, device='cpu')
    else:
        raise ValueError(f"Unknown controller: {controller_type}")


def run_evaluation_episode(model, vp_overrides, condition_cfg, seed=42):
    """Executes a single 2000-step episode under a specific virtual patient profile."""
    env = BiVADEnv(randomize=False)
    obs, info = env.reset(seed=seed)

    # Apply Virtual Patient baseline parameter overrides
    if vp_overrides:
        env.params.update(vp_overrides)

    perturb_step = condition_cfg['perturb_step']
    changes = condition_cfg['changes']

    maps = []
    cos = []
    fbes = []
    suction_steps = 0
    congestion_steps = 0

    for step_i in range(env.max_steps):
        # Apply dynamic condition perturbations if scheduled
        if step_i in changes and step_i > 0:
            changes[step_i](env.params)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        maps.append(info['MAP'])
        cos.append(info['CO_left'])
        fbes.append(info['flow_balance_error'])

        if info['suction_left'] or info['suction_right'] or env.state[2] < 3.0:
            suction_steps += 1
        if info['congestion_left'] or info['congestion_right'] or env.state[2] > 18.0 or env.state[3] > 18.0:
            congestion_steps += 1

        if terminated or truncated:
            break

    total_steps = len(maps)
    return {
        'MAP_mean': float(np.mean(maps)),
        'CO_mean': float(np.mean(cos)),
        'FBE_mean': float(np.mean(fbes)),
        'Suction_Pct': float(suction_steps / total_steps * 100.0),
        'Congestion_Pct': float(congestion_steps / total_steps * 100.0)
    }


def main():
    print("=" * 80)
    print("  VIRTUAL PATIENT ROBUSTNESS EVALUATION: Centralized SAC vs. Bilateral SAC")
    print("=" * 80)

    controllers = ['Centralized SAC', 'Bilateral SAC']
    raw_results = []

    for vp_name, vp_cfg in VIRTUAL_PATIENTS.items():
        print(f"\nEvaluating {vp_name}...")
        for cond_name, cond_cfg in CONDITIONS.items():
            for ctrl_name in controllers:
                for seed in SEEDS:
                    try:
                        model = load_model(ctrl_name, seed)
                        metrics = run_evaluation_episode(model, vp_cfg['overrides'], cond_cfg, seed=seed)
                        record = {
                            'Virtual_Patient': vp_name,
                            'Condition': cond_name,
                            'Controller': ctrl_name,
                            'Seed': seed,
                            **metrics
                        }
                        raw_results.append(record)
                    except Exception as e:
                        print(f"  [ERROR] {ctrl_name} Seed {seed} failed: {e}")

                # Quick progress print
                sub = [r for r in raw_results if r['Virtual_Patient'] == vp_name and r['Condition'] == cond_name and r['Controller'] == ctrl_name]
                if sub:
                    fbe_vals = [r['FBE_mean'] for r in sub]
                    print(f"  [{cond_name:<8s}] {ctrl_name:<16s}: FBE = {np.mean(fbe_vals):5.2f} +/- {np.std(fbe_vals):4.2f} mL/s")

    df_raw = pd.DataFrame(raw_results, columns=[
        'Virtual_Patient', 'Condition', 'Controller', 'Seed',
        'MAP_mean', 'CO_mean', 'FBE_mean', 'Suction_Pct', 'Congestion_Pct'
    ])
    raw_csv_path = os.path.join(CURRENT_DIR, "raw_virtual_patient_runs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n[Saved] Raw run results: {raw_csv_path}")

    if df_raw.empty:
        print("\n[ERROR] No successful evaluation runs recorded.")
        return

    # =========================================================================
    # Generate Publication-Ready Summary Table
    # =========================================================================
    summary_rows = []
    for vp_name in VIRTUAL_PATIENTS.keys():
        for cond_name in CONDITIONS.keys():
            for ctrl_name in controllers:
                sub = df_raw[(df_raw['Virtual_Patient'] == vp_name) & 
                             (df_raw['Condition'] == cond_name) & 
                             (df_raw['Controller'] == ctrl_name)]
                if len(sub) == 0:
                    continue

                row = {
                    'Virtual Patient': vp_name,
                    'Condition': cond_name,
                    'Controller': ctrl_name,
                    'MAP (mmHg)': f"{sub['MAP_mean'].mean():.1f} ± {sub['MAP_mean'].std():.1f}",
                    'CO (L/min)': f"{sub['CO_mean'].mean():.2f} ± {sub['CO_mean'].std():.2f}",
                    'FBE (mL/s)': f"{sub['FBE_mean'].mean():.2f} ± {sub['FBE_mean'].std():.2f}",
                    'Suction (%)': f"{sub['Suction_Pct'].mean():.1f}%",
                    'Congestion (%)': f"{sub['Congestion_Pct'].mean():.1f}%"
                }
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    table_csv_path = os.path.join(CURRENT_DIR, "table_virtual_patients.csv")
    table_md_path = os.path.join(CURRENT_DIR, "table_virtual_patients.md")
    df_summary.to_csv(table_csv_path, index=False)

    # Save clean Markdown table
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Zero-Shot Robustness Across Virtual Patient Cohort\n\n")
        f.write("Evaluation of Centralized-Flat SAC and Bilateral-Branch SAC across 5 virtual patient phenotypes under Rest and Exercise conditions ($n = 5$ random seeds, mean $\\pm$ SD).\n\n")
        
        # Markdown table formatter
        headers = list(df_summary.columns)
        widths = [max(len(str(val)) for val in [h] + list(df_summary[h])) for h in headers]
        hdr_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "|-" + "-|-".join("-" * w for w in widths) + "-|"
        f.write(hdr_line + "\n" + sep_line + "\n")
        for _, r in df_summary.iterrows():
            row_line = "| " + " | ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)) + " |"
            f.write(row_line + "\n")
        f.write("\n")

    print(f"[Saved] Publication Table (CSV): {table_csv_path}")
    print(f"[Saved] Publication Table (MD):  {table_md_path}")
    print("=" * 80)
    print("VIRTUAL PATIENT EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
