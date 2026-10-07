"""
Reward Component Ablation Training & Evaluation: Bilateral SAC Without R_balance
================================================================================
Ablation Study: Isolating the causal contribution of the novel flow conservation term:
    R_balance = -w2 * |Q_LVAD - Q_RVAD|

Protocol:
  1. Sets w2 = 0.0 (ablating R_balance) while keeping all other components identical:
     w1 = 0.678709 (R_track), w3 = 0.104327 (R_smooth), w_map = 0.05 (R_map).
  2. Pre-trains with Behavioral Cloning warm-start (8 epochs, expert_data_pi.npz).
  3. Trains Bilateral-Branch SAC for 500,000 steps using 4 CPU workers (SubprocVecEnv).
  4. Evaluates the trained model against the baseline model (with R_balance) across:
     - Rest
     - Exercise
     - SVR_down
     - Hypovolemia
     - Suction_Test
  5. Generates a publication-ready comparison table (CSV & Markdown).
"""

import os
import sys
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Set project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import reward
# Zero out the flow balance weight for the ablation
reward.DEFAULT_WEIGHTS['w2'] = 0.0
print(f"[ABLATION ACTIVE] reward.DEFAULT_WEIGHTS set to: {reward.DEFAULT_WEIGHTS}")

from bivad_env import BiVADEnv
from bilateral_extractor import BilateralFeaturesExtractor


# =====================================================================
# Behavioral Cloning Warm-Start
# =====================================================================
def pretrain_actor_bc(model, expert_data_path='expert_data_pi.npz', epochs=8, batch_size=256, lr=1e-3, device='cpu', save_path="./sac_bilateral_no_balance_bc.zip"):
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert dataset '{expert_data_path}' not found!")

    print(f"\n--- Starting BC Pretraining (w2 = 0) from {expert_data_path} ---")
    data = np.load(expert_data_path)
    obs = data['obs'].astype(np.float32)
    acts = data['acts'].astype(np.float32)

    dataset = TensorDataset(torch.tensor(obs), torch.tensor(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    actor = model.actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    actor.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_acts in loader:
            batch_obs = batch_obs.to(device)
            batch_acts = batch_acts.to(device)
            pred_acts = actor(batch_obs)
            loss = F.mse_loss(pred_acts, batch_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Supervised MSE Loss: {total_loss / n_batches:.6f}")

    model.save(save_path)
    print(f"[Saved] BC Pretrained model: {save_path}")
    return model


# =====================================================================
# Environment Factory for 4 CPU Workers
# =====================================================================
def make_train_env(n_envs=4, base_seed=42):
    def _init(rank):
        def _thunk():
            import reward
            reward.DEFAULT_WEIGHTS['w2'] = 0.0  # Ensure worker has w2 = 0.0
            env = BiVADEnv(randomize=True, raw_obs=False)
            env.reset(seed=base_seed + rank * 1000)
            return env
        return _thunk

    vec_env = SubprocVecEnv([_init(i) for i in range(n_envs)])
    vec_norm_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    return vec_norm_env


# =====================================================================
# Evaluation Helper Across Key Scenarios
# =====================================================================
EVAL_SCENARIOS = {
    'Rest': {
        'perturb_step': 0,
        'changes': {}
    },
    'Exercise': {
        'perturb_step': 500,
        'changes': {500: lambda p: p.update({'HR': min(p['HR'] * 1.33, 130.0), 'R_sys': p['R_sys'] * 0.80})}
    },
    'SVR_down': {
        'perturb_step': 500,
        'changes': {500: lambda p: p.update({'R_sys': 0.60})}
    },
    'Hypovolemia': {
        'perturb_step': 0,
        'changes': {0: lambda p: p.update({'C_sys': p['C_sys'] * 0.75, 'C_pulm': p['C_pulm'] * 0.75})}
    },
    'Suction_Test': {
        'perturb_step': 500,
        'changes': {
            500: lambda p: p.update({'C_la': p['C_la'] * 0.30, 'C_sys': p['C_sys'] * 0.50}),
            1200: lambda p: p.update({'C_la': p['C_la'] / 0.30, 'C_sys': p['C_sys'] / 0.50})
        }
    }
}


def run_single_eval(model, scenario_name, seed=42):
    cfg = EVAL_SCENARIOS[scenario_name]
    env = BiVADEnv(randomize=False)
    obs, info = env.reset(seed=seed)

    maps = []
    cos = []
    fbes = []
    suction_steps = 0
    congestion_steps = 0

    for step_i in range(env.max_steps):
        if step_i in cfg['changes']:
            cfg['changes'][step_i](env.params)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward_val, terminated, truncated, info = env.step(action)

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
        'Scenario': scenario_name,
        'MAP_mean': float(np.mean(maps)),
        'CO_mean': float(np.mean(cos)),
        'FBE_mean': float(np.mean(fbes)),
        'FBE_peak': float(np.max(fbes)),
        'Suction_Pct': float(suction_steps / total_steps * 100.0),
        'Congestion_Pct': float(congestion_steps / total_steps * 100.0)
    }


def main():
    parser = argparse.ArgumentParser(description="Bilateral SAC Ablation Training (Without R_balance)")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps (default: 500,000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--n_envs", type=int, default=4, help="Subprocess workers (default: 4)")
    parser.add_argument("--baseline_model", type=str, default="", help="Path to baseline model (with R_balance)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = CURRENT_DIR
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, "sac_bilateral_no_balance_final.zip")

    # 1. Initialize Bilateral SAC architecture
    print("=" * 80)
    print("  STEP 1: INITIALIZING BILATERAL SAC (ABLATION: WITHOUT R_BALANCE)")
    print("=" * 80)
    base_env = BiVADEnv(randomize=False, raw_obs=False)
    base_env.reset(seed=args.seed)

    policy_kwargs = dict(
        features_extractor_class=BilateralFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256]
    )

    model = SAC(
        policy="MlpPolicy",
        env=base_env,
        learning_rate=3e-4,
        buffer_size=300000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=1
    )

    # 2. Behavioral Cloning warm-start
    bc_path = os.path.join(save_dir, "sac_bilateral_no_balance_bc.zip")
    expert_path = os.path.join(PROJECT_ROOT, "expert_data_pi.npz")
    if not os.path.exists(expert_path):
        expert_path = "expert_data_pi.npz"
    model = pretrain_actor_bc(model, expert_data_path=expert_path, epochs=8, save_path=bc_path)

    # 3. Full SAC Training (500,000 steps with 4 CPU workers)
    print("\n" + "=" * 80)
    print(f"  STEP 2: TRAINING ON 4 CPU CORES ({args.steps:,} steps, w2 = 0.0)")
    print("=" * 80)
    train_env = make_train_env(n_envs=args.n_envs, base_seed=args.seed)
    
    # Correct SB3 pattern: Load BC pretrained model with the 4-worker vectorized environment
    model = SAC.load(
        bc_path,
        env=train_env,
        device="cpu",
        custom_objects={"features_extractor_class": BilateralFeaturesExtractor}
    )

    model.learn(total_timesteps=args.steps, log_interval=50)
    model.save(final_model_path)
    train_env.close()
    print(f"\n[Saved] Final ablated model: {final_model_path}")

    # 4. Comparative Evaluation: With vs. Without R_balance
    print("\n" + "=" * 80)
    print("  STEP 3: COMPARATIVE EVALUATION (WITH VS. WITHOUT R_BALANCE)")
    print("=" * 80)

    # Locate baseline model
    baseline_path = args.baseline_model
    if not baseline_path or not os.path.exists(baseline_path):
        candidates = [
            os.path.join(PROJECT_ROOT, "Results", "bilateral_sac_seed_42", "sac_bilateral_final.zip"),
            os.path.join(CURRENT_DIR, "sac_bilateral_baseline_seed42.zip"),
            "sac_bilateral_final.zip"
        ]
        for c in candidates:
            if os.path.exists(c):
                baseline_path = c
                break

    has_baseline = baseline_path and os.path.exists(baseline_path)
    if has_baseline:
        print(f"Loading baseline model (With R_balance) from: {baseline_path}")
        baseline_model = SAC.load(baseline_path, device='cpu', custom_objects={'features_extractor_class': BilateralFeaturesExtractor})
    else:
        print("[NOTE] Baseline model not found locally; evaluating ablated model only.")

    results_table = []
    for sc_name in EVAL_SCENARIOS.keys():
        row_ablation = run_single_eval(model, sc_name, seed=args.seed)
        if has_baseline:
            row_baseline = run_single_eval(baseline_model, sc_name, seed=args.seed)
            diff_fbe = row_ablation['FBE_mean'] - row_baseline['FBE_mean']
            pct_increase = (diff_fbe / max(row_baseline['FBE_mean'], 0.1)) * 100.0

            results_table.append({
                'Scenario': sc_name,
                'FBE With R_balance (mL/s)': f"{row_baseline['FBE_mean']:.2f}",
                'FBE Without R_balance (mL/s)': f"{row_ablation['FBE_mean']:.2f}",
                'FBE Deterioration': f"+{diff_fbe:.2f} mL/s (+{pct_increase:.1f}%)" if diff_fbe > 0 else f"{diff_fbe:.2f} mL/s",
                'MAP With / Without (mmHg)': f"{row_baseline['MAP_mean']:.1f} / {row_ablation['MAP_mean']:.1f}",
                'Suct % With / Without': f"{row_baseline['Suction_Pct']:.1f}% / {row_ablation['Suction_Pct']:.1f}%"
            })
        else:
            results_table.append({
                'Scenario': sc_name,
                'FBE Without R_balance (mL/s)': f"{row_ablation['FBE_mean']:.2f}",
                'MAP (mmHg)': f"{row_ablation['MAP_mean']:.1f}",
                'CO (L/min)': f"{row_ablation['CO_mean']:.2f}",
                'Suction (%)': f"{row_ablation['Suction_Pct']:.1f}%",
                'Congestion (%)': f"{row_ablation['Congestion_Pct']:.1f}%"
            })

    df_res = pd.DataFrame(results_table)
    csv_out = os.path.join(save_dir, "table_reward_ablation.csv")
    md_out = os.path.join(save_dir, "table_reward_ablation.md")
    df_res.to_csv(csv_out, index=False)

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Table: Reward Component Ablation — Impact of $R_{\\text{balance}}$\n\n")
        f.write("Comparison of Bilateral-Branch SAC trained with vs. without the flow-conservation penalty ($R_{\\text{balance}}$, Seed 42, 500k steps).\n\n")
        headers = list(df_res.columns)
        widths = [max(len(str(val)) for val in [h] + list(df_res[h])) for h in headers]
        hdr_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "|-" + "-|-".join("-" * w for w in widths) + "-|"
        f.write(hdr_line + "\n" + sep_line + "\n")
        for _, r in df_res.iterrows():
            row_line = "| " + " | ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)) + " |"
            f.write(row_line + "\n")
        f.write("\n")

    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY:")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\n[Saved] CSV Table: {csv_out}")
    print(f"[Saved] Markdown Table: {md_out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
