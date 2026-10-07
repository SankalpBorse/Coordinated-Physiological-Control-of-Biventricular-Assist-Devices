"""
Phase 6 — Independent Dual-Agent SAC Controller Comprehensive Evaluation Harness
================================================================================
Evaluates the trained Independent Dual-Agent SAC Policy across all 8 standardized scenarios:
  1. Rest
  2. SVR_up
  3. SVR_down
  4. PVR_up
  5. Exercise
  6. Exercise_Recovery
  7. Hypovolemia
  8. Combined

Architecture:
  - Agent Left  (LVAD): Observes [LAP, LVAD_Flow, AoP, V_LV, HR] (5 dims), Controls Delta_LVAD (1 dim)
  - Agent Right (RVAD): Observes [RAP, RVAD_Flow, PAP, V_RV, HR] (5 dims), Controls Delta_RVAD (1 dim)
  - Evaluates inter-ventricular synchronization under decentralized observation (Step 5.8 Core Ablation)

Outputs:
  - results_independent/independent_trajectories.csv
  - results_independent/independent_summary_metrics.csv
  - High-resolution publication figures in results_independent/figures/
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure workspace root is in path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from bivad_env import BiVADEnv

# Observation indices (Step 5.8)
LEFT_OBS_INDICES  = [0, 2, 4, 6, 8]  # [LAP, LVAD_flow, AoP, V_LV, HR]
RIGHT_OBS_INDICES = [1, 3, 5, 7, 8]  # [RAP, RVAD_flow, PAP, V_RV, HR]


# =====================================================================
# Actor Architecture
# =====================================================================
class Actor(nn.Module):
    """SAC Gaussian Actor Network with tanh squashing (net_arch=[256, 256])."""
    def __init__(self, obs_dim=5, act_dim=1, max_action=110.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.max_action = max_action

    def forward(self, obs, deterministic=True):
        feat = self.net(obs)
        mu = self.mu(feat)
        if deterministic:
            return torch.tanh(mu) * self.max_action
        else:
            log_std = torch.clamp(self.log_std(feat), -20, 2)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            return torch.tanh(dist.rsample()) * self.max_action


# =====================================================================
# Scenario Definitions
# =====================================================================
SCENARIOS = {
    1: {'name': 'Rest',              'changes': {}},
    2: {'name': 'SVR_up',            'changes': {500: lambda p: p.update({'R_sys': 1.5})}},
    3: {'name': 'SVR_down',          'changes': {500: lambda p: p.update({'R_sys': 0.6})}},
    4: {'name': 'PVR_up',            'changes': {500: lambda p: p.update({'R_pulm': 0.3})}},
    5: {'name': 'Exercise',          'changes': {500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys'] * 0.8})}},
    6: {'name': 'Exercise_Recovery',  'changes': {
            500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys'] * 0.8}),
            1200: lambda p: p.update({'HR': 75.0, 'R_sys': p['R_sys'] / 0.8})
    }},
    7: {'name': 'Hypovolemia',       'changes': {0: lambda p: p.update({
            'C_la': p['C_la'] * 0.8, 'C_ra': p['C_ra'] * 0.8,
            'C_sys': p['C_sys'] * 0.8, 'C_pulm': p['C_pulm'] * 0.8
    })}},
    8: {'name': 'Combined',          'changes': {500: lambda p: p.update({
            'R_sys': p['R_sys'] * 1.3, 'R_pulm': p['R_pulm'] * 1.5, 'HR': p['HR'] * 1.2
    })}},
}


# =====================================================================
# Model Search Helper
# =====================================================================
def find_independent_weights(checkpoint_dir="./checkpoints_independent"):
    """Locates Left and Right actor weights."""
    search_dirs = [
        checkpoint_dir,
        "independent_sac_results/checkpoints_independent",
        "independent_sac_results",
        "./checkpoints_independent",
        "checkpoints_independent"
    ]

    for d in search_dirs:
        act_l = os.path.join(d, "actor_l_final.pth")
        act_r = os.path.join(d, "actor_r_final.pth")
        if os.path.exists(act_l) and os.path.exists(act_r):
            return act_l, act_r

    # Fallback to latest step checkpoint
    for d in search_dirs:
        matches_l = glob.glob(os.path.join(d, "actor_l_step_*.pth"))
        if matches_l:
            best_l = sorted(matches_l)[-1]
            step_tag = os.path.basename(best_l).replace("actor_l_", "")
            best_r = os.path.join(d, f"actor_r_{step_tag}")
            if os.path.exists(best_r):
                return best_l, best_r

    # Fallback to BC weights
    for d in search_dirs:
        bc_l = os.path.join(d, "actor_left_bc.pth")
        bc_r = os.path.join(d, "actor_right_bc.pth")
        if os.path.exists(bc_l) and os.path.exists(bc_r):
            return bc_l, bc_r

    raise FileNotFoundError(
        "Could not locate independent actor weights ('actor_l_final.pth' / 'actor_r_final.pth'). "
        "Please specify --weights_dir."
    )


# =====================================================================
# Scenario Evaluation
# =====================================================================
def evaluate_scenario(actor_l, actor_r, scenario_id, scenario_name, scenario_dict, device='cpu'):
    env = BiVADEnv(randomize=False, raw_obs=False)
    env.perturbation_schedule = {}

    obs, _ = env.reset()

    if 0 in scenario_dict['changes']:
        scenario_dict['changes'][0](env.params)

    trajectories = []
    totals, tracks, balances, safeties, smooths = [], [], [], [], []
    laps, raps, maps, cos, fbes = [], [], [], [], []
    lvad_rpms, rvad_rpms = [], []

    suction_violations = 0
    congestion_violations = 0

    for step_i in range(env.max_steps):
        if step_i in scenario_dict['changes'] and step_i > 0:
            scenario_dict['changes'][step_i](env.params)

        # Slice decentralized views
        o_l = torch.as_tensor(obs[LEFT_OBS_INDICES], dtype=torch.float32, device=device).unsqueeze(0)
        o_r = torch.as_tensor(obs[RIGHT_OBS_INDICES], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            a_l = actor_l(o_l, deterministic=True).cpu().numpy()[0, 0]
            a_r = actor_r(o_r, deterministic=True).cpu().numpy()[0, 0]

        action = np.array([a_l, a_r], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        y = info.get('true_state', env.state)
        lap, rap = y[2], y[3]
        aop = y[4]
        lvad_flow = y[10]
        co = lvad_flow * 60.0 / 1000.0
        fbe = info.get('flow_balance_error', abs(y[10] - y[11]))
        lvad_rpm = info.get('LVAD_RPM', env.params['LVAD_RPM'])
        rvad_rpm = info.get('RVAD_RPM', env.params['RVAD_RPM'])

        if lap < 3.0 or rap < 1.0:
            suction_violations += 1
        if lap > 18.0 or rap > 12.0:
            congestion_violations += 1

        totals.append(reward)
        tracks.append(info.get('reward_track', 0.0))
        balances.append(info.get('reward_balance', 0.0))
        safeties.append(info.get('reward_safety', 0.0))
        smooths.append(info.get('reward_smooth', 0.0))

        laps.append(lap)
        raps.append(rap)
        maps.append(aop)
        cos.append(co)
        fbes.append(fbe)
        lvad_rpms.append(lvad_rpm)
        rvad_rpms.append(rvad_rpm)

        time_sec = round((step_i + 1) * env.dt, 2)
        trajectories.append({
            'Time': time_sec,
            'Controller': 'Independent_SAC',
            'Scenario': scenario_name,
            'MAP': aop,
            'LAP': lap,
            'RAP': rap,
            'LVAD_RPM': lvad_rpm,
            'RVAD_RPM': rvad_rpm,
            'CO_left': co,
            'flow_balance_error': fbe,
            'Reward': reward,
            'Delta_LVAD': a_l,
            'Delta_RVAD': a_r
        })

        if terminated or truncated:
            break

    n_steps = len(totals)
    summary = {
        'Scenario': scenario_name,
        'Mean_LAP': np.mean(laps),
        'Min_LAP': np.min(laps),
        'Max_LAP': np.max(laps),
        'Mean_RAP': np.mean(raps),
        'Min_RAP': np.min(raps),
        'Max_RAP': np.max(raps),
        'Mean_MAP': np.mean(maps),
        'Mean_CO': np.mean(cos),
        'Mean_FBE': np.mean(fbes),
        'Peak_FBE': np.max(fbes),
        'Min_LVAD_RPM': np.min(lvad_rpms),
        'Max_LVAD_RPM': np.max(lvad_rpms),
        'Min_RVAD_RPM': np.min(rvad_rpms),
        'Max_RVAD_RPM': np.max(rvad_rpms),
        'Mean_Reward': np.mean(totals),
        'Total_Reward': np.sum(totals),
        'Suction_Pct': 100.0 * suction_violations / n_steps,
        'Congestion_Pct': 100.0 * congestion_violations / n_steps,
        'Steps': n_steps
    }

    return summary, trajectories


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Independent Dual-Agent SAC Evaluation Harness")
    parser.add_argument("--weights_dir", type=str, default="./checkpoints_independent",
                        help="Path to directory containing actor_l_final.pth and actor_r_final.pth")
    parser.add_argument("--output_dir", type=str, default="results_independent",
                        help="Output directory for CSVs and figures")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if args.scenarios != "all":
        selected_raw = [s.strip() for s in args.scenarios.split(",")]
        run_scenarios = {
            s_id: s_info for s_id, s_info in SCENARIOS.items()
            if str(s_id) in selected_raw or s_info['name'] in selected_raw
        }
    else:
        run_scenarios = SCENARIOS

    print("=" * 80)
    print("      INDEPENDENT DUAL-AGENT SAC — COMPREHENSIVE EVALUATION HARNESS")
    print("=" * 80)

    act_l_path, act_r_path = find_independent_weights(args.weights_dir)
    print(f"[MODEL] Loading Left  Actor from: {act_l_path}")
    print(f"[MODEL] Loading Right Actor from: {act_r_path}")

    actor_l = Actor(obs_dim=5, act_dim=1).to(args.device)
    actor_r = Actor(obs_dim=5, act_dim=1).to(args.device)

    actor_l.load_state_dict(torch.load(act_l_path, map_location=args.device))
    actor_r.load_state_dict(torch.load(act_r_path, map_location=args.device))
    actor_l.eval()
    actor_r.eval()
    print("[MODEL] Dual agents loaded and set to eval mode.")

    summaries = []
    all_trajectories = []

    print(f"\n--- Running {len(run_scenarios)} Standardized Benchmark Scenario(s) ---")
    for s_id, s_info in run_scenarios.items():
        s_name = s_info['name']
        t_start = time.time()
        print(f"  [{s_id}/8] Evaluating Scenario: {s_name:20s} ... ", end='', flush=True)
        summary, traj = evaluate_scenario(actor_l, actor_r, s_id, s_name, s_info, device=args.device)
        dt = time.time() - t_start
        summaries.append(summary)
        all_trajectories.extend(traj)
        print(f"DONE ({dt:4.1f}s) | Mean LAP: {summary['Mean_LAP']:4.1f} | RAP: {summary['Mean_RAP']:4.1f} | FBE: {summary['Mean_FBE']:4.1f} mL/s | Suction: {summary['Suction_Pct']:3.1f}%")

    df_summary = pd.DataFrame(summaries)
    df_traj = pd.DataFrame(all_trajectories)

    overall_mean = {
        'Scenario': 'OVERALL_MEAN',
        'Mean_LAP': df_summary['Mean_LAP'].mean(),
        'Min_LAP': df_summary['Min_LAP'].min(),
        'Max_LAP': df_summary['Max_LAP'].max(),
        'Mean_RAP': df_summary['Mean_RAP'].mean(),
        'Min_RAP': df_summary['Min_RAP'].min(),
        'Max_RAP': df_summary['Max_RAP'].max(),
        'Mean_MAP': df_summary['Mean_MAP'].mean(),
        'Mean_CO': df_summary['Mean_CO'].mean(),
        'Mean_FBE': df_summary['Mean_FBE'].mean(),
        'Peak_FBE': df_summary['Peak_FBE'].max(),
        'Min_LVAD_RPM': df_summary['Min_LVAD_RPM'].min(),
        'Max_LVAD_RPM': df_summary['Max_LVAD_RPM'].max(),
        'Min_RVAD_RPM': df_summary['Min_RVAD_RPM'].min(),
        'Max_RVAD_RPM': df_summary['Max_RVAD_RPM'].max(),
        'Mean_Reward': df_summary['Mean_Reward'].mean(),
        'Total_Reward': df_summary['Total_Reward'].mean(),
        'Suction_Pct': df_summary['Suction_Pct'].mean(),
        'Congestion_Pct': df_summary['Congestion_Pct'].mean(),
        'Steps': df_summary['Steps'].sum()
    }
    df_summary_with_mean = pd.concat([df_summary, pd.DataFrame([overall_mean])], ignore_index=True)

    summary_csv = os.path.join(args.output_dir, "independent_summary_metrics.csv")
    traj_csv = os.path.join(args.output_dir, "independent_trajectories.csv")
    df_summary_with_mean.to_csv(summary_csv, index=False)
    df_traj.to_csv(traj_csv, index=False)
    print(f"\n[EXPORT] Saved summary metrics to:  {summary_csv}")
    print(f"[EXPORT] Saved trajectory dataset to: {traj_csv} ({len(df_traj):,} rows)")

    # Print Table
    print("\n" + "=" * 105)
    print(f"{'Scenario':20s} | {'Mean LAP':8s} | {'Mean RAP':8s} | {'Mean MAP':8s} | {'Mean CO':8s} | {'Mean FBE':8s} | {'Peak FBE':8s} | {'LVAD RPM':11s} | {'RVAD RPM':11s}")
    print("-" * 105)
    for _, row in df_summary.iterrows():
        print(f"{row['Scenario']:20s} | {row['Mean_LAP']:6.1f}   | {row['Mean_RAP']:6.1f}   | {row['Mean_MAP']:6.1f}   | {row['Mean_CO']:6.2f} L | {row['Mean_FBE']:6.2f}   | {row['Peak_FBE']:6.2f}   | {row['Min_LVAD_RPM']:.0f}-{row['Max_LVAD_RPM']:.0f} | {row['Min_RVAD_RPM']:.0f}-{row['Max_RVAD_RPM']:.0f}")
    print("-" * 105)
    print(f"{overall_mean['Scenario']:20s} | {overall_mean['Mean_LAP']:6.1f}   | {overall_mean['Mean_RAP']:6.1f}   | {overall_mean['Mean_MAP']:6.1f}   | {overall_mean['Mean_CO']:6.2f} L | {overall_mean['Mean_FBE']:6.2f}   | {overall_mean['Peak_FBE']:6.2f}   | {overall_mean['Min_LVAD_RPM']:.0f}-{overall_mean['Max_LVAD_RPM']:.0f} | {overall_mean['Min_RVAD_RPM']:.0f}-{overall_mean['Max_RVAD_RPM']:.0f}")
    print("=" * 105)


if __name__ == '__main__':
    main()

