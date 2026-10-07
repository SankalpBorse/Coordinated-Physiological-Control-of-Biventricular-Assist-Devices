"""
Phase 6 — Centralized SAC Controller Comprehensive Evaluation & Visualization Harness
======================================================================================
Evaluates the trained Centralized SAC Agent across all 8 standardized clinical scenarios:
  1. Rest
  2. SVR_up (Hypertensive shock / afterload increase)
  3. SVR_down (Vasodilation / afterload decrease)
  4. PVR_up (Pulmonary hypertension / right ventricular afterload increase)
  5. Exercise (Increased heart rate + vasodilation)
  6. Exercise_Recovery (Exercise onset -> rest recovery transition)
  7. Hypovolemia (20% circulating blood volume loss)
  8. Combined (Multifactorial severe hemodynamic stress)

Features:
  - Automatically loads the trained Centralized SAC model from './centralized_sac_results/checkpoints/sac_centralized_final.zip'
  - Evaluates all 8 scenarios with physiological perturbation schedules
  - Collects fine-grained time-series trajectories (16,000 steps total)
  - Computes complete hemodynamic, safety, and flow-balance metrics
  - Saves:
      * results_centralized/centralized_trajectories.csv
      * results_centralized/centralized_summary_metrics.csv
  - Generates high-resolution publication-quality figures in 'results_centralized/figures/':
      1. scenario_trajectories_dashboard.png (Time-series hemodynamics across key scenarios)
      2. flow_balance_error_all_scenarios.png (FBE bar chart across all 8 scenarios)
      3. atrial_pressure_safety_bands.png (LAP & RAP distributions vs clinical safety limits)
      4. pump_speed_dynamics.png (Dual-pump RPM phase portrait & speeds)
      5. training_loss_curves.png (Actor loss, Critic loss, and Alpha from TensorBoard logs)
      6. centralized_full_thesis_dashboard.png (Unified multi-panel master figure)
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure workspace root is in path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from stable_baselines3 import SAC
from bivad_env import BiVADEnv


# =====================================================================
# Scenario Definitions (Matches Benchmark Standards)
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
def find_centralized_model(candidate_path=None):
    """Finds the best available Centralized SAC model zip file."""
    if candidate_path and os.path.exists(candidate_path):
        return candidate_path

    search_locations = [
        "centralized_sac_results/checkpoints/sac_centralized_final.zip",
        "checkpoints/sac_centralized_final.zip",
        "centralized_sac_results/checkpoints/sac_centralized_500000_steps.zip",
        "checkpoints/sac_centralized_500000_steps.zip",
        "centralized_sac_results/sac_centralized_final.zip",
        "sac_centralized_final.zip"
    ]
    for loc in search_locations:
        p = os.path.join(WORKSPACE_ROOT, loc)
        if os.path.exists(p):
            return p
        if os.path.exists(loc):
            return loc

    # Check any sac_centralized_*.zip
    matches = glob.glob(os.path.join(WORKSPACE_ROOT, "centralized_sac_results", "checkpoints", "sac_centralized_*_steps.zip"))
    if matches:
        return sorted(matches)[-1]

    raise FileNotFoundError(
        "Could not locate 'sac_centralized_final.zip'. "
        "Please ensure the centralized model is in 'centralized_sac_results/checkpoints/' or specify --model_path."
    )


# =====================================================================
# Evaluation Function for a Single Scenario
# =====================================================================
def evaluate_scenario(model, scenario_id, scenario_name, scenario_dict):
    """Evaluates the Centralized SAC model on a specific scenario for 200s (2000 steps)."""
    env = BiVADEnv(randomize=False, raw_obs=False)
    env.perturbation_schedule = {}

    obs, _ = env.reset()

    # Apply time-0 changes if any (e.g. Hypovolemia)
    if 0 in scenario_dict['changes']:
        scenario_dict['changes'][0](env.params)

    trajectories = []
    totals, tracks, balances, safeties, smooths = [], [], [], [], []
    laps, raps, maps, cos, fbes = [], [], [], [], []
    lvad_rpms, rvad_rpms = [], []
    delta_lvads, delta_rvads = [], []

    suction_violations = 0
    congestion_violations = 0

    for step_i in range(env.max_steps):
        # Apply scheduled disturbances
        if step_i in scenario_dict['changes'] and step_i > 0:
            scenario_dict['changes'][step_i](env.params)

        # Predict action deterministically
        action, _ = model.predict(obs, deterministic=True)

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect states
        y = info.get('true_state', env.state)
        lap, rap = y[2], y[3]
        aop = y[4]
        lvad_flow = y[10]
        co = lvad_flow * 60.0 / 1000.0  # L/min
        fbe = info.get('flow_balance_error', abs(y[10] - y[11]))
        lvad_rpm = info.get('LVAD_RPM', env.params['LVAD_RPM'])
        rvad_rpm = info.get('RVAD_RPM', env.params['RVAD_RPM'])

        # Safety checks
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
        delta_lvads.append(action[0])
        delta_rvads.append(action[1])

        time_sec = round((step_i + 1) * env.dt, 2)
        trajectories.append({
            'Time': time_sec,
            'Controller': 'Centralized_SAC',
            'Scenario': scenario_name,
            'MAP': aop,
            'LAP': lap,
            'RAP': rap,
            'LVAD_RPM': lvad_rpm,
            'RVAD_RPM': rvad_rpm,
            'CO_left': co,
            'flow_balance_error': fbe,
            'Reward': reward,
            'r_track': info.get('reward_track', 0.0),
            'r_balance': info.get('reward_balance', 0.0),
            'r_safety': info.get('reward_safety', 0.0),
            'r_smooth': info.get('reward_smooth', 0.0),
            'Delta_LVAD': action[0],
            'Delta_RVAD': action[1]
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
# TensorBoard Training Log Extractor
# =====================================================================
def extract_tensorboard_training_data(tb_dir):
    """Extracts training metrics from TensorBoard event logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("[WARNING] tensorboard package not found. Skipping training log extraction.")
        return None

    files = sorted(glob.glob(os.path.join(tb_dir, "*", "events.out.tfevents.*")))
    if not files:
        files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not files:
        print(f"[INFO] No TensorBoard event files found in '{tb_dir}'.")
        return None

    all_data = {}
    for f in files:
        try:
            ea = event_accumulator.EventAccumulator(f)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                df = pd.DataFrame({'step': steps, 'value': vals})
                if tag not in all_data or len(df) > len(all_data[tag]):
                    all_data[tag] = df
        except Exception as e:
            print(f"[WARNING] Could not read TB event file {f}: {e}")

    return all_data


# =====================================================================
# Publication Plotting Routines
# =====================================================================
def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = '#e0e0e0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7


def plot_scenario_trajectories_dashboard(df_traj, out_path):
    """Creates a 4-scenario multi-panel dashboard: Rest, Exercise_Recovery, SVR_up, Hypovolemia."""
    set_plot_style()
    key_scenarios = ['Rest', 'Exercise_Recovery', 'SVR_up', 'Hypovolemia']
    available_scenarios = [s for s in key_scenarios if s in df_traj['Scenario'].unique()]
    if not available_scenarios:
        available_scenarios = list(df_traj['Scenario'].unique())[:4]

    fig, axes = plt.subplots(4, len(available_scenarios), figsize=(5.0 * len(available_scenarios), 12.0), sharex='col')
    if len(available_scenarios) == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, scen_name in enumerate(available_scenarios):
        sub = df_traj[df_traj['Scenario'] == scen_name].sort_values('Time')
        t = sub['Time'].values

        # Row 0: Atrial Pressures (LAP & RAP)
        ax0 = axes[0, col_idx]
        ax0.plot(t, sub['LAP'], color='#1f77b4', lw=2.0, label='LAP (Left)')
        ax0.plot(t, sub['RAP'], color='#ff7f0e', lw=2.0, label='RAP (Right)')
        ax0.axhspan(9.0, 12.0, color='#2ca02c', alpha=0.15, label='Nominal LAP Target')
        ax0.axhspan(4.0, 6.0, color='#ffbb78', alpha=0.20, label='Nominal RAP Target')
        ax0.axhline(3.0, color='red', ls=':', lw=1.5, label='Suction Limit (3 mmHg)')
        ax0.set_title(f"Scenario: {scen_name}", fontsize=13, fontweight='bold', pad=10)
        ax0.set_ylabel("Pressure (mmHg)", fontsize=11)
        ax0.set_ylim(0, 22)
        if col_idx == 0:
            ax0.legend(loc='upper right', fontsize=9, framealpha=0.85)

        # Row 1: Flow Balance Error
        ax1 = axes[1, col_idx]
        ax1.plot(t, sub['flow_balance_error'], color='#d62728', lw=2.0, label='Flow Balance Error')
        ax1.axhline(20.0, color='#7f7f7f', ls='--', lw=1.5, label='Safety Threshold (20 mL/s)')
        ax1.fill_between(t, 0, sub['flow_balance_error'], color='#d62728', alpha=0.15)
        ax1.set_ylabel("FBE (mL/s)", fontsize=11)
        ax1.set_ylim(0, max(45.0, sub['flow_balance_error'].max() * 1.15))
        if col_idx == 0:
            ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)

        # Row 2: Pump Speeds (LVAD & RVAD RPM)
        ax2 = axes[2, col_idx]
        ax2.plot(t, sub['LVAD_RPM'], color='#9467bd', lw=2.0, label='LVAD Speed')
        ax2.plot(t, sub['RVAD_RPM'], color='#8c564b', lw=2.0, label='RVAD Speed')
        ax2.set_ylabel("Pump Speed (RPM)", fontsize=11)
        ax2.set_ylim(1600, 4200)
        if col_idx == 0:
            ax2.legend(loc='lower right', fontsize=9, framealpha=0.85)

        # Row 3: MAP & Cardiac Output
        ax3 = axes[3, col_idx]
        ax3.plot(t, sub['MAP'], color='#2ca02c', lw=2.0, label='MAP (Left axis)')
        ax3.set_ylabel("MAP (mmHg)", color='#2ca02c', fontsize=11)
        ax3.set_xlabel("Time (seconds)", fontsize=11)
        ax3.set_ylim(50, 170)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(t, sub['CO_left'], color='#e377c2', lw=1.8, ls='--', label='CO (Right axis)')
        ax3_twin.set_ylabel("CO (L/min)", color='#e377c2', fontsize=11)
        ax3_twin.set_ylim(2.0, 11.0)
        ax3_twin.grid(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_flow_balance_error_all_scenarios(df_summary, out_path):
    """Bar chart comparing Mean & Peak Flow Balance Error across all 8 scenarios."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(df_summary))
    width = 0.38

    bars_mean = ax.bar(x - width/2, df_summary['Mean_FBE'], width, label='Mean Flow Balance Error', color='#1f77b4', edgecolor='black', alpha=0.85)
    bars_peak = ax.bar(x + width/2, df_summary['Peak_FBE'], width, label='Peak Flow Balance Error', color='#ff7f0e', edgecolor='black', alpha=0.85)

    ax.axhline(20.0, color='red', ls='--', lw=1.8, label='Clinical Safety Threshold (20 mL/s)')
    ax.set_ylabel("Flow Balance Error (mL/s)", fontsize=12, fontweight='bold')
    ax.set_title("Centralized SAC: Inter-Ventricular Flow Balance Error Across 8 Benchmark Scenarios", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary['Scenario'], rotation=25, ha='right', fontsize=11, fontweight='medium')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(df_summary['Peak_FBE'].max() * 1.25, 30))

    # Add numeric labels on bars
    for bar in bars_mean:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars_peak:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#a04000')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_atrial_pressure_safety_bands(df_traj, out_path):
    """Generates boxplots of LAP & RAP across all scenarios showing adherence to clinical bounds."""
    set_plot_style()
    fig, (ax_lap, ax_rap) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    scenarios = list(df_traj['Scenario'].unique())
    lap_data = [df_traj[df_traj['Scenario'] == s]['LAP'].values for s in scenarios]
    rap_data = [df_traj[df_traj['Scenario'] == s]['RAP'].values for s in scenarios]

    # LAP Boxplot
    bp_lap = ax_lap.boxplot(lap_data, tick_labels=scenarios, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow', markersize=6))
    for patch in bp_lap['boxes']:
        patch.set_facecolor('#a6cee3')
        patch.set_edgecolor('#1f78b4')
    ax_lap.axhspan(9.0, 12.0, color='#33a02c', alpha=0.20, label='Nominal Target (9 - 12 mmHg)')
    ax_lap.axhline(3.0, color='red', ls=':', lw=2.0, label='Left Suction Threshold (< 3 mmHg)')
    ax_lap.axhline(18.0, color='#e31a1c', ls='--', lw=1.5, label='Congestion Threshold (> 18 mmHg)')
    ax_lap.set_title("Left Atrial Pressure (LAP) Adherence", fontsize=12, fontweight='bold')
    ax_lap.set_ylabel("LAP (mmHg)", fontsize=11)
    ax_lap.set_xticklabels(scenarios, rotation=35, ha='right', fontsize=10)
    ax_lap.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_lap.set_ylim(0, 24)

    # RAP Boxplot
    bp_rap = ax_rap.boxplot(rap_data, tick_labels=scenarios, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow', markersize=6))
    for patch in bp_rap['boxes']:
        patch.set_facecolor('#fdbf6f')
        patch.set_edgecolor('#ff7f00')
    ax_rap.axhspan(4.0, 6.0, color='#33a02c', alpha=0.20, label='Nominal Target (4 - 6 mmHg)')
    ax_rap.axhline(1.0, color='red', ls=':', lw=2.0, label='Right Suction Threshold (< 1 mmHg)')
    ax_rap.axhline(12.0, color='#e31a1c', ls='--', lw=1.5, label='Congestion Threshold (> 12 mmHg)')
    ax_rap.set_title("Right Atrial Pressure (RAP) Adherence", fontsize=12, fontweight='bold')
    ax_rap.set_ylabel("RAP (mmHg)", fontsize=11)
    ax_rap.set_xticklabels(scenarios, rotation=35, ha='right', fontsize=10)
    ax_rap.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_rap.set_ylim(0, 16)

    fig.suptitle("Centralized SAC: Hemodynamic Safety & Suction Prevention Bounds", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_dual_pump_speed_envelope(df_traj, out_path):
    """Plots the dual-pump coordination operating envelope (RVAD RPM vs LVAD RPM)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    scenarios = df_traj['Scenario'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for idx, scen in enumerate(scenarios):
        sub = df_traj[df_traj['Scenario'] == scen]
        ax.scatter(sub['LVAD_RPM'], sub['RVAD_RPM'], color=colors[idx], label=scen, alpha=0.45, s=15, edgecolors='none')

    ax.set_xlabel("LVAD Speed (RPM)", fontsize=12, fontweight='bold')
    ax.set_ylabel("RVAD Speed (RPM)", fontsize=12, fontweight='bold')
    ax.set_title("Centralized SAC: Dual-Pump Speed Coordination Operating Envelope", fontsize=13, fontweight='bold', pad=12)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(1700, 4000)
    ax.set_ylim(1700, 3000)

    # Reference diagonal line
    ax.plot([1800, 3800], [1800, 3800], color='gray', ls=':', alpha=0.6, label='Equal Speeds (1:1)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_training_phase_curves(tb_data, out_path):
    """Plots Actor Loss, Critic Loss, Entropy Temperature Alpha, and FPS from TensorBoard data."""
    if not tb_data:
        print("[INFO] No TensorBoard data available to plot training phase curves.")
        return

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Critic Loss
    ax = axes[0, 0]
    if 'train/critic_loss' in tb_data:
        df = tb_data['train/critic_loss']
        ax.plot(df['step'], df['value'], color='#1f77b4', lw=1.2, alpha=0.5, label='Raw Critic Loss')
        rolling = df['value'].rolling(window=5, min_periods=1).mean()
        ax.plot(df['step'], rolling, color='#08519c', lw=2.2, label='Smoothed (MA-5)')
        ax.set_title("Twin-Critic Loss (MSE)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Critic Loss", fontsize=11)
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Critic Loss not found", ha='center', va='center')

    # 2. Actor Loss
    ax = axes[0, 1]
    if 'train/actor_loss' in tb_data:
        df = tb_data['train/actor_loss']
        ax.plot(df['step'], df['value'], color='#ff7f0e', lw=1.2, alpha=0.5, label='Raw Actor Loss')
        rolling = df['value'].rolling(window=5, min_periods=1).mean()
        ax.plot(df['step'], rolling, color='#d95f02', lw=2.2, label='Smoothed (MA-5)')
        ax.set_title("Policy Actor Loss", fontsize=12, fontweight='bold')
        ax.set_ylabel("Actor Loss", fontsize=11)
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Actor Loss not found", ha='center', va='center')

    # 3. Entropy Coefficient (Alpha)
    ax = axes[1, 0]
    if 'train/ent_coef' in tb_data:
        df = tb_data['train/ent_coef']
        ax.plot(df['step'], df['value'], color='#2ca02c', lw=2.0)
        ax.set_title("Automatic Entropy Temperature (Alpha)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Alpha", fontsize=11)
        ax.set_xlabel("Training Steps", fontsize=11)
    else:
        ax.text(0.5, 0.5, "Alpha not found", ha='center', va='center')

    # 4. Training FPS (Throughput)
    ax = axes[1, 1]
    if 'time/fps' in tb_data:
        df = tb_data['time/fps']
        ax.plot(df['step'], df['value'], color='#9467bd', lw=2.0)
        mean_fps = df['value'].mean()
        ax.axhline(mean_fps, color='black', ls='--', lw=1.5, label=f"Mean: {mean_fps:.1f} steps/s")
        ax.set_title("Training Throughput (Parallel SubprocVecEnv)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Speed (steps/sec)", fontsize=11)
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "FPS not found", ha='center', va='center')

    fig.suptitle("Centralized SAC Training Phase Progression (500,000 Steps on Kaggle)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_centralized_full_thesis_dashboard(df_summary, df_traj, tb_data, out_path):
    """Master multi-panel summary figure for thesis inclusion."""
    set_plot_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Panel 1: Rest Scenario Atrial Pressures (LAP & RAP)
    ax1 = fig.add_subplot(gs[0, 0])
    sub_rest = df_traj[df_traj['Scenario'] == 'Rest']
    if not sub_rest.empty:
        ax1.plot(sub_rest['Time'], sub_rest['LAP'], color='#1f77b4', lw=2.0, label='LAP')
        ax1.plot(sub_rest['Time'], sub_rest['RAP'], color='#ff7f0e', lw=2.0, label='RAP')
        ax1.axhspan(9.0, 12.0, color='#2ca02c', alpha=0.15)
        ax1.axhspan(4.0, 6.0, color='#ffbb78', alpha=0.20)
        ax1.axhline(3.0, color='red', ls=':', lw=1.5)
        ax1.set_title("(A) Rest: Atrial Pressures", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Pressure (mmHg)", fontsize=10)
        ax1.set_ylim(0, 20)
        ax1.legend(fontsize=8, loc='upper right')

    # Panel 2: Rest Scenario Pump Speeds
    ax2 = fig.add_subplot(gs[0, 1])
    if not sub_rest.empty:
        ax2.plot(sub_rest['Time'], sub_rest['LVAD_RPM'], color='#9467bd', lw=2.0, label='LVAD')
        ax2.plot(sub_rest['Time'], sub_rest['RVAD_RPM'], color='#8c564b', lw=2.0, label='RVAD')
        ax2.set_title("(B) Rest: Pump Speeds", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Speed (RPM)", fontsize=10)
        ax2.set_ylim(1600, 4000)
        ax2.legend(fontsize=8, loc='lower right')

    # Panel 3: Rest Scenario Flow Balance Error
    ax3 = fig.add_subplot(gs[0, 2])
    if not sub_rest.empty:
        ax3.plot(sub_rest['Time'], sub_rest['flow_balance_error'], color='#d62728', lw=2.0)
        ax3.axhline(20.0, color='gray', ls='--', lw=1.2)
        ax3.fill_between(sub_rest['Time'], 0, sub_rest['flow_balance_error'], color='#d62728', alpha=0.15)
        ax3.set_title("(C) Rest: Flow Balance Error", fontsize=11, fontweight='bold')
        ax3.set_ylabel("FBE (mL/s)", fontsize=10)
        ax3.set_ylim(0, 40)

    # Panel 4: Exercise_Recovery Scenario Hemodynamics
    ax4 = fig.add_subplot(gs[1, 0])
    sub_ex = df_traj[df_traj['Scenario'] == 'Exercise_Recovery']
    if not sub_ex.empty:
        ax4.plot(sub_ex['Time'], sub_ex['LAP'], color='#1f77b4', lw=2.0, label='LAP')
        ax4.plot(sub_ex['Time'], sub_ex['RAP'], color='#ff7f0e', lw=2.0, label='RAP')
        ax4.axvline(50.0, color='gray', ls='--', alpha=0.7)
        ax4.axvline(120.0, color='gray', ls=':', alpha=0.7)
        ax4.set_title("(D) Exercise-Recovery: Pressures", fontsize=11, fontweight='bold')
        ax4.set_ylabel("Pressure (mmHg)", fontsize=10)
        ax4.set_ylim(0, 22)
        ax4.legend(fontsize=8, loc='upper right')

    # Panel 5: Exercise_Recovery Scenario Pump Speeds
    ax5 = fig.add_subplot(gs[1, 1])
    if not sub_ex.empty:
        ax5.plot(sub_ex['Time'], sub_ex['LVAD_RPM'], color='#9467bd', lw=2.0, label='LVAD')
        ax5.plot(sub_ex['Time'], sub_ex['RVAD_RPM'], color='#8c564b', lw=2.0, label='RVAD')
        ax5.axvline(50.0, color='gray', ls='--', alpha=0.7)
        ax5.axvline(120.0, color='gray', ls=':', alpha=0.7)
        ax5.set_title("(E) Exercise-Recovery: Speed Ramping", fontsize=11, fontweight='bold')
        ax5.set_ylabel("Speed (RPM)", fontsize=10)
        ax5.set_ylim(1600, 4200)
        ax5.legend(fontsize=8, loc='lower right')

    # Panel 6: Flow Balance Error Comparison across all 8 scenarios
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(df_summary))
    ax6.bar(x, df_summary['Mean_FBE'], color='#1f77b4', edgecolor='black', alpha=0.85)
    ax6.axhline(20.0, color='red', ls='--', lw=1.5)
    ax6.set_xticks(x)
    ax6.set_xticklabels(df_summary['Scenario'], rotation=40, ha='right', fontsize=8)
    ax6.set_title("(F) Mean Flow Balance Error (8 Scenarios)", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Mean FBE (mL/s)", fontsize=10)
    ax6.set_ylim(0, 30)

    # Panel 7: Training Critic Loss
    ax7 = fig.add_subplot(gs[2, 0])
    if tb_data and 'train/critic_loss' in tb_data:
        df_c = tb_data['train/critic_loss']
        ax7.plot(df_c['step'], df_c['value'], color='#1f77b4', lw=1.0, alpha=0.4)
        ax7.plot(df_c['step'], df_c['value'].rolling(5, min_periods=1).mean(), color='#08519c', lw=2.0)
        ax7.set_title("(G) Critic MSE Loss Convergence", fontsize=11, fontweight='bold')
        ax7.set_ylabel("Critic Loss", fontsize=10)
        ax7.set_xlabel("Steps", fontsize=10)
    else:
        ax7.text(0.5, 0.5, "Critic Loss N/A", ha='center', va='center')

    # Panel 8: Training Policy Actor Loss
    ax8 = fig.add_subplot(gs[2, 1])
    if tb_data and 'train/actor_loss' in tb_data:
        df_a = tb_data['train/actor_loss']
        ax8.plot(df_a['step'], df_a['value'], color='#ff7f0e', lw=1.0, alpha=0.4)
        ax8.plot(df_a['step'], df_a['value'].rolling(5, min_periods=1).mean(), color='#d95f02', lw=2.0)
        ax8.set_title("(H) Policy Actor Loss Progression", fontsize=11, fontweight='bold')
        ax8.set_ylabel("Actor Loss", fontsize=10)
        ax8.set_xlabel("Steps", fontsize=10)
    else:
        ax8.text(0.5, 0.5, "Actor Loss N/A", ha='center', va='center')

    # Panel 9: Dual Pump Operating Envelope
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(df_traj['LVAD_RPM'], df_traj['RVAD_RPM'], c='#3182bd', alpha=0.25, s=8, edgecolors='none')
    ax9.plot([1800, 3800], [1800, 3800], color='gray', ls=':', lw=1.5)
    ax9.set_title("(I) Dual-Pump Speed Coordination Envelope", fontsize=11, fontweight='bold')
    ax9.set_xlabel("LVAD Speed (RPM)", fontsize=10)
    ax9.set_ylabel("RVAD Speed (RPM)", fontsize=10)
    ax9.set_xlim(1800, 4000)
    ax9.set_ylim(1800, 2600)

    fig.suptitle("Centralized SAC Controller — Master Physiological & Training Performance Dashboard",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {out_path}")


# =====================================================================
# Main Execution Pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Centralized SAC Comprehensive Evaluation Harness")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained sac_centralized_final.zip (auto-searched if omitted)")
    parser.add_argument("--tb_dir", type=str, default="centralized_sac_results/tb_centralized",
                        help="Path to TensorBoard event directory")
    parser.add_argument("--output_dir", type=str, default="results_centralized",
                        help="Directory to store evaluation CSVs and figures")
    parser.add_argument("--scenarios", type=str, default="all",
                        help="Comma-separated scenario IDs (1-8) or names to run, or 'all' (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Filter scenarios if specified
    if args.scenarios != "all":
        selected_raw = [s.strip() for s in args.scenarios.split(",")]
        run_scenarios = {
            s_id: s_info for s_id, s_info in SCENARIOS.items()
            if str(s_id) in selected_raw or s_info['name'] in selected_raw
        }
        if not run_scenarios:
            print(f"[ERROR] No matching scenarios found for: {args.scenarios}")
            return
    else:
        run_scenarios = SCENARIOS

    print("=" * 80)
    print("      CENTRALIZED SAC CONTROLLER — COMPREHENSIVE EVALUATION HARNESS")
    print("=" * 80)

    # 1. Locate and load model
    resolved_model_path = find_centralized_model(args.model_path)
    print(f"[MODEL] Loading Centralized SAC policy from: {resolved_model_path}")
    t_load_start = time.time()
    model = SAC.load(resolved_model_path, device=args.device)
    print(f"[MODEL] Successfully loaded in {time.time() - t_load_start:.2f} seconds.")

    # 2. Evaluate across standardized clinical scenarios
    print(f"\n--- Running {len(run_scenarios)} Standardized Benchmark Scenario(s) (200s each = {len(run_scenarios)*2000:,} steps) ---")
    summaries = []
    all_trajectories = []

    for s_id, s_info in run_scenarios.items():
        s_name = s_info['name']
        t_start = time.time()
        print(f"  [{s_id}/8] Evaluating Scenario: {s_name:20s} ... ", end='', flush=True)
        summary, traj = evaluate_scenario(model, s_id, s_name, s_info)
        dt = time.time() - t_start
        summaries.append(summary)
        all_trajectories.extend(traj)
        print(f"DONE ({dt:4.1f}s) | Mean LAP: {summary['Mean_LAP']:4.1f} | RAP: {summary['Mean_RAP']:4.1f} | FBE: {summary['Mean_FBE']:4.1f} mL/s | Suction: {summary['Suction_Pct']:3.1f}%")

    # 3. Create Summary DataFrames
    df_summary = pd.DataFrame(summaries)
    df_traj = pd.DataFrame(all_trajectories)

    # Add Mean Overall Row to Summary Table
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

    # 4. Save CSV Datasets
    summary_csv = os.path.join(args.output_dir, "centralized_summary_metrics.csv")
    traj_csv = os.path.join(args.output_dir, "centralized_trajectories.csv")
    df_summary_with_mean.to_csv(summary_csv, index=False)
    df_traj.to_csv(traj_csv, index=False)
    print(f"\n[EXPORT] Saved summary metrics to:  {summary_csv}")
    print(f"[EXPORT] Saved trajectory dataset to: {traj_csv} ({len(df_traj):,} rows)")

    # 5. Print High-Visibility Console Table
    print("\n" + "=" * 105)
    print(f"{'Scenario':20s} | {'Mean LAP':8s} | {'Mean RAP':8s} | {'Mean MAP':8s} | {'Mean CO':8s} | {'Mean FBE':8s} | {'Peak FBE':8s} | {'LVAD RPM':11s} | {'RVAD RPM':11s}")
    print("-" * 105)
    for _, row in df_summary.iterrows():
        print(f"{row['Scenario']:20s} | {row['Mean_LAP']:6.1f}   | {row['Mean_RAP']:6.1f}   | {row['Mean_MAP']:6.1f}   | {row['Mean_CO']:6.2f} L | {row['Mean_FBE']:6.2f}   | {row['Peak_FBE']:6.2f}   | {row['Min_LVAD_RPM']:.0f}-{row['Max_LVAD_RPM']:.0f} | {row['Min_RVAD_RPM']:.0f}-{row['Max_RVAD_RPM']:.0f}")
    print("-" * 105)
    print(f"{overall_mean['Scenario']:20s} | {overall_mean['Mean_LAP']:6.1f}   | {overall_mean['Mean_RAP']:6.1f}   | {overall_mean['Mean_MAP']:6.1f}   | {overall_mean['Mean_CO']:6.2f} L | {overall_mean['Mean_FBE']:6.2f}   | {overall_mean['Peak_FBE']:6.2f}   | {overall_mean['Min_LVAD_RPM']:.0f}-{overall_mean['Max_LVAD_RPM']:.0f} | {overall_mean['Min_RVAD_RPM']:.0f}-{overall_mean['Max_RVAD_RPM']:.0f}")
    print("=" * 105)

    # 6. Extract TensorBoard Training Data
    print("\n--- Extracting Training Phase Data from TensorBoard ---")
    tb_data = extract_tensorboard_training_data(args.tb_dir)
    if tb_data:
        print(f"[TENSORBOARD] Extracted {len(tb_data)} training scalar tags successfully.")

    # 7. Generate Publication-Quality Visualizations
    print("\n--- Generating High-Resolution Publication Figures ---")
    plot_scenario_trajectories_dashboard(df_traj, os.path.join(fig_dir, "scenario_trajectories_dashboard.png"))
    plot_flow_balance_error_all_scenarios(df_summary, os.path.join(fig_dir, "flow_balance_error_all_scenarios.png"))
    plot_atrial_pressure_safety_bands(df_traj, os.path.join(fig_dir, "atrial_pressure_safety_bands.png"))
    plot_dual_pump_speed_envelope(df_traj, os.path.join(fig_dir, "pump_speed_operating_envelope.png"))
    if tb_data:
        plot_training_phase_curves(tb_data, os.path.join(fig_dir, "training_phase_curves.png"))
    plot_centralized_full_thesis_dashboard(df_summary, df_traj, tb_data, os.path.join(fig_dir, "centralized_full_thesis_dashboard.png"))

    print("\n" + "=" * 80)
    print("      CENTRALIZED SAC EVALUATION & VISUALIZATION COMPLETE!")
    print(f"      All results and figures saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == '__main__':
    main()

