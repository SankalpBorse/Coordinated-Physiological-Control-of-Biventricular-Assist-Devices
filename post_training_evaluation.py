"""
Post-Training Evaluation & Visualization Suite
==============================================
Automatically generates publication-grade figures and performance tables
immediately after training completes, saving them inside the seed checkpoint
folder for automatic inclusion in the downloadable zip archive.

Artifacts Generated:
  1. tables/performance_table.csv & tables/performance_table.md (All 9 clinical scenarios)
  2. plots/clinical_dynamics_dashboard.png (Multi-scenario 4-panel hemodynamic trajectories)
  3. plots/suction_safety_response.png (Dedicated Scenario 9 acute hypovolemia boundary dynamics)
  4. plots/training_curves.png (TensorBoard loss, reward, and component progression)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local imports work
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from evaluate_all import SCENARIOS, run_episode


# =============================================================================
# Adapters for Unified Evaluation Interface
# =============================================================================
class CentralizedModelAdapter:
    """Wraps an SB3 SAC model for standardized evaluation."""
    def __init__(self, model, name="SAC_Centralized"):
        self.model = model
        self.name = name

    def reset(self):
        pass

    def get_action(self, obs, info):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class IndependentModelAdapter:
    """Wraps dual independent SAC agents for standardized evaluation."""
    def __init__(self, agent_left, agent_right, name="SAC_Independent"):
        self.agent_left = agent_left
        self.agent_right = agent_right
        self.name = name
        self.left_indices = [0, 2, 4, 6, 8]
        self.right_indices = [1, 3, 5, 7, 8]

    def reset(self):
        pass

    def get_action(self, obs, info):
        o_l = obs[self.left_indices]
        o_r = obs[self.right_indices]
        a_l = self.agent_left.select_action(o_l, deterministic=True)
        a_r = self.agent_right.select_action(o_r, deterministic=True)
        return np.array([a_l[0], a_r[0]], dtype=np.float32)


# =============================================================================
# 1. Multi-Scenario Benchmark Evaluation
# =============================================================================
def evaluate_all_scenarios(adapter, seed=42):
    """
    Evaluates the model across all 9 standardized scenarios.
    Returns:
      df_metrics     : DataFrame containing summary metrics per scenario
      trajectories   : Dictionary of {scenario_name: df_trajectory}
    """
    print("\n--- Running 9-Scenario Clinical Evaluation ---")
    rows = []
    trajectories = {}

    for s_id in sorted(SCENARIOS.keys()):
        s_name = SCENARIOS[s_id]['name']
        save_traj = s_name in ['Rest', 'SVR_up', 'Exercise_Recovery', 'Suction_Test']
        metrics, traj_rows = run_episode(adapter, scenario_id=s_id, seed=seed, save_trajectory=save_traj)
        rows.append(metrics)
        if save_traj and traj_rows:
            trajectories[s_name] = pd.DataFrame(traj_rows)
        print(f"  Scenario [{s_id}/9] {s_name:<18s}: "
              f"MAP={metrics['MAP_mean']:5.1f} mmHg | "
              f"CO={metrics['CO_mean']:4.2f} L/min | "
              f"FBE={metrics['FBE_mean']:4.1f} mL/s | "
              f"Reward={metrics['Mean_Reward']:6.2f}")

    df_metrics = pd.DataFrame(rows)
    return df_metrics, trajectories


# =============================================================================
# 2. Plotting: Multi-Scenario Dynamics Dashboard
# =============================================================================
def plot_dynamics_dashboard(trajectories, save_path):
    """
    Generates a 4-panel comparison dashboard across key clinical scenarios:
      Panel 1: Left Atrial Pressure (LAP) with 9.0 target and 3.0 suction limit
      Panel 2: Flow Balance Error (FBE) with 20.0 mL/s safety threshold
      Panel 3: Mean Arterial Pressure (MAP) with 70-120 mmHg physiological band
      Panel 4: LVAD and RVAD Motor Speeds (RPM)
    """
    scenarios = [s for s in ['Rest', 'SVR_up', 'Exercise_Recovery', 'Suction_Test'] if s in trajectories]
    if not scenarios:
        return

    n_cols = len(scenarios)
    fig, axes = plt.subplots(4, n_cols, figsize=(4.8 * n_cols, 11.0), sharex='col')
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, s_name in enumerate(scenarios):
        df = trajectories[s_name].sort_values('Time')
        t = df['Time'].values

        # 1. LAP & RAP
        axes[0, col_idx].plot(t, df['LAP'], color='#2ca02c', lw=1.8, label='LAP')
        if 'RAP' in df.columns:
            axes[0, col_idx].plot(t, df['RAP'], color='#1f77b4', lw=1.2, ls=':', label='RAP')
        axes[0, col_idx].axhline(9.0, color='darkgreen', ls='--', lw=1.0, label='Target (9.0)' if col_idx == 0 else "")
        axes[0, col_idx].axhline(3.0, color='crimson', ls=':', lw=1.2, label='Suction Limit (3.0)' if col_idx == 0 else "")
        axes[0, col_idx].set_title(f"Scenario: {s_name}", fontsize=11, fontweight='bold')
        axes[0, col_idx].set_ylabel("Pressure (mmHg)", fontsize=9)
        axes[0, col_idx].set_ylim(0, 22)
        axes[0, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axes[0, col_idx].legend(loc='upper right', fontsize=8)

        # 2. Flow Balance Error
        axes[1, col_idx].plot(t, df['flow_balance_error'], color='#d62728', lw=1.8, label='FBE')
        axes[1, col_idx].axhline(20.0, color='black', ls='--', lw=1.0, label='Safety Limit (20.0)' if col_idx == 0 else "")
        axes[1, col_idx].set_ylabel("FBE (mL/s)", fontsize=9)
        max_fbe = max(35.0, df['flow_balance_error'].max() * 1.15)
        axes[1, col_idx].set_ylim(0, max_fbe)
        axes[1, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axes[1, col_idx].legend(loc='upper right', fontsize=8)

        # 3. MAP & CO
        axes[2, col_idx].plot(t, df['MAP'], color='#9467bd', lw=1.8, label='MAP (mmHg)')
        axes[2, col_idx].axhspan(70, 120, color='green', alpha=0.1, label='Target Band (70-120)' if col_idx == 0 else "")
        axes[2, col_idx].set_ylabel("MAP (mmHg)", fontsize=9)
        axes[2, col_idx].set_ylim(50, 160)
        axes[2, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axes[2, col_idx].legend(loc='upper right', fontsize=8)

        # 4. Pump Speeds
        axes[3, col_idx].plot(t, df['LVAD_RPM'], color='#e377c2', lw=1.8, label='LVAD RPM')
        axes[3, col_idx].plot(t, df['RVAD_RPM'], color='#17becf', lw=1.8, ls='--', label='RVAD RPM')
        axes[3, col_idx].set_ylabel("Pump Speed (RPM)", fontsize=9)
        axes[3, col_idx].set_xlabel("Time (s)", fontsize=9)
        axes[3, col_idx].set_ylim(1700, 3900)
        axes[3, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axes[3, col_idx].legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved dynamics dashboard to: {save_path}")


# =============================================================================
# 3. Plotting: Dedicated Suction Response (Scenario 9)
# =============================================================================
def plot_suction_response(df_suction, save_path):
    """
    Dedicated visualization of the acute hypovolemia boundary test (Scenario 9).
    Demonstrates active cannula collapse avoidance and subsequent fluid resuscitation.
    """
    if df_suction is None or len(df_suction) == 0:
        return

    df = df_suction.sort_values('Time')
    t = df['Time'].values

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True)

    # 1. Pressures
    ax1.plot(t, df['LAP'], color='green', lw=2.0, label='LAP')
    if 'RAP' in df.columns:
        ax1.plot(t, df['RAP'], color='blue', lw=1.5, ls=':', label='RAP')
    ax1.axhline(9.0, color='darkgreen', ls='--', lw=1.0, label='Target LAP (9.0)')
    ax1.axhline(3.0, color='red', ls=':', lw=1.5, label='Suction Threshold (3.0 mmHg)')
    ax1.axvspan(50.0, 120.0, color='red', alpha=0.12, label='Volume Evacuation Phase (70% Loss)')
    ax1.set_ylabel("Atrial Pressure (mmHg)")
    ax1.set_ylim(-1, 18)
    ax1.set_title("Scenario 9: Acute Hypovolemia Boundary Dynamics & Suction Prevention", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Pump Speeds
    ax2.plot(t, df['LVAD_RPM'], color='#e377c2', lw=2.0, label='LVAD Speed')
    ax2.plot(t, df['RVAD_RPM'], color='#17becf', lw=2.0, ls='--', label='RVAD Speed')
    ax2.axvspan(50.0, 120.0, color='red', alpha=0.12)
    ax2.set_ylabel("Speed (RPM)")
    ax2.set_ylim(1700, 3900)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=8)

    # 3. Flow Balance & Per-Step Reward
    ax3.plot(t, df['flow_balance_error'], color='crimson', lw=1.8, label='Flow Balance Error (mL/s)')
    ax3.plot(t, df['reward'], color='black', lw=1.2, ls='-.', alpha=0.7, label='Step Reward')
    ax3.axvspan(50.0, 120.0, color='red', alpha=0.12)
    ax3.set_ylabel("Metric / Reward")
    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved suction response figure to: {save_path}")


# =============================================================================
# 4. Plotting: Training Curves from TensorBoard Logs
# =============================================================================
def plot_training_curves_from_tb(tb_dir, save_path):
    """
    Extracts scalar events from TensorBoard logs and plots training convergence curves.
    If TensorBoard event reading fails, falls back gracefully.
    """
    if not tb_dir or not os.path.exists(tb_dir):
        return

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True)
        if not event_files:
            return

        ea = EventAccumulator(event_files[0])
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if not tags:
            return

        def get_series(tag):
            if tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                return steps, vals
            return [], []

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

        # Reward
        r_step, r_val = get_series("reward/raw_step_mean") or get_series("rollout/ep_rew_mean")
        if r_step:
            axes[0, 0].plot(r_step, r_val, color='green', lw=1.5)
            axes[0, 0].set_title("Step / Episode Reward", fontsize=10, fontweight='bold')
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True, alpha=0.3)

        # Losses
        c_step, c_val = get_series("train/critic_loss") or get_series("loss/critic_left")
        a_step, a_val = get_series("train/actor_loss") or get_series("loss/actor_left")
        if c_step:
            axes[0, 1].plot(c_step, c_val, color='red', lw=1.2, label='Critic Loss')
        if a_step:
            axes[0, 1].plot(a_step, a_val, color='blue', lw=1.2, label='Actor Loss')
        axes[0, 1].set_title("Actor & Critic Losses", fontsize=10, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='best', fontsize=8)

        # Clinical: MAP
        m_step, m_val = get_series("clinical/MAP") or get_series("clinical/clinical/MAP")
        if m_step:
            axes[1, 0].plot(m_step, m_val, color='purple', lw=1.5)
            axes[1, 0].axhspan(70, 120, color='green', alpha=0.1)
            axes[1, 0].set_title("Mean Arterial Pressure (mmHg)", fontsize=10, fontweight='bold')
            axes[1, 0].set_ylabel("MAP (mmHg)")
            axes[1, 0].set_xlabel("Timesteps")
            axes[1, 0].grid(True, alpha=0.3)

        # Clinical: Flow Balance Error
        f_step, f_val = get_series("clinical/flow_balance_error") or get_series("clinical/clinical/flow_balance_error")
        if f_step:
            axes[1, 1].plot(f_step, f_val, color='crimson', lw=1.5)
            axes[1, 1].axhline(20.0, color='black', ls='--', lw=1.0)
            axes[1, 1].set_title("Flow Balance Error (mL/s)", fontsize=10, fontweight='bold')
            axes[1, 1].set_ylabel("FBE (mL/s)")
            axes[1, 1].set_xlabel("Timesteps")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  Saved training curves to: {save_path}")
    except Exception as e:
        print(f"  [NOTE] Could not parse TensorBoard curves: {e}")


# =============================================================================
# 5. Main Public Workflow: Generate All Post-Training Reports
# =============================================================================
def generate_post_training_artifacts(adapter, output_dir="./checkpoints", tb_dir=None, seed=42):
    """
    Executes the entire post-training evaluation, reporting, and visualization suite.
    Artifacts are stored inside output_dir/tables/ and output_dir/plots/.
    """
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"GENERATING POST-TRAINING ARTIFACTS: {getattr(adapter, 'name', 'DRL_Agent')} (Seed {seed})")
    print("="*70)

    # 1. Run 9-scenario evaluation
    df_metrics, trajectories = evaluate_all_scenarios(adapter, seed=seed)

    # 2. Save performance tables
    csv_path = os.path.join(tables_dir, "performance_table.csv")
    md_path = os.path.join(tables_dir, "performance_table.md")
    df_metrics.to_csv(csv_path, index=False)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Performance Summary Table: {getattr(adapter, 'name', 'DRL_Agent')} (Seed {seed})\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n")
    print(f"  Saved performance tables to:\n    - {csv_path}\n    - {md_path}")

    # 3. Generate figures
    plot_dynamics_dashboard(trajectories, os.path.join(plots_dir, "clinical_dynamics_dashboard.png"))
    if 'Suction_Test' in trajectories:
        plot_suction_response(trajectories['Suction_Test'], os.path.join(plots_dir, "suction_safety_response.png"))
    if tb_dir:
        plot_training_curves_from_tb(tb_dir, os.path.join(plots_dir, "training_curves.png"))

    print("="*70)
    print(">>> All post-training plots and tables successfully generated and saved!")
    print("="*70 + "\n")

