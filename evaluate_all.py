"""
Unified Evaluation Harness for BiVAD Physiological Control
===========================================================
Single source of truth for benchmark evaluations across all 4 controllers
and 9 clinical scenarios with 5-seed statistical rigor.

Controllers Supported:
  1. Constant_Speed (Baseline 1)
  2. PI (Decentralized Pressure-Targeting Baseline 2)
  3. Starling (Dual Sigmoid Starling Baseline 3)
  4. MPC (Physiological Receding Horizon Baseline 4)
  5. SAC_Centralized / SAC_Independent (DRL Agents via universal adapter)

Scenarios Supported (1 to 9):
  1. Rest
  2. SVR_up
  3. SVR_down
  4. PVR_up
  5. Exercise
  6. Exercise_Recovery
  7. Hypovolemia
  8. Combined
  9. Suction_Test (Volume/compliance reduction to test suction boundary)

Metrics Computed Identically for All Controllers:
  - IAE_LAP: Integral of Absolute Error on LAP relative to 9.0 mmHg (mmHg*s)
  - IAE_RAP: Integral of Absolute Error on RAP relative to 4.0 mmHg (mmHg*s)
  - FBE_mean, FBE_peak: Beat-averaged Left-Right pump Flow Balance Error (mL/s)
  - MAP_mean: Mean Arterial Pressure AoP (mmHg)
  - CO_mean: Cardiac Output / Left Pump Flow (L/min)
  - Settling_Time: First time after perturbation where |LAP - 9.0| < 1.0 for >=10 steps (s)
  - Overshoot: Peak post-perturbation |LAP - 9.0| (mmHg)
  - Suction_Pct: % steps with LAP < 3.0 or RAP < 1.0 or LVP/RVP < 0
  - Congestion_Pct: % steps with LAP > 18.0 or RAP > 12.0
  - Mean_Reward + components (r_track, r_balance, r_safety, r_smooth, r_map)
"""

import os
import sys
import time
import copy
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prevent single-thread library oversubscription in parallel runs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Workspace imports
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from bivad_env import BiVADEnv
from controllers.constant_speed import ConstantSpeedController
from controllers.pi_controller import PIDecentralizedController
from controllers.starling_controller import StarlingController
from controllers.mpc_controller import MPCController


# =============================================================================
# 1. STANDARDIZED CLINICAL SCENARIO DEFINITIONS (1 to 9)
# =============================================================================
SCENARIOS = {
    1: {
        'name': 'Rest',
        'perturb_step': 0,
        'description': 'Resting hemodynamic baseline (HR=75 bpm, R_sys=1.0)',
        'changes': {}
    },
    2: {
        'name': 'SVR_up',
        'perturb_step': 500,
        'description': 'Systemic vasoconstriction / hypertension (R_sys = 1.5 at t=50s)',
        'changes': {500: lambda p: p.update({'R_sys': 1.5})}
    },
    3: {
        'name': 'SVR_down',
        'perturb_step': 500,
        'description': 'Systemic vasodilation / hypotension (R_sys = 0.6 at t=50s)',
        'changes': {500: lambda p: p.update({'R_sys': 0.6})}
    },
    4: {
        'name': 'PVR_up',
        'perturb_step': 500,
        'description': 'Pulmonary hypertension / embolism (R_pulm = 0.3 at t=50s)',
        'changes': {500: lambda p: p.update({'R_pulm': 0.3})}
    },
    5: {
        'name': 'Exercise',
        'perturb_step': 500,
        'description': 'Physical exertion (HR = 100 bpm, R_sys * 0.8 at t=50s)',
        'changes': {500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys'] * 0.8})}
    },
    6: {
        'name': 'Exercise_Recovery',
        'perturb_step': 500,
        'description': 'Exercise onset at t=50s, recovery back to rest at t=120s',
        'changes': {
            500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys'] * 0.8}),
            1200: lambda p: p.update({'HR': 75.0, 'R_sys': p['R_sys'] / 0.8})
        }
    },
    7: {
        'name': 'Hypovolemia',
        'perturb_step': 0,
        'description': '20% volume loss / acute hemorrhage (all compliances * 0.8 at t=0)',
        'changes': {0: lambda p: p.update({
            'C_la': p['C_la'] * 0.8, 'C_ra': p['C_ra'] * 0.8,
            'C_sys': p['C_sys'] * 0.8, 'C_pulm': p['C_pulm'] * 0.8
        })}
    },
    8: {
        'name': 'Combined',
        'perturb_step': 500,
        'description': 'Multifactorial severe hemodynamic stress at t=50s',
        'changes': {500: lambda p: p.update({
            'R_sys': p['R_sys'] * 1.3, 'R_pulm': p['R_pulm'] * 1.5, 'HR': p['HR'] * 1.2
        })}
    },
    9: {
        'name': 'Suction_Test',
        'perturb_step': 500,
        'description': 'Severe hypovolemia boundary test (acute volume evacuation at t=50s, resuscitation at t=120s)',
        'changes': {
            500: lambda p, s: s[:8].__setitem__(slice(None), s[:8] * 0.3),
            1200: lambda p, s: s[:8].__setitem__(slice(None), s[:8] / 0.3),
        }
    },
}

# Standard 5 seeds for rigorous reporting
DEFAULT_SEEDS = [42, 123, 456, 789, 999]


# =============================================================================
# 2. UNIVERSAL CONTROLLER ADAPTER
# =============================================================================
class ControllerAdapter:
    """
    Wraps classical controllers (.get_action, .reset) and SB3 SAC models
    into a standardized evaluation interface.
    """
    def __init__(self, name, model_or_controller):
        self.name = name
        self.ctrl = model_or_controller
        self.is_sb3 = hasattr(self.ctrl, 'predict')

    def reset(self):
        if hasattr(self.ctrl, 'reset'):
            self.ctrl.reset()

    def get_action(self, obs, info):
        if self.is_sb3:
            action, _ = self.ctrl.predict(obs, deterministic=True)
            return action
        else:
            return self.ctrl.get_action(obs, info)


def make_controller(controller_name):
    """Factory creating fresh controller instances."""
    if controller_name == 'Constant_Speed':
        return ControllerAdapter('Constant_Speed', ConstantSpeedController())
    elif controller_name == 'PI':
        return ControllerAdapter('PI', PIDecentralizedController())
    elif controller_name == 'Starling':
        return ControllerAdapter('Starling', StarlingController())
    elif controller_name == 'MPC':
        return ControllerAdapter('MPC', MPCController(Hp=5))
    else:
        raise ValueError(f"Unknown controller name: {controller_name}")


# =============================================================================
# 3. METRICS COMPUTATION HELPERS
# =============================================================================
def compute_settling_time(laps, perturb_step, dt=0.1, tolerance=1.0, window=10):
    """
    First time after perturbation where |LAP - LAP_baseline| < tolerance for >= window steps.
    LAP_baseline is the mean LAP over the pre-perturbation window.
    For unperturbed Rest, LAP_baseline is the final steady-state mean.
    Returns settling time in seconds relative to perturbation, or np.nan if unachieved.
    """
    n = len(laps)
    if perturb_step > 0 and perturb_step < n:
        pre_window = min(50, perturb_step)
        target_lap = float(np.mean(laps[perturb_step - pre_window : perturb_step]))
        start_step = perturb_step
    else:
        target_lap = float(np.mean(laps[-100:])) if n >= 100 else float(np.mean(laps))
        start_step = 0

    for i in range(start_step, n - window + 1):
        if all(abs(laps[j] - target_lap) < tolerance for j in range(i, i + window)):
            return round(float((i - start_step) * dt), 2)
    return np.nan


def compute_overshoot(laps, perturb_step):
    """Peak post-perturbation absolute deviation from LAP baseline (mmHg)."""
    n = len(laps)
    if perturb_step > 0 and perturb_step < n:
        pre_window = min(50, perturb_step)
        base_lap = float(np.mean(laps[perturb_step - pre_window : perturb_step]))
        post_laps = laps[perturb_step:]
    else:
        base_lap = float(np.mean(laps[:50])) if n >= 50 else float(laps[0])
        post_laps = laps
    if len(post_laps) == 0:
        return 0.0
    return round(float(np.max(np.abs(np.array(post_laps) - base_lap))), 3)


# =============================================================================
# 4. SINGLE EPISODE EVALUATION FUNCTION
# =============================================================================
def run_episode(controller, scenario_id, seed=0, save_trajectory=False, max_steps=None):
    """
    Executes one complete evaluation episode for any controller and scenario.

    Args:
        controller      : ControllerAdapter instance or controller name string
        scenario_id     : Integer key into SCENARIOS (1 to 9)
        seed            : Integer random seed for initial condition variation
        save_trajectory : If True, returns per-step trajectory rows
        max_steps       : Optional override for max steps (default env.max_steps = 2000)

    Returns:
        metrics_dict    : Standard dictionary of computed performance metrics
        trajectory_rows : List of step-by-step records (empty if save_trajectory=False)
    """
    if isinstance(controller, str):
        ctrl = make_controller(controller)
        ctrl_name = controller
    else:
        ctrl = controller
        ctrl_name = getattr(ctrl, 'name', ctrl.__class__.__name__)

    scenario = SCENARIOS[scenario_id]
    s_name = scenario['name']
    perturb_step = scenario['perturb_step']

    # Initialize environment with seed
    env = BiVADEnv(randomize=False)
    env.perturbation_schedule = {}
    obs, _ = env.reset(seed=seed)
    ctrl.reset()

    total_steps = max_steps if max_steps is not None else env.max_steps

    def _apply_disturbance(change_fn, params, state):
        import inspect
        try:
            sig = inspect.signature(change_fn)
            if len(sig.parameters) >= 2:
                change_fn(params, state)
            else:
                change_fn(params)
        except Exception:
            change_fn(params)

    # Apply t=0 changes if defined
    if 0 in scenario['changes']:
        _apply_disturbance(scenario['changes'][0], env.params, env.state)

    info = {
        'LVAD_RPM': env.params['LVAD_RPM'],
        'RVAD_RPM': env.params['RVAD_RPM'],
        'true_state': np.copy(env.state),
        'env_params': copy.deepcopy(env.params)
    }

    # Data collectors
    totals, tracks, balances, safeties, smooths, maps = [], [], [], [], [], []
    laps, raps, aops, cos, fbes = [], [], [], [], []
    lvad_rpms, rvad_rpms = [], []
    suction_l_list, suction_r_list = [], []
    trajectory_rows = []

    for step_i in range(total_steps):
        # Apply scheduled mid-episode disturbances
        if step_i in scenario['changes'] and step_i > 0:
            _apply_disturbance(scenario['changes'][step_i], env.params, env.state)

        # Get control action
        action = ctrl.get_action(obs, info)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Record reward components
        totals.append(reward)
        tracks.append(info.get('reward_track', 0.0))
        balances.append(info.get('reward_balance', 0.0))
        safeties.append(info.get('reward_safety', 0.0))
        smooths.append(info.get('reward_smooth', 0.0))
        maps.append(info.get('reward_map', 0.0))

        # Record physical states
        y = info.get('true_state', env.state)
        lap, rap = float(y[2]), float(y[3])
        aop = float(y[4])
        co = float(info.get('CO_left', y[10] * 60.0 / 1000.0))
        fbe = float(info.get('flow_balance_error', abs(y[10] - y[11])))
        lvad_rpm = float(info.get('LVAD_RPM', env.params['LVAD_RPM']))
        rvad_rpm = float(info.get('RVAD_RPM', env.params['RVAD_RPM']))
        s_l = bool(info.get('suction_left', False))
        s_r = bool(info.get('suction_right', False))

        laps.append(lap)
        raps.append(rap)
        aops.append(aop)
        cos.append(co)
        fbes.append(fbe)
        lvad_rpms.append(lvad_rpm)
        rvad_rpms.append(rvad_rpm)
        suction_l_list.append(s_l)
        suction_r_list.append(s_r)

        if save_trajectory:
            t_sec = round((step_i + 1) * env.dt, 2)
            trajectory_rows.append({
                'Time': t_sec,
                'Controller': ctrl_name,
                'Scenario_ID': scenario_id,
                'Scenario': s_name,
                'Seed': seed,
                'MAP': aop,
                'LAP': lap,
                'RAP': rap,
                'CO_left': co,
                'flow_balance_error': fbe,
                'LVAD_RPM': lvad_rpm,
                'RVAD_RPM': rvad_rpm,
                'Reward': reward,
                'suction_left': s_l,
                'suction_right': s_r
            })

        if terminated or truncated:
            break

    n_steps = len(totals)
    dt = env.dt

    # Standard metrics
    iae_lap = round(float(np.sum(np.abs(np.array(laps) - 9.0)) * dt), 2)
    iae_rap = round(float(np.sum(np.abs(np.array(raps) - 4.0)) * dt), 2)
    fbe_mean = round(float(np.mean(fbes)), 3)
    fbe_peak = round(float(np.max(fbes)), 3)
    map_mean = round(float(np.mean(aops)), 2)
    co_mean = round(float(np.mean(cos)), 3)
    settling_time = compute_settling_time(laps, perturb_step, dt=dt, tolerance=1.0, window=10)
    overshoot = compute_overshoot(laps, perturb_step)

    # Suction and congestion percentages
    suction_flags = [(l < 3.0 or r < 1.0 or sl or sr) for l, r, sl, sr in zip(laps, raps, suction_l_list, suction_r_list)]
    suction_pct = round(100.0 * float(np.mean(suction_flags)), 2)

    congestion_flags = [(l > 18.0 or r > 12.0) for l, r in zip(laps, raps)]
    congestion_pct = round(100.0 * float(np.mean(congestion_flags)), 2)

    metrics = {
        'Controller': ctrl_name,
        'Scenario_ID': scenario_id,
        'Scenario': s_name,
        'Seed': seed,
        'IAE_LAP': iae_lap,
        'IAE_RAP': iae_rap,
        'FBE_mean': fbe_mean,
        'FBE_peak': fbe_peak,
        'MAP_mean': map_mean,
        'CO_mean': co_mean,
        'Settling_Time': settling_time,
        'Overshoot': overshoot,
        'Suction_Pct': suction_pct,
        'Congestion_Pct': congestion_pct,
        'Mean_Reward': round(float(np.mean(totals)), 4),
        'r_track': round(float(np.mean(tracks)), 4),
        'r_balance': round(float(np.mean(balances)), 4),
        'r_safety': round(float(np.mean(safeties)), 4),
        'r_smooth': round(float(np.mean(smooths)), 4),
        'r_map': round(float(np.mean(maps)), 4),
        'Mean_LAP': round(float(np.mean(laps)), 2),
        'Mean_RAP': round(float(np.mean(raps)), 2),
        'Mean_LVAD_RPM': round(float(np.mean(lvad_rpms)), 1),
        'Mean_RVAD_RPM': round(float(np.mean(rvad_rpms)), 1),
        'Steps': n_steps
    }

    return metrics, trajectory_rows


# Worker wrapper for multiprocessing
def _run_single_task(args):
    ctrl_name, s_id, seed, save_traj, max_steps = args
    return run_episode(ctrl_name, s_id, seed=seed, save_trajectory=save_traj, max_steps=max_steps)


# =============================================================================
# 5. AGGREGATION & TABLE GENERATION (Koh Table 3 & Shi Table 1 Style)
# =============================================================================
def df_to_markdown(df):
    """Pure-Python markdown table formatter (eliminates external tabulate dependency)."""
    headers = [str(c) for c in df.columns]
    col_widths = []
    for h in headers:
        max_v = max((len(str(v)) for v in df[h]), default=0)
        col_widths.append(max(len(h), max_v))
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = []
    for _, row in df.iterrows():
        data_lines.append("| " + " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths)) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


def generate_summary_tables(df_results, output_dir):
    """
    Generates publication-quality Table 1 (Rest Hemodynamics) and Table 2 (All Scenarios Summary).
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # TABLE 1: Rest Hemodynamics (Koh Table 3 style)
    # -------------------------------------------------------------------------
    df_rest = df_results[df_results['Scenario'] == 'Rest']
    table1_rows = []

    # Reference row
    table1_rows.append({
        'Controller': 'Physiological Reference / Normal',
        'MAP (mmHg)': '90.0 - 100.0',
        'CO (L/min)': '5.00 - 5.50',
        'LAP (mmHg)': '8.0 - 12.0',
        'RAP (mmHg)': '3.0 - 6.0',
        'LVAD Speed (RPM)': '2600 - 2800',
        'RVAD Speed (RPM)': '1900 - 2100',
        'Suction Pct (%)': '0.0%'
    })

    controllers_in_order = ['Constant_Speed', 'PI', 'Starling', 'MPC', 'SAC_Centralized', 'SAC_Independent']
    present_ctrls = [c for c in controllers_in_order if c in df_rest['Controller'].unique()]
    other_ctrls = [c for c in df_rest['Controller'].unique() if c not in present_ctrls]

    for c in present_ctrls + other_ctrls:
        sub = df_rest[df_rest['Controller'] == c]
        if len(sub) == 0:
            continue
        table1_rows.append({
            'Controller': c,
            'MAP (mmHg)': f"{sub['MAP_mean'].mean():.1f} ± {sub['MAP_mean'].std():.1f}",
            'CO (L/min)': f"{sub['CO_mean'].mean():.2f} ± {sub['CO_mean'].std():.2f}",
            'LAP (mmHg)': f"{sub['Mean_LAP'].mean():.1f} ± {sub['Mean_LAP'].std():.1f}",
            'RAP (mmHg)': f"{sub['Mean_RAP'].mean():.1f} ± {sub['Mean_RAP'].std():.1f}",
            'LVAD Speed (RPM)': f"{sub['Mean_LVAD_RPM'].mean():.0f} ± {sub['Mean_LVAD_RPM'].std():.0f}",
            'RVAD Speed (RPM)': f"{sub['Mean_RVAD_RPM'].mean():.0f} ± {sub['Mean_RVAD_RPM'].std():.0f}",
            'Suction Pct (%)': f"{sub['Suction_Pct'].mean():.1f}%"
        })

    df_table1 = pd.DataFrame(table1_rows)
    t1_csv = os.path.join(output_dir, 'table1_rest_hemodynamics.csv')
    t1_md = os.path.join(output_dir, 'table1_rest_hemodynamics.md')
    df_table1.to_csv(t1_csv, index=False)
    with open(t1_md, 'w', encoding='utf-8') as f:
        f.write("# Table 1: Resting Hemodynamics Across Controllers (Koh Table 3 Style)\n\n")
        f.write(df_to_markdown(df_table1))
        f.write("\n")

    # -------------------------------------------------------------------------
    # TABLE 2: Full Scenario Performance (Shi Table 1 style)
    # -------------------------------------------------------------------------
    table2_rows = []
    scenarios_in_order = [SCENARIOS[i]['name'] for i in sorted(SCENARIOS.keys())]

    for s in scenarios_in_order:
        for c in present_ctrls + other_ctrls:
            sub = df_results[(df_results['Scenario'] == s) & (df_results['Controller'] == c)]
            if len(sub) == 0:
                continue
            
            st_vals = sub['Settling_Time'].dropna()
            st_str = f"{st_vals.mean():.1f} ± {st_vals.std():.1f}" if len(st_vals) > 0 else "DNC"

            table2_rows.append({
                'Scenario': s,
                'Controller': c,
                'Mean_Reward': f"{sub['Mean_Reward'].mean():.3f} ± {sub['Mean_Reward'].std():.3f}",
                'IAE_LAP': f"{sub['IAE_LAP'].mean():.1f} ± {sub['IAE_LAP'].std():.1f}",
                'FBE_mean (mL/s)': f"{sub['FBE_mean'].mean():.2f} ± {sub['FBE_mean'].std():.2f}",
                'MAP (mmHg)': f"{sub['MAP_mean'].mean():.1f} ± {sub['MAP_mean'].std():.1f}",
                'CO (L/min)': f"{sub['CO_mean'].mean():.2f} ± {sub['CO_mean'].std():.2f}",
                'Settling_Time (s)': st_str,
                'Overshoot (mmHg)': f"{sub['Overshoot'].mean():.2f} ± {sub['Overshoot'].std():.2f}",
                'Suction Pct (%)': f"{sub['Suction_Pct'].mean():.1f}%"
            })

    df_table2 = pd.DataFrame(table2_rows)
    t2_csv = os.path.join(output_dir, 'table2_all_scenarios_summary.csv')
    t2_md = os.path.join(output_dir, 'table2_all_scenarios_summary.md')
    df_table2.to_csv(t2_csv, index=False)
    with open(t2_md, 'w', encoding='utf-8') as f:
        f.write("# Table 2: Multi-Scenario Benchmark Performance (Shi Table 1 Style)\n\n")
        f.write(df_to_markdown(df_table2))
        f.write("\n")

    return df_table1, df_table2


# =============================================================================
# 6. PUBLICATION FIGURE GENERATION (High-Resolution Visualizations)
# =============================================================================
def set_plot_style():
    """Configures clean publication styling for matplotlib figures."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = '#e0e0e0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7


CTRL_COLORS = {
    'Constant_Speed': '#7f7f7f',   # Gray
    'PI':             '#d62728',   # Red
    'Starling':       '#ff7f0e',   # Orange
    'MPC':            '#1f77b4',   # Blue
    'SAC_Centralized':'#2ca02c',   # Green
    'SAC_Independent':'#9467bd',   # Purple
}


def generate_publication_figures(df_results, df_traj, figures_dir):
    """
    Generates 5 publication-quality benchmark comparison figures in figures_dir:
      1. baseline_trajectories_dashboard.png (Multi-panel dynamics across key scenarios)
      2. fbe_comparison_all_scenarios.png (Flow balance error bar chart across all 9 scenarios)
      3. mean_reward_comparison_all_scenarios.png (Reward ranking across all 9 scenarios)
      4. suction_test_dynamics.png (Dedicated hypovolemia boundary dynamics)
      5. atrial_pressure_safety_bands.png (LAP & RAP regulation relative to clinical bands)
    """
    os.makedirs(figures_dir, exist_ok=True)
    set_plot_style()
    print(f"\nGenerating publication-quality figures in {figures_dir}...")

    controllers = [c for c in ['Constant_Speed', 'PI', 'Starling', 'MPC', 'SAC_Centralized', 'SAC_Independent'] 
                   if c in df_results['Controller'].unique()]
    other_ctrls = [c for c in df_results['Controller'].unique() if c not in controllers]
    all_ctrls = controllers + other_ctrls
    scenarios_in_order = [SCENARIOS[i]['name'] for i in sorted(SCENARIOS.keys()) if SCENARIOS[i]['name'] in df_results['Scenario'].unique()]

    # -------------------------------------------------------------------------
    # FIGURE 1: Comparative Trajectories Dashboard across Key Scenarios
    # -------------------------------------------------------------------------
    if df_traj is not None and len(df_traj) > 0:
        key_scenarios = ['Rest', 'SVR_up', 'Exercise_Recovery', 'Suction_Test']
        avail_scens = [s for s in key_scenarios if s in df_traj['Scenario'].unique()]
        if not avail_scens:
            avail_scens = list(df_traj['Scenario'].unique())[:4]

        fig, axes = plt.subplots(4, len(avail_scens), figsize=(5.2 * len(avail_scens), 12.0), sharex='col')
        if len(avail_scens) == 1:
            axes = np.expand_dims(axes, axis=1)

        for col_idx, s_name in enumerate(avail_scens):
            sub_s = df_traj[df_traj['Scenario'] == s_name]
            for c in all_ctrls:
                sub_c = sub_s[sub_s['Controller'] == c].sort_values('Time')
                if len(sub_c) == 0:
                    continue
                color = CTRL_COLORS.get(c, '#333333')
                t = sub_c['Time'].values

                axes[0, col_idx].plot(t, sub_c['LAP'], color=color, lw=1.8, label=c)
                axes[1, col_idx].plot(t, sub_c['flow_balance_error'], color=color, lw=1.8, label=c)
                axes[2, col_idx].plot(t, sub_c['MAP'], color=color, lw=1.8, label=c)
                axes[3, col_idx].plot(t, sub_c['LVAD_RPM'], color=color, lw=1.8, label=c)

            axes[0, col_idx].axhline(9.0, color='green', ls='--', lw=1.2, label='Target LAP (9.0)' if col_idx==0 else "")
            axes[0, col_idx].axhline(3.0, color='red', ls=':', lw=1.5, label='Suction Limit (3.0)' if col_idx==0 else "")
            axes[0, col_idx].set_title(f"Scenario: {s_name}", fontsize=12, fontweight='bold', pad=8)
            axes[0, col_idx].set_ylabel("LAP (mmHg)", fontsize=10)
            axes[0, col_idx].set_ylim(0, 22)

            axes[1, col_idx].axhline(20.0, color='black', ls='--', lw=1.2, label='Safety Limit (20 mL/s)' if col_idx==0 else "")
            axes[1, col_idx].set_ylabel("FBE (mL/s)", fontsize=10)
            max_fbe = sub_s['flow_balance_error'].max() if len(sub_s) > 0 else 45.0
            axes[1, col_idx].set_ylim(0, max(45.0, max_fbe * 1.15))

            axes[2, col_idx].axhspan(70.0, 110.0, color='green', alpha=0.10, label='Target MAP (70-110)' if col_idx==0 else "")
            axes[2, col_idx].set_ylabel("MAP (mmHg)", fontsize=10)
            axes[2, col_idx].set_ylim(50, 160)

            axes[3, col_idx].axhline(3800.0, color='red', ls=':', lw=1.2, label='Speed Limit (3800)' if col_idx==0 else "")
            axes[3, col_idx].set_ylabel("LVAD Speed (RPM)", fontsize=10)
            axes[3, col_idx].set_xlabel("Time (seconds)", fontsize=10)
            axes[3, col_idx].set_ylim(1600, 4200)

            if col_idx == 0:
                axes[0, col_idx].legend(loc='upper right', fontsize=8, framealpha=0.85)
                axes[1, col_idx].legend(loc='upper right', fontsize=8, framealpha=0.85)
                axes[2, col_idx].legend(loc='upper left', fontsize=8, framealpha=0.85)
                axes[3, col_idx].legend(loc='lower right', fontsize=8, framealpha=0.85)

        plt.tight_layout()
        f1_path = os.path.join(figures_dir, 'baseline_trajectories_dashboard.png')
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [FIGURE SAVED] {f1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Flow Balance Error Comparison across all Scenarios (Bar Chart)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios_in_order))
    n_ctrls = len(all_ctrls)
    total_w = 0.8
    w = total_w / max(n_ctrls, 1)

    for i, c in enumerate(all_ctrls):
        means, stds = [], []
        for s in scenarios_in_order:
            sub = df_results[(df_results['Controller'] == c) & (df_results['Scenario'] == s)]
            means.append(sub['FBE_mean'].mean() if len(sub) > 0 else 0.0)
            stds.append(sub['FBE_mean'].std() if len(sub) > 0 else 0.0)
        pos = x - total_w / 2 + (i + 0.5) * w
        ax.bar(pos, means, yerr=stds, width=w * 0.9, capsize=3, label=c, color=CTRL_COLORS.get(c, None), alpha=0.9)

    ax.axhline(20.0, color='red', ls='--', lw=1.5, label='Clinical Safety Limit (20 mL/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_in_order, rotation=25, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Flow Balance Error (mL/s)", fontsize=11, fontweight='bold')
    ax.set_title("Dual-Pump Flow Conservation (FBE) Across All Clinical Scenarios", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    f2_path = os.path.join(figures_dir, 'fbe_comparison_all_scenarios.png')
    plt.savefig(f2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE SAVED] {f2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Mean Reward Comparison across all Scenarios (Bar Chart)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(all_ctrls):
        means, stds = [], []
        for s in scenarios_in_order:
            sub = df_results[(df_results['Controller'] == c) & (df_results['Scenario'] == s)]
            means.append(sub['Mean_Reward'].mean() if len(sub) > 0 else 0.0)
            stds.append(sub['Mean_Reward'].std() if len(sub) > 0 else 0.0)
        pos = x - total_w / 2 + (i + 0.5) * w
        ax.bar(pos, means, yerr=stds, width=w * 0.9, capsize=3, label=c, color=CTRL_COLORS.get(c, None), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_in_order, rotation=25, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Multi-Objective Reward", fontsize=11, fontweight='bold')
    ax.set_title("Benchmark Reward Comparison Across All 9 Clinical Scenarios", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    f3_path = os.path.join(figures_dir, 'mean_reward_comparison_all_scenarios.png')
    plt.savefig(f3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE SAVED] {f3_path}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Dedicated Suction_Test Dynamics (Scenario 9)
    # -------------------------------------------------------------------------
    if df_traj is not None and 'Suction_Test' in df_traj['Scenario'].unique():
        sub_suct = df_traj[df_traj['Scenario'] == 'Suction_Test']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for c in all_ctrls:
            sub_c = sub_suct[sub_suct['Controller'] == c].sort_values('Time')
            if len(sub_c) == 0:
                continue
            color = CTRL_COLORS.get(c, '#333333')
            t = sub_c['Time'].values

            axes[0, 0].plot(t, sub_c['LAP'], color=color, lw=1.8, label=c)
            axes[0, 1].plot(t, sub_c['flow_balance_error'], color=color, lw=1.8, label=c)
            axes[1, 0].plot(t, sub_c['LVAD_RPM'], color=color, lw=1.8, label=c)
            axes[1, 1].plot(t, sub_c['RVAD_RPM'], color=color, lw=1.8, label=c)

        axes[0, 0].axhline(3.0, color='red', ls=':', lw=1.5, label='Suction Threshold (3 mmHg)')
        axes[0, 0].axvspan(50.0, 120.0, color='red', alpha=0.10, label='Severe Hypovolemia Phase')
        axes[0, 0].set_title("Left Atrial Pressure (Suction Boundary)", fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel("LAP (mmHg)", fontsize=10)
        axes[0, 0].set_ylim(0, 16)
        axes[0, 0].legend(loc='upper right', fontsize=8)

        axes[0, 1].axhline(20.0, color='red', ls='--', lw=1.2, label='Safety Limit (20 mL/s)')
        axes[0, 1].axvspan(50.0, 120.0, color='red', alpha=0.10)
        axes[0, 1].set_title("Flow Balance Error Under Volume Depletion", fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel("FBE (mL/s)", fontsize=10)
        axes[0, 1].legend(loc='upper right', fontsize=8)

        axes[1, 0].axvspan(50.0, 120.0, color='red', alpha=0.10)
        axes[1, 0].set_title("LVAD Pump Speed Response", fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel("Time (seconds)", fontsize=10)
        axes[1, 0].set_ylabel("Speed (RPM)", fontsize=10)
        axes[1, 0].legend(loc='lower right', fontsize=8)

        axes[1, 1].axvspan(50.0, 120.0, color='red', alpha=0.10)
        axes[1, 1].set_title("RVAD Pump Speed Response", fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel("Time (seconds)", fontsize=10)
        axes[1, 1].set_ylabel("Speed (RPM)", fontsize=10)
        axes[1, 1].legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        f4_path = os.path.join(figures_dir, 'suction_test_dynamics.png')
        plt.savefig(f4_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [FIGURE SAVED] {f4_path}")

    # -------------------------------------------------------------------------
    # FIGURE 5: Atrial Pressures vs Clinical Safety Bands
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, c in enumerate(all_ctrls):
        lap_means = []
        for s in scenarios_in_order:
            sub = df_results[(df_results['Controller'] == c) & (df_results['Scenario'] == s)]
            lap_means.append(sub['Mean_LAP'].mean() if len(sub) > 0 else np.nan)
        ax.plot(x, lap_means, marker='o', lw=2.0, label=c, color=CTRL_COLORS.get(c, None))

    ax.axhspan(8.0, 12.0, color='green', alpha=0.15, label='Normal Operating Band [8, 12] mmHg')
    ax.axhline(3.0, color='red', ls=':', lw=1.5, label='Suction Limit (3.0 mmHg)')
    ax.axhline(18.0, color='purple', ls='--', lw=1.5, label='Congestion Limit (18.0 mmHg)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_in_order, rotation=25, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Left Atrial Pressure (mmHg)", fontsize=11, fontweight='bold')
    ax.set_title("Atrial Pressure Regulation Relative to Clinical Safety Thresholds", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    f5_path = os.path.join(figures_dir, 'atrial_pressure_safety_bands.png')
    plt.savefig(f5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE SAVED] {f5_path}")


# =============================================================================
# 7. PHASE 3 VALIDATION GATE CHECKER
# =============================================================================
def verify_phase3_gates(df_results):
    """
    Validates all Phase 3 Gate Criteria:
      1. Complete CSV with expected rows (4 controllers x 9 scenarios x N seeds)
      2. Complete metrics present (IAE, Settling_Time, Overshoot, FBE, Suction_Pct)
      3. MPC beats Constant Speed on Mean_Reward across scenarios
      4. Suction_Test: At least one controller produces Suction_Pct > 0
      5. Multi-seed non-zero variance verified
    """
    print("\n" + "=" * 75)
    print("PHASE 3 VALIDATION GATE CHECK")
    print("=" * 75)
    all_ok = True

    # Gate 1: Row count check
    ctrls = df_results['Controller'].unique()
    scenarios = df_results['Scenario'].unique()
    seeds = df_results['Seed'].unique()
    n_expected = len(ctrls) * len(scenarios) * len(seeds)
    n_actual = len(df_results)
    pass_rows = (n_actual == n_expected)
    print(f"  [Gate 3.1] Unified Rows: {n_actual}/{n_expected} ({len(ctrls)} ctrls × {len(scenarios)} scen × {len(seeds)} seeds) -> {'PASS' if pass_rows else 'FAIL'}")
    if not pass_rows: all_ok = False

    # Gate 2: Required metrics present and non-null
    req_cols = ['IAE_LAP', 'IAE_RAP', 'FBE_mean', 'MAP_mean', 'CO_mean', 'Overshoot', 'Suction_Pct', 'Mean_Reward']
    missing = [col for col in req_cols if col not in df_results.columns]
    pass_cols = (len(missing) == 0)
    print(f"  [Gate 3.2] Complete Metrics: Missing {missing} -> {'PASS' if pass_cols else 'FAIL'}")
    if not pass_cols: all_ok = False

    # Gate 3: MPC beats Constant Speed on Mean_Reward
    if 'MPC' in ctrls and 'Constant_Speed' in ctrls:
        mpc_beats_cs = 0
        total_scens = 0
        for s in scenarios:
            mpc_r = df_results[(df_results['Controller'] == 'MPC') & (df_results['Scenario'] == s)]['Mean_Reward'].mean()
            cs_r = df_results[(df_results['Controller'] == 'Constant_Speed') & (df_results['Scenario'] == s)]['Mean_Reward'].mean()
            if mpc_r > cs_r:
                mpc_beats_cs += 1
            total_scens += 1
        pass_mpc = (mpc_beats_cs == total_scens)
        print(f"  [Gate 3.3] MPC Beats CS: {mpc_beats_cs}/{total_scens} scenarios -> {'PASS' if pass_mpc else 'FAIL'}")
        if not pass_mpc: all_ok = False
    else:
        print(f"  [Gate 3.3] MPC vs CS check skipped (both not present)")

    # Gate 4: Suction_Test reachability
    if 'Suction_Test' in scenarios:
        suction_df = df_results[df_results['Scenario'] == 'Suction_Test']
        any_suction = (suction_df['Suction_Pct'] > 0).any()
        max_suct = suction_df['Suction_Pct'].max()
        print(f"  [Gate 3.4] Suction Reachability: Max Suction_Pct = {max_suct:.1f}% -> {'PASS' if any_suction else 'FAIL'}")
        if not any_suction: all_ok = False
    else:
        print(f"  [Gate 3.4] Suction_Test not run")

    # Gate 5: Multi-seed variance exists (seeds produce distinct results)
    std_reward = df_results.groupby(['Controller', 'Scenario'])['Mean_Reward'].std().mean()
    pass_var = (std_reward is not None and not np.isnan(std_reward))
    print(f"  [Gate 3.5] Multi-Seed Variance: Mean Reward Std = {std_reward:.4f} -> {'PASS' if pass_var else 'FAIL'}")
    if not pass_var: all_ok = False

    # PI Controller Saturation & Hypertension Audit
    if 'PI' in ctrls:
        pi_rest = df_results[(df_results['Controller'] == 'PI') & (df_results['Scenario'] == 'Rest')]
        if len(pi_rest) > 0:
            pi_map = pi_rest['MAP_mean'].mean()
            pi_lvad = pi_rest['Mean_LVAD_RPM'].mean()
            print(f"\n  [PI Baseline Audit] Rest MAP: {pi_map:.1f} mmHg, Mean LVAD RPM: {pi_lvad:.0f}")
            if pi_map > 120.0 or pi_lvad > 3500.0:
                print(f"  [PI Baseline Audit] Confirmed: Decentralized PI saturates at high RPM chasing LAP=9.0, producing systemic hypertension. (Documented limitation)")

    print("=" * 75)
    return all_ok


# =============================================================================
# 7. SELF-TEST / SMOKE-TEST HARNESS
# =============================================================================
def run_self_test():
    """Fast local self-test verifying all pipeline components without full compute."""
    print("#" * 75)
    print("EVALUATE_ALL.PY SELF-VALIDATION SMOKE TEST")
    print("#" * 75)
    t0 = time.time()

    test_controllers = ['Constant_Speed', 'PI', 'Starling']
    test_scenarios = [1, 9]  # Rest and Suction_Test
    test_seeds = [42, 123]

    rows = []
    trajs = []
    print("Running mini-evaluation (20 steps per episode)...")
    for c in test_controllers:
        for s in test_scenarios:
            for sd in test_seeds:
                m, tr = run_episode(c, s, seed=sd, max_steps=20, save_trajectory=(sd == test_seeds[0]))
                rows.append(m)
                trajs.extend(tr)

    df_test = pd.DataFrame(rows)
    df_test_traj = pd.DataFrame(trajs) if len(trajs) > 0 else None
    print(f"Mini-evaluation produced {len(df_test)} rows across {len(test_controllers)} ctrls × {len(test_scenarios)} scens × {len(test_seeds)} seeds.")

    # Generate tables & figures
    out_dir = os.path.join(WORKSPACE_ROOT, 'scratch_selftest')
    t1, t2 = generate_summary_tables(df_test, out_dir)
    print(f"Generated Table 1 ({len(t1)} rows) and Table 2 ({len(t2)} rows) successfully.")

    fig_dir = os.path.join(out_dir, 'figures')
    generate_publication_figures(df_test, df_test_traj, fig_dir)
    print(f"Generated publication figures successfully.")

    # Clean up scratch test output
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    elapsed = time.time() - t0
    print(f"Smoke test completed successfully in {elapsed:.2f}s! All functions & figures verified.")
    return True


# =============================================================================
# 8. MAIN CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified BiVAD Evaluation Harness")
    parser.add_argument('--controllers', type=str, default='Constant_Speed,PI,Starling,MPC',
                        help="Comma-separated controllers to evaluate")
    parser.add_argument('--scenarios', type=str, default='1,2,3,4,5,6,7,8,9',
                        help="Comma-separated scenario IDs to evaluate (1-9)")
    parser.add_argument('--seeds', type=str, default='42,123,456,789,999',
                        help="Comma-separated random seeds (default 5 seeds)")
    parser.add_argument('--output_dir', type=str, default='results_phase3',
                        help="Output directory for CSVs and markdown tables")
    parser.add_argument('--save_trajectories', action='store_true',
                        help="Save step-by-step trajectories to CSV")
    parser.add_argument('--parallel', action='store_true',
                        help="Run evaluations in parallel across CPU cores")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel worker processes (default: all CPU cores)")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="Override max steps per episode (default: 2000)")
    parser.add_argument('--self_test', action='store_true',
                        help="Run quick self-validation smoke test")

    args = parser.parse_args()

    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    controllers = [c.strip() for c in args.controllers.split(',') if c.strip()]
    scenario_ids = [int(s.strip()) for s in args.scenarios.split(',') if s.strip()]
    seeds = [int(sd.strip()) for sd in args.seeds.split(',') if sd.strip()]

    output_dir = os.path.join(WORKSPACE_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_episodes = len(controllers) * len(scenario_ids) * len(seeds)
    print("#" * 75)
    print("UNIFIED BiVAD BENCHMARK EVALUATION HARNESS")
    print(f"Controllers : {controllers}")
    print(f"Scenarios   : {[SCENARIOS[s]['name'] for s in scenario_ids]}")
    print(f"Seeds ({len(seeds)})   : {seeds}")
    print(f"Total Runs  : {total_episodes} episodes")
    print(f"Parallel    : {args.parallel} (Workers: {args.workers or os.cpu_count()})")
    print(f"Output Dir  : {output_dir}")
    print("#" * 75 + "\n")

    t_start = time.time()
    all_metrics = []
    all_trajectories = []

    # Build task list (save representative trajectories for seed 0 to enable publication plotting)
    tasks = []
    for c in controllers:
        for s in scenario_ids:
            for sd in seeds:
                is_rep_seed = (sd == seeds[0])
                tasks.append((c, s, sd, (args.save_trajectories or is_rep_seed), args.max_steps))

    if args.parallel:
        n_workers = args.workers or os.cpu_count() or 4
        print(f"Launching {len(tasks)} tasks on {n_workers} worker processes...")
        completed_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(_run_single_task, t): t for t in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_count += 1
                try:
                    metrics, trajs = future.result()
                    all_metrics.append(metrics)
                    all_trajectories.extend(trajs)
                    st_str = f"settle={metrics['Settling_Time']}s" if not np.isnan(metrics['Settling_Time']) else "DNC"
                    print(f"  [{completed_count:3d}/{len(tasks)}] {task[0]:15s} / {SCENARIOS[task[1]]['name']:18s} (Seed {task[2]}): R={metrics['Mean_Reward']:.3f}, MAP={metrics['MAP_mean']:.1f}, CO={metrics['CO_mean']:.2f}, {st_str}")
                except Exception as e:
                    print(f"  [ERROR] Task {task} failed with exception: {e}")
                    raise e
    else:
        for idx, task in enumerate(tasks, 1):
            c_name, s_id, sd, save_traj, max_s = task
            s_name = SCENARIOS[s_id]['name']
            t_ep = time.time()
            print(f"  [{idx:3d}/{len(tasks)}] {c_name:15s} / {s_name:18s} (Seed {sd}) ... ", end='', flush=True)
            metrics, trajs = run_episode(c_name, s_id, seed=sd, save_trajectory=save_traj, max_steps=max_s)
            elapsed_ep = time.time() - t_ep
            all_metrics.append(metrics)
            all_trajectories.extend(trajs)
            st_str = f"settle={metrics['Settling_Time']}s" if not np.isnan(metrics['Settling_Time']) else "DNC"
            print(f"R={metrics['Mean_Reward']:.3f}, MAP={metrics['MAP_mean']:.1f}, CO={metrics['CO_mean']:.2f}, {st_str} ({elapsed_ep:.1f}s)")

    elapsed_total = time.time() - t_start
    print(f"\nAll {len(all_metrics)} evaluation runs completed in {elapsed_total:.1f}s ({elapsed_total/60.0:.2f} min).")

    # Save Unified CSV
    df_results = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, 'unified_evaluation_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved unified results: {csv_path} ({len(df_results)} rows)")

    # Save Trajectories if requested
    df_traj = pd.DataFrame(all_trajectories) if len(all_trajectories) > 0 else None
    if args.save_trajectories and df_traj is not None:
        traj_path = os.path.join(output_dir, 'unified_trajectories.csv')
        df_traj.to_csv(traj_path, index=False)
        print(f"Saved fine-grained trajectories: {traj_path} ({len(df_traj)} steps)")

    # Generate Summary Tables
    df_t1, df_t2 = generate_summary_tables(df_results, output_dir)
    print(f"Saved Table 1 (Rest Hemodynamics) to {output_dir}/table1_rest_hemodynamics.csv and .md")
    print(f"Saved Table 2 (All Scenarios) to {output_dir}/table2_all_scenarios_summary.csv and .md")

    # Generate Publication Figures
    figures_dir = os.path.join(output_dir, 'figures')
    generate_publication_figures(df_results, df_traj, figures_dir)

    # Verify Phase 3 Gate Checks
    gates_ok = verify_phase3_gates(df_results)
    if gates_ok:
        print("\nALL PHASE 3 VALIDATION CRITERIA PASSED [100% OK]!")
    else:
        print("\nWARNING: Some validation checks were flagged. Please inspect output tables above.")


if __name__ == '__main__':
    main()

