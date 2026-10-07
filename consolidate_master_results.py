"""
Master Benchmark & Publication Report Generator
===============================================
Consolidates evaluation results across all controller classes and scenarios:
  1. Classical Baselines: Constant Speed, PI, Starling, MPC (Phase 3 + updated Scenario 9)
  2. Centralized SAC (Phase 5, 5 seeds)
  3. Independent SAC (Phase 6, 5 seeds)
  4. Bilateral-Branch SAC (Phase 7, 5 seeds, auto-detected)

Outputs:
  - master_table1_rest.md & .csv (Koh Table 3 style)
  - master_table2_all_scenarios.md & .csv (Shi Table 1 style)
  - master_figures/ publication plots
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, 'Results')
MASTER_OUT = os.path.join(WORKSPACE_ROOT, 'master_results')
FIG_OUT = os.path.join(MASTER_OUT, 'figures')
os.makedirs(MASTER_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
SCENARIOS_ORDER = [
    'Rest', 'SVR_up', 'SVR_down', 'PVR_up', 'Exercise',
    'Exercise_Recovery', 'Hypovolemia', 'Combined', 'Suction_Test'
]

CTRL_DISPLAY_NAMES = {
    'Constant_Speed': 'Constant Speed',
    'PI': 'Dual-PI',
    'Starling': 'Starling-like',
    'MPC': 'CMO-MPC (Koh 2020)',
    'Independent_SAC': 'Independent SAC',
    'Centralized_SAC': 'Centralized SAC (Flat)',
    'Bilateral_SAC': 'Bilateral SAC (Ours)'
}

CTRL_COLORS = {
    'Constant Speed': '#7f7f7f',
    'Dual-PI': '#d62728',
    'Starling-like': '#ff7f0e',
    'CMO-MPC (Koh 2020)': '#1f77b4',
    'Independent SAC': '#9467bd',
    'Centralized SAC (Flat)': '#2ca02c',
    'Bilateral SAC (Ours)': '#17becf'
}


def load_baselines():
    """Loads Phase 3 baseline data, updating Scenario 9 with the new physics-correct evaluation for all 4 controllers."""
    phase3_path = os.path.join(WORKSPACE_ROOT, 'phase3_results', 'unified_evaluation_results.csv')
    df = pd.read_csv(phase3_path)
    
    # Exclude old Scenario 9 for all controllers
    df_base = df[df['Scenario'] != 'Suction_Test'].copy()
    
    # Load new Scenario 9 for CS, PI, Starling
    s9_scratch = os.path.join(WORKSPACE_ROOT, 'scratch_s9_test', 'unified_evaluation_results.csv')
    if os.path.exists(s9_scratch):
        df_s9 = pd.read_csv(s9_scratch)
        df_base = pd.concat([df_base, df_s9], ignore_index=True)
        print(f"Loaded {len(df_s9)} updated Scenario 9 rows for CS, PI, Starling.")
        
    # Load new Scenario 9 for MPC from mpc_output/
    mpc_csv1 = os.path.join(WORKSPACE_ROOT, 'mpc_output', 'mpc_suction_results.csv')
    mpc_csv2 = os.path.join(WORKSPACE_ROOT, 'mpc_output', 'mpc_suction_results_1024.csv')
    mpc_dfs = []
    if os.path.exists(mpc_csv1):
        mpc_dfs.append(pd.read_csv(mpc_csv1))
    if os.path.exists(mpc_csv2):
        mpc_dfs.append(pd.read_csv(mpc_csv2))
    if mpc_dfs:
        df_mpc = pd.concat(mpc_dfs, ignore_index=True)
        df_base = pd.concat([df_base, df_mpc], ignore_index=True)
        print(f"Loaded {len(df_mpc)} updated Scenario 9 rows for MPC across seeds: {df_mpc['Seed'].tolist()}.")
        
    return df_base


def load_drl_family(prefix, display_name):
    """Loads all 5 seeds for a DRL family (Centralized, Independent, or Bilateral)."""
    dfs = []
    for s in SEEDS:
        table_path = os.path.join(RESULTS_DIR, f"{prefix}_seed_{s}", 'tables', 'performance_table.csv')
        if os.path.exists(table_path):
            df = pd.read_csv(table_path)
            df['Controller'] = display_name
            df['Seed'] = s
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined)} rows for {display_name} across {len(dfs)} seeds.")
        return combined
    return pd.DataFrame()


def build_master_dataset():
    """Gathers all baseline and DRL results into a single clean DataFrame."""
    print("=== Consolidating Master Benchmark Dataset ===")
    df_base = load_baselines()
    df_base['Controller'] = df_base['Controller'].map(lambda c: CTRL_DISPLAY_NAMES.get(c, c))
    
    df_cent = load_drl_family('centralized_sac', 'Centralized SAC (Flat)')
    df_indep = load_drl_family('independent_sac', 'Independent SAC')
    df_bilat = load_drl_family('bilateral_sac', 'Bilateral SAC (Ours)')
    
    parts = [df_base, df_cent, df_indep]
    if not df_bilat.empty:
        parts.append(df_bilat)
    else:
        print("[NOTE] Bilateral SAC not yet found in Results/ — will be included once Kaggle run is downloaded.")
        
    master_df = pd.concat(parts, ignore_index=True)
    master_csv = os.path.join(MASTER_OUT, 'master_unified_results.csv')
    master_df.to_csv(master_csv, index=False)
    print(f"[SUCCESS] Master dataset assembled: {len(master_df)} rows saved to {master_csv}")
    return master_df


def df_to_markdown(df):
    """Pure-Python markdown table formatter without tabulate dependency."""
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


def generate_publication_tables(df):
    """Generates Table 1 (Rest Hemodynamics) and Table 2 (All Scenarios Summary)."""
    # -------------------------------------------------------------
    # Table 1: Rest Hemodynamics (Koh Table 3 style)
    # -------------------------------------------------------------
    df_rest = df[df['Scenario'] == 'Rest']
    ctrls = [c for c in CTRL_DISPLAY_NAMES.values() if c in df_rest['Controller'].unique()]
    
    t1_rows = []
    # Physiological reference
    t1_rows.append({
        'Controller': 'Target / Physiological Norm',
        'MAP (mmHg)': '90.0 - 100.0',
        'CO (L/min)': '5.00 - 5.50',
        'LAP (mmHg)': '8.0 - 12.0',
        'FBE (mL/s)': '< 20.0',
        'LVAD Speed (RPM)': '2600 - 2800',
        'RVAD Speed (RPM)': '1900 - 2100',
        'IAE_LAP': 'Target Tracking'
    })
    
    for c in ctrls:
        sub = df_rest[df_rest['Controller'] == c]
        t1_rows.append({
            'Controller': c,
            'MAP (mmHg)': f"{sub['MAP_mean'].mean():.1f} ± {sub['MAP_mean'].std():.1f}",
            'CO (L/min)': f"{sub['CO_mean'].mean():.2f} ± {sub['CO_mean'].std():.2f}",
            'LAP (mmHg)': f"{sub['Mean_LAP'].mean():.1f} ± {sub['Mean_LAP'].std():.1f}",
            'FBE (mL/s)': f"{sub['FBE_mean'].mean():.2f} ± {sub['FBE_mean'].std():.2f}",
            'LVAD Speed (RPM)': f"{sub['Mean_LVAD_RPM'].mean():.0f}",
            'RVAD Speed (RPM)': f"{sub['Mean_RVAD_RPM'].mean():.0f}",
            'IAE_LAP': f"{sub['IAE_LAP'].mean():.1f} ± {sub['IAE_LAP'].std():.1f}"
        })
        
    df_t1 = pd.DataFrame(t1_rows)
    df_t1.to_csv(os.path.join(MASTER_OUT, 'master_table1_rest.csv'), index=False)
    with open(os.path.join(MASTER_OUT, 'master_table1_rest.md'), 'w', encoding='utf-8') as f:
        f.write("# Table 1: Rest Hemodynamics Across All Controllers (Mean ± SD, 5 Seeds)\n\n")
        f.write(df_to_markdown(df_t1))
        f.write("\n")
    print("Generated Table 1 (Rest Hemodynamics).")
    
    # -------------------------------------------------------------
    # Table 2: All Scenarios Summary (Shi Table 1 style)
    # -------------------------------------------------------------
    t2_rows = []
    for sc in SCENARIOS_ORDER:
        for c in ctrls:
            sub = df[(df['Scenario'] == sc) & (df['Controller'] == c)]
            if sub.empty:
                continue
            t2_rows.append({
                'Scenario': sc,
                'Controller': c,
                'IAE_LAP': f"{sub['IAE_LAP'].mean():.1f} ± {sub['IAE_LAP'].std():.1f}",
                'FBE_mean': f"{sub['FBE_mean'].mean():.2f} ± {sub['FBE_mean'].std():.2f}",
                'FBE_peak': f"{sub['FBE_peak'].mean():.1f}",
                'MAP_mean': f"{sub['MAP_mean'].mean():.1f} ± {sub['MAP_mean'].std():.1f}",
                'CO_mean': f"{sub['CO_mean'].mean():.2f} ± {sub['CO_mean'].std():.2f}",
                'Suction_Pct': f"{sub['Suction_Pct'].mean():.1f}%",
                'Mean_Reward': f"{sub['Mean_Reward'].mean():.2f} ± {sub['Mean_Reward'].std():.2f}"
            })
            
    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(os.path.join(MASTER_OUT, 'master_table2_all_scenarios.csv'), index=False)
    with open(os.path.join(MASTER_OUT, 'master_table2_all_scenarios.md'), 'w', encoding='utf-8') as f:
        f.write("# Table 2: Multi-Scenario Benchmark Performance (Mean ± SD Across 5 Seeds)\n\n")
        f.write(df_to_markdown(df_t2))
        f.write("\n")
    print("Generated Table 2 (Multi-Scenario Benchmark).")


def generate_publication_charts(df):
    """Generates clean publication bar charts for IAE, FBE, and the Ablation ladder."""
    ctrls = [c for c in CTRL_DISPLAY_NAMES.values() if c in df['Controller'].unique()]
    scenarios = [s for s in SCENARIOS_ORDER if s in df['Scenario'].unique()]
    
    # -------------------------------------------------------------
    # 1. Flow Balance Error Comparison (Bar Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    n_ctrls = len(ctrls)
    total_w = 0.82
    w = total_w / n_ctrls
    
    for i, c in enumerate(ctrls):
        means, stds = [], []
        for sc in scenarios:
            sub = df[(df['Scenario'] == sc) & (df['Controller'] == c)]
            means.append(sub['FBE_mean'].mean() if not sub.empty else 0.0)
            stds.append(sub['FBE_mean'].std() if not sub.empty else 0.0)
        pos = x - total_w / 2 + (i + 0.5) * w
        ax.bar(pos, means, yerr=stds, width=w * 0.9, capsize=3, label=c,
               color=CTRL_COLORS.get(c, '#333333'), alpha=0.9)
        
    ax.axhline(20.0, color='red', ls='--', lw=1.5, label='Clinical Safety Limit (20 mL/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Flow Balance Error (mL/s)", fontsize=11, fontweight='bold')
    ax.set_title("Dual-Pump Flow Balance Across All 9 Clinical Scenarios", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT, 'master_fbe_comparison.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Tracking Error (IAE_LAP) Comparison (Bar Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, c in enumerate(ctrls):
        means, stds = [], []
        for sc in scenarios:
            sub = df[(df['Scenario'] == sc) & (df['Controller'] == c)]
            means.append(sub['IAE_LAP'].mean() if not sub.empty else 0.0)
            stds.append(sub['IAE_LAP'].std() if not sub.empty else 0.0)
        pos = x - total_w / 2 + (i + 0.5) * w
        ax.bar(pos, means, yerr=stds, width=w * 0.9, capsize=3, label=c,
               color=CTRL_COLORS.get(c, '#333333'), alpha=0.9)
        
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("IAE LAP (mmHg·steps, Lower is Better)", fontsize=11, fontweight='bold')
    ax.set_title("Target Flow Tracking Error Across All 9 Clinical Scenarios", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT, 'master_iae_comparison.png'), dpi=300)
    plt.close()
    print("Generated publication figures in master_results/figures/.")


if __name__ == '__main__':
    df_all = build_master_dataset()
    generate_publication_tables(df_all)
    generate_publication_charts(df_all)
    print("\n>>> [COMPLETE] Master consolidation ready!")
