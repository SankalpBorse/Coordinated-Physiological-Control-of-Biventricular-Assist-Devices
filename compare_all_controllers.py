"""
Phase 6 — Comprehensive Multi-Controller Comparative Benchmark & Visualization
==============================================================================
Aggregates and benchmarks all controllers across all 8 standardized scenarios:
  1. Constant Speed (Baseline)
  2. Starling-Like Controller (Clinical Heuristic)
  3. PI Decentralized Controller (Industrial Classical Control)
  4. Centralized Model Predictive Control (MPC, Hp=5)
  5. Centralized SAC (MIMO DRL)
  6. Independent Dual-Agent SAC (Decentralized DRL Ablation, if available)

Outputs:
  - comparative_analysis/overall_controller_benchmark.csv
  - comparative_analysis/scenario_breakdown_matrix.csv
  - High-resolution comparative figures in comparative_analysis/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0


def main():
    out_dir = os.path.join(WORKSPACE_ROOT, "comparative_analysis")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    dfs = []
    # 1. Baseline controllers
    base_csv = os.path.join(WORKSPACE_ROOT, "evaluation_results.csv")
    if os.path.exists(base_csv):
        print(f"[DATA] Loading baselines from {base_csv}")
        dfs.append(pd.read_csv(base_csv))

    # 2. Centralized SAC
    sac_csv = os.path.join(WORKSPACE_ROOT, "results_centralized", "centralized_trajectories.csv")
    if os.path.exists(sac_csv):
        print(f"[DATA] Loading Centralized SAC from {sac_csv}")
        dfs.append(pd.read_csv(sac_csv))

    # 3. Independent SAC (if available)
    indep_csv = os.path.join(WORKSPACE_ROOT, "results_independent", "independent_trajectories.csv")
    if os.path.exists(indep_csv):
        print(f"[DATA] Loading Independent SAC from {indep_csv}")
        dfs.append(pd.read_csv(indep_csv))

    if not dfs:
        print("[ERROR] No evaluation datasets found!")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    controllers = df_all['Controller'].unique().tolist()
    print(f"[DATA] Loaded {len(df_all):,} rows across controllers: {controllers}")

    # Aggregations
    overall_summary = df_all.groupby('Controller').agg({
        'flow_balance_error': ['mean', 'max'],
        'LAP': ['mean', 'min', 'max'],
        'RAP': ['mean', 'min', 'max'],
        'MAP': 'mean',
        'CO_left': 'mean',
        'Reward': ['mean', 'sum']
    }).round(2)

    # Flatten multiindex columns
    overall_summary.columns = ['_'.join(c).strip('_') for c in overall_summary.columns]
    overall_summary = overall_summary.sort_values('flow_balance_error_mean')
    overall_csv = os.path.join(out_dir, "overall_controller_benchmark.csv")
    overall_summary.to_csv(overall_csv)
    print(f"\n[EXPORT] Saved overall benchmark to: {overall_csv}")
    print("\n" + "=" * 95)
    print(overall_summary[['flow_balance_error_mean', 'LAP_mean', 'RAP_mean', 'MAP_mean', 'CO_left_mean', 'Reward_mean']])
    print("=" * 95)

    # -------------------------------------------------------------
    # Figure 1: Flow Balance Error Comparison Bar Plot
    # -------------------------------------------------------------
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    order = overall_summary.index.tolist()
    fbe_means = overall_summary['flow_balance_error_mean'].values
    colors = ['#1f77b4' if 'SAC' in c else '#aec7e8' if 'MPC' in c else '#c7c7c7' for c in order]

    bars = ax.bar(order, fbe_means, color=colors, edgecolor='black', width=0.55, alpha=0.9)
    ax.axhline(20.0, color='red', ls='--', lw=1.8, label='Clinical Safety Limit (20 mL/s)')
    ax.set_ylabel("Mean Flow Balance Error (mL/s)", fontsize=12, fontweight='bold')
    ax.set_title("Inter-Ventricular Flow Balance Error Across Controllers (8 Benchmark Scenarios)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticklabels(order, rotation=20, ha='right', fontsize=11, fontweight='medium')
    ax.legend(fontsize=10, loc='upper left')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f} mL/s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fbe_fig_path = os.path.join(fig_dir, "overall_flow_balance_error_comparison.png")
    plt.savefig(fbe_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Saved {fbe_fig_path}")

    # -------------------------------------------------------------
    # Figure 2: LAP and RAP Atrial Pressure Safety Ranges
    # -------------------------------------------------------------
    fig, (ax_lap, ax_rap) = plt.subplots(1, 2, figsize=(14, 6))
    lap_data = [df_all[df_all['Controller'] == c]['LAP'].values for c in order]
    rap_data = [df_all[df_all['Controller'] == c]['RAP'].values for c in order]

    bp1 = ax_lap.boxplot(lap_data, tick_labels=order, patch_artist=True)
    for box in bp1['boxes']:
        box.set_facecolor('#a6cee3')
    ax_lap.axhspan(9.0, 12.0, color='#33a02c', alpha=0.2, label='Nominal Target (9-12 mmHg)')
    ax_lap.axhline(3.0, color='red', ls=':', lw=2.0, label='Suction Limit (3 mmHg)')
    ax_lap.set_title("LAP Regulation Across Controllers", fontsize=12, fontweight='bold')
    ax_lap.set_ylabel("LAP (mmHg)", fontsize=11)
    ax_lap.set_xticklabels(order, rotation=25, ha='right', fontsize=10)
    ax_lap.legend(loc='upper right', fontsize=9)
    ax_lap.set_ylim(0, 25)

    bp2 = ax_rap.boxplot(rap_data, tick_labels=order, patch_artist=True)
    for box in bp2['boxes']:
        box.set_facecolor('#fdbf6f')
    ax_rap.axhspan(4.0, 6.0, color='#33a02c', alpha=0.2, label='Nominal Target (4-6 mmHg)')
    ax_rap.axhline(1.0, color='red', ls=':', lw=2.0, label='Suction Limit (1 mmHg)')
    ax_rap.set_title("RAP Regulation Across Controllers", fontsize=12, fontweight='bold')
    ax_rap.set_ylabel("RAP (mmHg)", fontsize=11)
    ax_rap.set_xticklabels(order, rotation=25, ha='right', fontsize=10)
    ax_rap.legend(loc='upper right', fontsize=9)
    ax_rap.set_ylim(0, 20)

    fig.suptitle("Comparative Hemodynamic Safety & Atrial Pressure Regulation", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    press_fig_path = os.path.join(fig_dir, "atrial_pressures_comparative_boxplots.png")
    plt.savefig(press_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Saved {press_fig_path}")

    print("\n" + "=" * 80)
    print("      MULTI-CONTROLLER BENCHMARK COMPARISON COMPLETE!")
    print(f"      Reports and figures saved to: {out_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

