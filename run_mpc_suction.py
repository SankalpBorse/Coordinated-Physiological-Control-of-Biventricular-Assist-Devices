"""
Parallel MPC Scenario 9 (Suction Test) Evaluation Runner
========================================================
Runs CMO-MPC (Koh 2020) on the acute volume-evacuation Suction Test scenario
across multiple random seeds using dedicated parallel CPU workers.

Features:
  - Parallel multiprocessing across 4 vCPUs on Kaggle or local workstations
  - Periodic live step logging (every 200 steps) so you can monitor progress
  - Automatic results saving to CSV and markdown tables
  - Outputs ready for seamless ingestion by consolidate_master_results.py
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from bivad_env import BiVADEnv
from controllers.mpc_controller import MPCController
from evaluate_all import SCENARIOS, compute_settling_time, compute_overshoot


def run_single_mpc_suction(seed, max_steps=2000, log_interval=200):
    """
    Executes one complete 2000-step Suction Test episode with MPC,
    printing live progress updates.
    """
    print(f"\n>>> [START Worker Seed {seed}] Starting MPC Suction Test ({max_steps} steps)...")
    t0 = time.time()

    scenario = SCENARIOS[9]  # Suction_Test
    changes = scenario['changes']
    perturb_step = scenario['perturb_step']

    # Initialize environment and controller
    env = BiVADEnv(randomize=False, raw_obs=False)
    obs, info = env.reset(seed=seed)
    ctrl = MPCController(Hp=5)

    laps = []
    raps = []
    cos = []
    fbes = []
    lvad_rpms = []
    rvad_rpms = []
    map_pressures = []
    totals = []
    tracks = []
    balances = []
    safeties = []
    smooths = []
    maps = []

    for step in range(max_steps):
        # Apply scheduled mid-episode physiological perturbations
        if step in changes:
            handler = changes[step]
            import inspect
            sig = inspect.signature(handler)
            if len(sig.parameters) == 2:
                handler(env.params, env.state)
            else:
                handler(env.params)
            print(f"  [Worker Seed {seed}] Step {step}: Perturbation trigger active!")

        # Compute MPC action
        action = ctrl.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        # Log states and rewards
        lap = env.state[2]
        rap = env.state[3]
        fbe = info['flow_balance_error']
        co = info['CO_left']
        lvad = info['LVAD_RPM']
        rvad = info['RVAD_RPM']
        map_p = info['MAP']

        laps.append(lap)
        raps.append(rap)
        fbes.append(fbe)
        cos.append(co)
        lvad_rpms.append(lvad)
        rvad_rpms.append(rvad)
        map_pressures.append(map_p)

        totals.append(reward)
        tracks.append(info.get('reward_track', 0.0))
        balances.append(info.get('reward_balance', 0.0))
        safeties.append(info.get('reward_safety', 0.0))
        smooths.append(info.get('reward_smooth', 0.0))
        maps.append(info.get('reward_map', 0.0))

        # Periodic live progress print
        if (step + 1) % log_interval == 0 or (step + 1) == max_steps:
            elapsed = time.time() - t0
            pct = (step + 1) / max_steps * 100.0
            pace = elapsed / (step + 1)
            eta = pace * (max_steps - (step + 1))
            print(f"  [Worker Seed {seed}] Step {step+1:4d}/{max_steps} ({pct:5.1f}%) | "
                  f"LAP={lap:4.1f} mmHg | CO={co:4.2f} L/min | FBE={fbe:4.1f} mL/s | "
                  f"Elapsed={elapsed/60:.1f}m (ETA: {eta/60:.1f}m)")

        if terminated or truncated:
            break

    elapsed_total = time.time() - t0
    n_steps = len(laps)

    # Compute metrics
    suction_steps = sum(1 for p in laps if p < 3.0)
    congestion_steps = sum(1 for p in laps if p > 18.0)
    suction_pct = round(100.0 * suction_steps / max(n_steps, 1), 2)
    congestion_pct = round(100.0 * congestion_steps / max(n_steps, 1), 2)

    iae_lap = round(float(np.sum(np.abs(np.array(laps) - 9.0)) * 0.1), 2)
    iae_rap = round(float(np.sum(np.abs(np.array(raps) - 4.0)) * 0.1), 2)

    settling_time = compute_settling_time(laps, perturb_step, dt=0.1)
    overshoot = compute_overshoot(laps, perturb_step)

    metrics = {
        'Controller': 'MPC',
        'Scenario_ID': 9,
        'Scenario': 'Suction_Test',
        'Seed': seed,
        'IAE_LAP': iae_lap,
        'IAE_RAP': iae_rap,
        'FBE_mean': round(float(np.mean(fbes)), 3),
        'FBE_peak': round(float(np.max(fbes)), 3),
        'MAP_mean': round(float(np.mean(map_pressures)), 2),
        'CO_mean': round(float(np.mean(cos)), 3),
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

    print(f">>> [FINISHED Worker Seed {seed}] in {elapsed_total/60:.2f} min! "
          f"IAE={iae_lap:.1f}, Suction_Pct={suction_pct:.1f}%, CO={metrics['CO_mean']:.2f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parallel MPC Suction Test Evaluator")
    parser.add_argument("--seeds", type=str, default="42,123,456,789",
                        help="Comma-separated seed list (e.g. 42,123,456,789 or 1024)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum simulation steps per episode (default: 2000)")
    parser.add_argument("--output_dir", type=str, default="./mpc_suction_output",
                        help="Output directory for CSV results")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PARALLEL MPC SCENARIO 9 (SUCTION TEST) EVALUATOR")
    print(f"Seeds to Evaluate : {seeds}")
    print(f"Parallel Workers  : {args.workers}")
    print(f"Episode Steps     : {args.max_steps} (200s)")
    print(f"Output Directory  : {args.output_dir}")
    print("=" * 70)

    t_start = time.time()
    results = []

    if len(seeds) == 1:
        # Run directly in main process if single seed
        res = run_single_mpc_suction(seeds[0], max_steps=args.max_steps)
        results.append(res)
    else:
        # Run in parallel worker pool
        n_workers = min(args.workers, len(seeds))
        print(f"Dispatching {len(seeds)} seeds across {n_workers} concurrent processes...\n")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_seed = {
                executor.submit(run_single_mpc_suction, sd, args.max_steps): sd
                for sd in seeds
            }
            for future in as_completed(future_to_seed):
                sd = future_to_seed[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"[ERROR] Worker Seed {sd} failed: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "mpc_suction_results.csv")
    df.to_csv(csv_path, index=False)

    elapsed_all = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE IN {elapsed_all/60:.2f} MINUTES!")
    print(f"Results saved to: {csv_path}")
    print("=" * 70)
    print(df[['Seed', 'IAE_LAP', 'Suction_Pct', 'FBE_mean', 'MAP_mean', 'CO_mean', 'Mean_Reward']])


if __name__ == '__main__':
    main()

