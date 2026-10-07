"""
Full evaluation harness: 4 controllers × 8 scenarios.

Reads ALL reward components from env.step() info dict (single source of truth).
Outputs a clean summary table + per-controller action diagnostics.
This is the final integration test before Phase 5 DRL training.
"""
import numpy as np
import time
import sys
from bivad_env import BiVADEnv
from controllers.constant_speed import ConstantSpeedController
from controllers.pi_controller import PIDecentralizedController
from controllers.starling_controller import StarlingController
from controllers.mpc_controller import MPCController

SCENARIOS = {
    1: {'name': 'Rest',              'changes': {}},
    2: {'name': 'SVR_up',            'changes': {500: lambda p: p.update({'R_sys': 1.5})}},
    3: {'name': 'SVR_down',          'changes': {500: lambda p: p.update({'R_sys': 0.6})}},
    4: {'name': 'PVR_up',            'changes': {500: lambda p: p.update({'R_pulm': 0.3})}},
    5: {'name': 'Exercise',          'changes': {500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys']*0.8})}},
    6: {'name': 'Exercise_Recovery',  'changes': {
            500: lambda p: p.update({'HR': 100.0, 'R_sys': p['R_sys']*0.8}),
            1200: lambda p: p.update({'HR': 75.0, 'R_sys': p['R_sys']/0.8})
    }},
    7: {'name': 'Hypovolemia',       'changes': {0: lambda p: p.update({
            'C_la': p['C_la']*0.8, 'C_ra': p['C_ra']*0.8,
            'C_sys': p['C_sys']*0.8, 'C_pulm': p['C_pulm']*0.8
    })}},
    8: {'name': 'Combined',          'changes': {500: lambda p: p.update({
            'R_sys': p['R_sys']*1.3, 'R_pulm': p['R_pulm']*1.5, 'HR': p['HR']*1.2
    })}},
}


def run_episode(controller_name, controller, scenario_id):
    """Run one controller through one scenario. Returns summary dict and trajectory rows."""
    scenario = SCENARIOS[scenario_id]
    env = BiVADEnv(randomize=False)
    env.perturbation_schedule = {}

    obs, _ = env.reset()
    if hasattr(controller, 'reset'):
        controller.reset()

    # Apply time-0 changes
    if 0 in scenario['changes']:
        scenario['changes'][0](env.params)

    info = {
        'LVAD_RPM': env.params['LVAD_RPM'],
        'RVAD_RPM': env.params['RVAD_RPM'],
        'true_state': env.state,
        'env_params': env.params
    }

    totals = []
    tracks = []
    balances = []
    safeties = []
    smooths = []
    maps = []
    action_deltas = []
    laps = []
    raps = []
    flow_imbalances = []
    trajectory_rows = []

    for step_i in range(env.max_steps):
        if step_i in scenario['changes'] and step_i > 0:
            scenario['changes'][step_i](env.params)

        action = controller.get_action(obs, info)
        obs, reward, terminated, _, info = env.step(action)

        totals.append(reward)
        tracks.append(info['reward_track'])
        balances.append(info['reward_balance'])
        safeties.append(info['reward_safety'])
        smooths.append(info['reward_smooth'])
        maps.append(info.get('reward_map', 0.0))
        action_deltas.append((abs(action[0]) + abs(action[1])) / 2.0)

        # Physiological metrics
        y = info.get('true_state', env.state)
        lap, rap = y[2], y[3]
        aop = y[4]
        lvad_flow = y[10]
        co = lvad_flow * 60.0 / 1000.0
        fb_err = info.get('flow_balance_error', abs(y[10] - y[11]))
        
        laps.append(lap)
        raps.append(rap)
        flow_imbalances.append(fb_err)

        # Record trajectory row matching Phase 3 format
        trajectory_rows.append({
            'Time': round((step_i + 1) * env.dt, 2),
            'Controller': controller_name,
            'Scenario': scenario['name'],
            'MAP': aop,
            'LAP': lap,
            'RAP': rap,
            'LVAD_RPM': info['LVAD_RPM'],
            'RVAD_RPM': info['RVAD_RPM'],
            'CO_left': co,
            'flow_balance_error': fb_err,
            'Reward': reward
        })

        if terminated:
            break

    summary = {
        'total': np.mean(totals),
        'r_track': np.mean(tracks),
        'r_balance': np.mean(balances),
        'r_safety': np.mean(safeties),
        'r_smooth': np.mean(smooths),
        'r_map': np.mean(maps),
        'mean_action': np.mean(action_deltas),
        'mean_lap': np.mean(laps),
        'mean_rap': np.mean(raps),
        'mean_balance_err': np.mean(flow_imbalances),
        'safety_pct': 100.0 * np.mean(np.array(safeties) != 0),
        'n_steps': len(totals),
    }

    return summary, trajectory_rows


if __name__ == '__main__':
    controllers = {
        'Constant_Speed': ConstantSpeedController(),
        'PI':             PIDecentralizedController(),
        'Starling':       StarlingController(),
        'MPC':            MPCController(Hp=5),
    }

    skip_mpc = '--skip-mpc' in sys.argv
    only_mpc = '--only-mpc' in sys.argv
    only_starling = '--only-starling' in sys.argv

    ctrl_arg = [arg for arg in sys.argv if arg.startswith('--controllers=')]

    if ctrl_arg:
        selected = ctrl_arg[0].split('=')[1].split(',')
        controllers = {k: v for k, v in controllers.items() if k in selected}
        print(f"Running controllers: {list(controllers.keys())}\n")
    elif skip_mpc:
        del controllers['MPC']
        print("Skipping MPC (--skip-mpc flag set)\n")
    elif only_mpc:
        controllers = {'MPC': MPCController(Hp=5)}
        print("Running ONLY MPC (--only-mpc flag set) — will update existing CSV\n")
    elif only_starling:
        controllers = {'Starling': StarlingController()}
        print("Running ONLY Starling (--only-starling flag set) — will update existing CSV\n")

    all_results = {}
    all_trajectories = []
    print(f"Running baseline evaluation across 8 scenarios...\n")

    for c_name, ctrl in controllers.items():
        for s_id, scenario in SCENARIOS.items():
            s_name = scenario['name']
            t0 = time.time()
            print(f"  {c_name:18s} / {s_name:20s} ...", end='', flush=True)
            result, traj = run_episode(c_name, ctrl, s_id)
            elapsed = time.time() - t0
            all_results[(c_name, s_name)] = result
            all_trajectories.extend(traj)
            print(f"  {elapsed:.1f}s")

    # ================================================================
    # Save CSV Trajectory (merges or creates evaluation_results.csv)
    # ================================================================
    import pandas as pd
    import os
    df_new = pd.DataFrame(all_trajectories)
    output_csv = 'evaluation_results.csv'
    is_subset = len(controllers) < 4
    if is_subset and os.path.exists(output_csv):
        df_old = pd.read_csv(output_csv)
        for c in controllers:
            df_old = df_old[df_old['Controller'] != c]
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged.to_csv(output_csv, index=False)
        print(f"\n[UPDATED] Merged {list(controllers.keys())} runs into {output_csv} ({len(df_merged)} total rows)")
    else:
        df_new.to_csv(output_csv, index=False)
        print(f"\n[SAVED] Full trajectory ({len(df_new)} rows) saved to {output_csv}")

    # ================================================================
    # Summary table — reward components
    # ================================================================
    print(f"\n{'='*125}")
    print("REWARD SUMMARY TABLE")
    print(f"{'='*125}")
    print(f"{'Controller':18s} {'Scenario':20s} {'Total':>9s} {'R_track':>9s} {'R_bal':>9s} {'R_safe':>9s} {'R_smooth':>9s} {'R_map':>9s} {'|delta|':>8s} {'LAP':>6s} {'RAP':>6s} {'BalErr':>8s}")
    print("-" * 125)

    for c_name in controllers:
        for s_id in sorted(SCENARIOS.keys()):
            s_name = SCENARIOS[s_id]['name']
            r = all_results[(c_name, s_name)]
            print(f"{c_name:18s} {s_name:20s} {r['total']:9.3f} {r['r_track']:9.3f} {r['r_balance']:9.3f} {r['r_safety']:9.3f} {r['r_smooth']:9.3f} {r.get('r_map', 0.0):9.3f} {r['mean_action']:8.2f} {r['mean_lap']:6.1f} {r['mean_rap']:6.1f} {r['mean_balance_err']:8.2f}")
        print()

    # ================================================================
    # Per-scenario winner table
    # ================================================================
    print(f"\n{'='*70}")
    print("PER-SCENARIO WINNER")
    print(f"{'='*70}")
    print(f"{'Scenario':20s} {'Winner':18s} {'Total':>9s} {'Runner-up':18s} {'Total':>9s}")
    print("-" * 70)

    for s_id in sorted(SCENARIOS.keys()):
        s_name = SCENARIOS[s_id]['name']
        scores = [(c_name, all_results[(c_name, s_name)]['total']) for c_name in controllers]
        scores.sort(key=lambda x: x[1], reverse=True)
        winner = scores[0]
        runner = scores[1] if len(scores) > 1 else ('N/A', 0.0)
        print(f"{s_name:20s} {winner[0]:18s} {winner[1]:9.3f} {runner[0]:18s} {runner[1]:9.3f}")

    # ================================================================
    # Rest ranking — explicit pass bar
    # ================================================================
    print(f"\n{'='*70}")
    print("REST RANKING — Pass Bar Check")
    print(f"{'='*70}")

    rest_scores = [(c_name, all_results[(c_name, 'Rest')]['total']) for c_name in controllers]
    rest_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, val) in enumerate(rest_scores, 1):
        print(f"  {rank}. {name:18s}: {val:.4f}")

    if 'Constant_Speed' in all_results and 'PI' in all_results:
        cs_val = all_results[('Constant_Speed', 'Rest')]['total']
        pi_val = all_results[('PI', 'Rest')]['total']

        if 'MPC' in controllers:
            mpc_val = all_results[('MPC', 'Rest')]['total']
            if mpc_val > cs_val:
                print(f"\n[PASS] MPC ({mpc_val:.4f}) > Constant_Speed ({cs_val:.4f})")
            else:
                print(f"\n[FAIL] MPC ({mpc_val:.4f}) <= Constant_Speed ({cs_val:.4f})")
                
            if mpc_val > pi_val:
                print(f"[INFO] MPC ({mpc_val:.4f}) > PI ({pi_val:.4f}) — MPC leads baseline ranking")
            else:
                print(f"[INFO] PI ({pi_val:.4f}) > MPC ({mpc_val:.4f}) — gap: {pi_val - mpc_val:.4f}")

        if pi_val > cs_val:
            print(f"[PASS] PI ({pi_val:.4f}) > Constant_Speed ({cs_val:.4f})")
        else:
            print(f"[FAIL] PI ({pi_val:.4f}) <= Constant_Speed ({cs_val:.4f})")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
