"""
Step 4.6 — Reward weight calibration.

Runs all 4 baseline controllers through Rest + PVR_up + SVR_up scenarios,
logs unweighted reward component magnitudes, and computes
suggested weights so that under MPC-at-Rest all three tunable
terms (track, balance, smooth) contribute comparable magnitude.

Single source of truth: all reward components read from env.step() info dict.
"""
import numpy as np
import time
from bivad_env import BiVADEnv
from controllers.constant_speed import ConstantSpeedController
from controllers.pi_controller import PIDecentralizedController
from controllers.starling_controller import StarlingController
from controllers.mpc_controller import MPCController

SCENARIOS = {
    'Rest':    {},
    'PVR_up':  {500: lambda p: p.update({'R_pulm': 0.3})},
    'SVR_up':  {500: lambda p: p.update({'R_sys': 1.5})},
}


def run_calibration(controller_name, controller, scenario_name, scenario_changes):
    env = BiVADEnv(randomize=False)
    env.perturbation_schedule = {}
    obs, _ = env.reset()
    if hasattr(controller, 'reset'):
        controller.reset()

    # Apply time-0 changes
    if 0 in scenario_changes:
        scenario_changes[0](env.params)

    info = {
        'LVAD_RPM': env.params['LVAD_RPM'],
        'RVAD_RPM': env.params['RVAD_RPM'],
        'true_state': env.state,
        'env_params': env.params
    }

    tracking_errors = []
    flow_imbalances = []
    safety_penalties = []
    smooth_values = []
    actions_left = []
    actions_right = []

    for step_i in range(env.max_steps):
        if step_i in scenario_changes and step_i > 0:
            scenario_changes[step_i](env.params)

        action = controller.get_action(obs, info)
        obs, _, terminated, _, info = env.step(action)

        # Single source of truth: read directly from info dict
        tracking_errors.append(info['tracking_error_raw'])
        flow_imbalances.append(info['flow_imbalance_raw'])
        safety_penalties.append(info['reward_safety'])
        smooth_values.append(info['smooth_raw'])
        actions_left.append(abs(action[0]))
        actions_right.append(abs(action[1]))

        if terminated:
            break

    te_arr = np.array(tracking_errors)
    fi_arr = np.array(flow_imbalances)
    sp_arr = np.array(safety_penalties)
    sv_arr = np.array(smooth_values)
    al_arr = np.array(actions_left)
    ar_arr = np.array(actions_right)
    act_arr = (al_arr + ar_arr) / 2.0  # mean of left+right per step

    return {
        'tracking_mean': np.mean(te_arr),
        'tracking_std': np.std(te_arr),
        'balance_mean': np.mean(fi_arr),
        'balance_std': np.std(fi_arr),
        'safety_mean': np.mean(sp_arr),
        'safety_nonzero_pct': 100.0 * np.mean(sp_arr != 0),
        'smooth_mean': np.mean(sv_arr),
        'smooth_nonzero_pct': 100.0 * np.mean(sv_arr != 0),
        'action_mean': np.mean(act_arr),
        'action_max': np.max(act_arr),
        'action_p95': np.percentile(act_arr, 95),
        'action_p99': np.percentile(act_arr, 99),
        'actions_left': al_arr,
        'actions_right': ar_arr,
    }


if __name__ == '__main__':
    controllers = {
        'Constant_Speed': ConstantSpeedController(),
        'PI':             PIDecentralizedController(),
        'Starling':       StarlingController(),
        'MPC':            MPCController(Hp=5),
    }

    results = {}

    for c_name, ctrl in controllers.items():
        for s_name, s_changes in SCENARIOS.items():
            t0 = time.time()
            print(f"Running {c_name:18s} / {s_name:8s} ...", end='', flush=True)
            r = run_calibration(c_name, ctrl, s_name, s_changes)
            elapsed = time.time() - t0
            results[(c_name, s_name)] = r
            print(f"  {elapsed:.1f}s")
            print(f"    Track:   {r['tracking_mean']:8.2f} +/- {r['tracking_std']:.2f} mL/s")
            print(f"    Balance: {r['balance_mean']:8.2f} +/- {r['balance_std']:.2f} mL/s")
            print(f"    Safety:  {r['safety_mean']:8.2f}  (nonzero: {r['safety_nonzero_pct']:.1f}%)")
            print(f"    Smooth:  {r['smooth_mean']:8.2f}  (nonzero: {r['smooth_nonzero_pct']:.1f}%)")
            print(f"    Action:  mean={r['action_mean']:.2f}  max={r['action_max']:.2f}  p95={r['action_p95']:.2f}  p99={r['action_p99']:.2f} RPM")

    # ================================================================
    # MPC Action Diagnostic — first 10 steps detail
    # ================================================================
    print(f"\n{'='*70}")
    print("MPC ACTION DIAGNOSTIC — First 10 steps at Rest")
    print(f"{'='*70}")
    mpc_rest = results[('MPC', 'Rest')]
    al = mpc_rest['actions_left']
    ar = mpc_rest['actions_right']
    print(f"{'Step':>5s}  {'|dL|':>8s}  {'|dR|':>8s}  {'> deadzone?':>12s}")
    print("-" * 40)
    for i in range(min(10, len(al))):
        fires = "YES" if (al[i] > 30.0 or ar[i] > 30.0) else "no"
        print(f"{i:5d}  {al[i]:8.2f}  {ar[i]:8.2f}  {fires:>12s}")

    # Count steps where dead zone fires across full episode
    dz_steps = np.sum((al > 30.0) | (ar > 30.0))
    print(f"\nTotal steps > dead zone: {dz_steps} / {len(al)} ({100*dz_steps/len(al):.1f}%)")

    # ================================================================
    # Weight calibration
    # ================================================================
    print(f"\n{'='*70}")
    print("WEIGHT CALIBRATION — Multi-scenario reference: MPC (Rest, PVR_up, SVR_up)")
    print(f"{'='*70}")

    rep_scenarios = ['Rest', 'PVR_up', 'SVR_up']
    T = float(np.mean([results[('MPC', s)]['tracking_mean'] for s in rep_scenarios]))
    F = float(np.mean([results[('MPC', s)]['balance_mean'] for s in rep_scenarios]))
    S = float(np.mean([results[('MPC', s)]['smooth_mean'] for s in rep_scenarios]))

    TARGET = 0.5

    w1 = TARGET / T if T > 1e-6 else 0.01
    w2 = TARGET / F if F > 1e-6 else 0.02
    w3 = TARGET / S if S > 1e-6 else 0.01

    print(f"\n  MPC Multi-Scenario mean raw magnitudes (Rest, PVR_up, SVR_up):")
    print(f"    Tracking error : {T:.2f} mL/s")
    print(f"    Flow imbalance : {F:.2f} mL/s")
    print(f"    Smooth penalty : {S:.2f} RPM")
    print(f"\n  Suggested weights (target ~{TARGET} per component):")
    print(f"    w1 = {w1:.6f}")
    print(f"    w2 = {w2:.6f}")
    print(f"    w3 = {w3:.6f}")

    # ================================================================
    # Cross-check: estimated per-step reward under each controller
    # ================================================================
    for check_scenario in ['Rest', 'PVR_up', 'SVR_up']:
        print(f"\n{'='*70}")
        print(f"CROSS-CHECK — Estimated mean per-step reward at {check_scenario}")
        print(f"{'='*70}")
        print(f"{'Controller':18s} {'R_track':>10s} {'R_balance':>10s} {'R_safety':>10s} {'R_smooth':>10s} {'TOTAL':>10s}")
        print("-" * 70)

        for c_name in controllers:
            r = results[(c_name, check_scenario)]
            rt = -w1 * r['tracking_mean']
            rb = -w2 * r['balance_mean']
            rs = r['safety_mean']     # already signed
            rsm = -w3 * r['smooth_mean']
            total = rt + rb + rs + rsm
            print(f"{c_name:18s} {rt:10.3f} {rb:10.3f} {rs:10.3f} {rsm:10.3f} {total:10.3f}")

    # ================================================================
    # Pass bar check
    # ================================================================
    print(f"\n{'='*70}")
    print("PASS BAR CHECK — MPC vs Constant_Speed at Rest")
    print(f"{'='*70}")
    r_mpc = results[('MPC', 'Rest')]
    r_cs  = results[('Constant_Speed', 'Rest')]
    mpc_total = (-w1 * r_mpc['tracking_mean'] - w2 * r_mpc['balance_mean']
                 + r_mpc['safety_mean'] - w3 * r_mpc['smooth_mean'])
    cs_total = (-w1 * r_cs['tracking_mean'] - w2 * r_cs['balance_mean']
                + r_cs['safety_mean'] - w3 * r_cs['smooth_mean'])
    print(f"  MPC total:            {mpc_total:.4f}")
    print(f"  Constant_Speed total: {cs_total:.4f}")
    if mpc_total > cs_total:
        print(f"  [PASS] MPC > Constant_Speed by {mpc_total - cs_total:.4f}")
    else:
        print(f"  [FAIL] MPC < Constant_Speed by {cs_total - mpc_total:.4f}")
        print(f"  ACTION NEEDED: w3 may be too large, or MPC startup transient needs grace period")

    # ================================================================
    # Scenario Divergence Check (Rest vs SVR_up & Rest vs PVR_up)
    # Detects if any active controller exhibits identical metrics across distinct scenarios
    # ================================================================
    print(f"\n{'='*70}")
    print("SCENARIO DIVERGENCE CHECK (Rest vs SVR_up & Rest vs PVR_up)")
    print(f"{'='*70}")
    div_pass = True
    for c_name in controllers:
        if c_name == 'Constant_Speed':
            continue
        r_rest = results[(c_name, 'Rest')]
        for s_other in ['SVR_up', 'PVR_up']:
            r_other = results[(c_name, s_other)]
            d_act = abs(r_rest['action_mean'] - r_other['action_mean'])
            d_sm = abs(r_rest['smooth_mean'] - r_other['smooth_mean'])
            if d_act < 1e-4 and r_rest['action_mean'] > 0.01:
                print(f"  [WARN] {c_name} action_mean is IDENTICAL between Rest and {s_other}: {r_rest['action_mean']:.4f}")
                div_pass = False
            if d_sm < 1e-4 and r_rest['smooth_mean'] > 0.01:
                print(f"  [WARN] {c_name} smooth_mean is IDENTICAL between Rest and {s_other}: {r_rest['smooth_mean']:.4f}")
                div_pass = False
    if div_pass:
        print("  [PASS] All active controllers respond distinctly across scenarios.")

    # ================================================================
    # Final output — copy-paste into reward.py
    # ================================================================
    print(f"\n{'='*70}")
    print("COPY INTO reward.py DEFAULT_WEIGHTS:")
    print(f"{'='*70}")
    print(f"DEFAULT_WEIGHTS = {{")
    print(f"    'w1': {w1:.6f},    # R_track")
    print(f"    'w2': {w2:.6f},    # R_balance")
    print(f"    'w3': {w3:.6f},    # R_smooth")
    print(f"}}")
