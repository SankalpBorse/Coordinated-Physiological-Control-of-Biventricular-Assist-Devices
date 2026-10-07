"""
Phase 4 — Multi-objective reward function for BiVAD DRL controller.

Architecture:  R = R_track + R_balance + R_safety + R_smooth

Components:
    R_track  : Demand tracking via Frank-Starling flow reference (same as MPC baseline, Koh 2020)
    R_balance: Left-right flow conservation (novel contribution)
    R_safety : Koh's threshold-based suction/congestion penalty (zero inside safe band)
    R_smooth : Control effort penalty with dead zone (Shi 2025 style)

Sources:
    - Starling reference: Koh 2020, Eq. 11 (shared with mpc_controller.py)
    - Safety thresholds: Koh 2020, Table 3  (marginal suction <3 mmHg, congestion >18 mmHg)
    - Smooth dead zone: adapted from Shi 2025 R3, scaled to our ±110 RPM action space
"""
import numpy as np

# ============================================================
# Frank-Starling Flow Reference Constants
# Shared with MPC baseline (mpc_controller.py, Step 3.5)
# Q_ref = K1 / (1 + exp(-K2 * (P_atrial - offset)))
# Units: Q in mL/s, P in mmHg
# ============================================================
STARLING_K1 = 115.0           # mL/s asymptote (~6.9 L/min, calibrated in Phase 1)
STARLING_K2 = 0.5             # sigmoid steepness
STARLING_OFFSET_L = 9.0       # left inflection point (mmHg)
STARLING_OFFSET_R = 4.0       # right inflection point (mmHg)

# ============================================================
# Safety Thresholds (Koh 2020)
# Zero penalty inside [3, 18] mmHg — intentionally off during
# normal operation so tracking signal isn't drowned out.
# ============================================================
SUCTION_MARGIN = 3.0          # marginal suction below this (mmHg)
CONGESTION_LIMIT = 18.0       # congestion above this (mmHg)
SUCTION_LINEAR_W = 20.0       # linear ramp slope (marginal suction)
CONGESTION_LINEAR_W = 20.0    # linear ramp slope (congestion)
REAL_SUCTION_PENALTY = 50.0   # flat penalty for LVP/RVP < 0 mmHg

# ============================================================
# MAP Penalty Thresholds (Shi et al. 2025 adaptation)
# Penalizes mean arterial pressure deviation outside [70, 110] mmHg
# ============================================================
MAP_LOW = 70.0                # lower physiological limit (mmHg)
MAP_HIGH = 120.0              # upper physiological limit (mmHg, compensatory threshold)
MAP_WEIGHT = 10.0             # linear ramp slope

# ============================================================
# Smooth Dead Zone
# 27% of max action (110 RPM). Below this, small corrections
# are free — only penalize large oscillatory control effort.
# Continuous ramp above dead zone without cross-actuator penalty.
# ============================================================
SMOOTH_DEADZONE = 30.0

# ============================================================
# Reward Weights — calibrated in Phase 1 & 3 (under K1 = 115.0 mL/s)
# Target ~0.5 per active component per step.
# ============================================================
DEFAULT_WEIGHTS = {
    'w1': 0.678709,    # R_track   (calibrated: 0.5 / 0.7367 multi-scenario mean)
    'w2': 0.012261,    # R_balance (calibrated: 0.5 / 40.78 mL/s multi-scenario mean)
    'w3': 0.104327,    # R_smooth  (calibrated: 0.5 / 4.79 RPM multi-scenario mean)
    'w_map': 0.05,     # R_map     (calibrated: 0.5 / ~10.0 mean deviation when active)
}

def compute_starling_reference(lap, rap):
    """
    Frank-Starling sigmoid flow reference (same curve as MPC baseline).
    Returns physiological target flows (mL/s) for left and right pumps.
    """
    q_l = STARLING_K1 / (1.0 + np.exp(-STARLING_K2 * (lap - STARLING_OFFSET_L)))
    q_r = STARLING_K1 / (1.0 + np.exp(-STARLING_K2 * (rap - STARLING_OFFSET_R)))
    return q_l, q_r


def compute_safety_penalty(lap, rap, lvp, rvp):
    """
    Koh's exact safety thresholds.
    Returns penalty (negative or zero).
    
    Zero inside safe band [3, 18] mmHg for atrial pressures.
    Heavy penalty outside — intentionally asymmetric so it dominates
    only when triggered, not during normal operation.
    """
    p = 0.0
    # Marginal suction: linear ramp below 3 mmHg
    if lap < SUCTION_MARGIN:
        p -= SUCTION_LINEAR_W * (SUCTION_MARGIN - lap)
    if rap < SUCTION_MARGIN:
        p -= SUCTION_LINEAR_W * (SUCTION_MARGIN - rap)
    # Congestion: linear ramp above 18 mmHg
    if lap > CONGESTION_LIMIT:
        p -= CONGESTION_LINEAR_W * (lap - CONGESTION_LIMIT)
    if rap > CONGESTION_LIMIT:
        p -= CONGESTION_LINEAR_W * (rap - CONGESTION_LIMIT)
    # Real ventricular suction: flat penalty
    if lvp < 0:
        p -= REAL_SUCTION_PENALTY
    if rvp < 0:
        p -= REAL_SUCTION_PENALTY
    return p


def compute_map_penalty(map_value):
    """
    Linear-hinge penalty when MAP falls outside physiological range [70, 110] mmHg.
    Inspired by Shi et al. (2025).
    """
    if map_value > MAP_HIGH:
        return -MAP_WEIGHT * (map_value - MAP_HIGH)
    elif map_value < MAP_LOW:
        return -MAP_WEIGHT * (MAP_LOW - map_value)
    return 0.0


def compute_reward(lap, rap, lvp, rvp, lvad_flow, rvad_flow, action, map_value=90.0, weights=None):
    """
    Multi-objective reward for BiVAD DRL training.

    Args:
        lap, rap       : Atrial pressures (mmHg)
        lvp, rvp       : Ventricular pressures (mmHg)
        lvad_flow      : Left pump flow (mL/s)
        rvad_flow      : Right pump flow (mL/s)
        action         : [delta_lvad_rpm, delta_rvad_rpm]
        map_value      : Mean arterial pressure AoP (mmHg)
        weights        : dict with 'w1', 'w2', 'w3'; uses DEFAULT_WEIGHTS if None

    Returns:
        total_reward (float), components (dict)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    w1 = weights['w1']
    w2 = weights['w2']
    w3 = weights['w3']
    w_map = weights.get('w_map', 0.05)

    # ---- R_track: demand tracking via Frank-Starling reference (per-side normalized) ----
    q_target_l, q_target_r = compute_starling_reference(lap, rap)
    tracking_error = (abs(lvad_flow - q_target_l) / max(q_target_l, 1.0) +
                      abs(rvad_flow - q_target_r) / max(q_target_r, 1.0))
    r_track = -w1 * tracking_error

    # ---- R_balance: left-right flow conservation (novel) ----
    flow_imbalance = abs(lvad_flow - rvad_flow)
    r_balance = -w2 * flow_imbalance

    # ---- R_safety: Koh's thresholds (unweighted, internally scaled) ----
    r_safety = compute_safety_penalty(lap, rap, lvp, rvp)

    # ---- R_smooth: control effort with continuous dead zone ----
    delta_l = abs(action[0])
    delta_r = abs(action[1])
    smooth_raw = max(0.0, delta_l - SMOOTH_DEADZONE) + max(0.0, delta_r - SMOOTH_DEADZONE)
    r_smooth = -w3 * smooth_raw

    # ---- R_map: linear-hinge MAP penalty outside [70, 120] mmHg (weight calibrated) ----
    r_map = w_map * compute_map_penalty(map_value)

    total = r_track + r_balance + r_safety + r_smooth + r_map

    components = {
        'r_track': r_track,
        'r_balance': r_balance,
        'r_safety': r_safety,
        'r_smooth': r_smooth,
        'r_map': r_map,
        'tracking_error_raw': tracking_error,
        'flow_imbalance_raw': flow_imbalance,
        'smooth_raw': smooth_raw,
        'map_raw': map_value,
        'w_map': w_map,
        'q_target_l': q_target_l,
        'q_target_r': q_target_r,
    }

    return total, components
