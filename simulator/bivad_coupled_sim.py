import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from simulator.ventricular_elastance import time_varying_elastance
from simulator.vascular_primitives import compute_dp_dt, compute_flow_resistive, compute_dq_dt
from simulator.lvad_heart_sim import compute_suction_resistance, compute_pump_head

# Coupled BiVAD Parameters
P = {
    # Systemic (Left)
    'R_sys': 1.0000,
    'R_c': 0.0398,
    'R_m': 0.0050,
    'R_a': 0.0010,
    'C_la': 4.4000,
    'C_sys': 1.3300,
    'C_ao': 0.0800,
    'L_ao': 0.0005,
    'E_max_lv': 0.80,
    'E_min_lv': 0.06,

    # Pulmonary (Right)
    'R_pulm': 0.1000,
    'R_pc': 0.0398,
    'R_tv': 0.0050,
    'R_pv': 0.0010,
    'C_ra': 4.4000,
    'C_pulm': 1.3300,
    'C_pa': 0.0800,
    'L_pa': 0.0005,
    'E_max_rv': 0.30,
    'E_min_rv': 0.03,

    # Shared / Global
    'HR': 75.0,
    'Ri': 0.0677,  # Cannulas (used for both pumps)
    'Ro': 0.0677,
    'Li': 0.0127,
    'Lo': 0.0127,
    
    # Koh HVAD coefficients
    'a_hvad': 0.18,
    'b_hvad': 0.0017,
    'c_hvad': 1.5e-5,
    
    # Pump Speeds - We tune RVAD much lower to avoid pulmonary hemorrhage 
    # and achieve a balanced steady-state flow of ~5 L/min.
    'LVAD_RPM': 2600.0,
    'RVAD_RPM': 1900.0,
}

def bivad_derivatives(t, y, params):
    """
    Fully Coupled BiVAD Simulation (12 State Variables)
    y = [V_LV, V_RV, LAP, RAP, AoP, PAP, SysP, PulmP, AoFlow, PAFlow, LVAD_Flow, RVAD_Flow]
    """
    V_lv, V_rv, lap, rap, aop, pap, sysp, pulmp, aoflow, paflow, lvad_flow, rvad_flow = y

    # 1. Ventricular Elastance
    E_lv = time_varying_elastance(t, hr=params['HR'], E_max=params['E_max_lv'], E_min=params['E_min_lv'])
    E_rv = time_varying_elastance(t, hr=params['HR'], E_max=params['E_max_rv'], E_min=params['E_min_rv'])
    
    lvp = V_lv * E_lv
    rvp = V_rv * E_rv

    # 2. Native Valves (Ideal Diodes)
    q_mitral = compute_flow_resistive(p_in=lap, p_out=lvp, resistance=params['R_m'], is_valve=True)
    q_aortic = compute_flow_resistive(p_in=lvp, p_out=aop, resistance=params['R_a'], is_valve=True)
    q_tricuspid = compute_flow_resistive(p_in=rap, p_out=rvp, resistance=params['R_tv'], is_valve=True)
    q_pulmonary = compute_flow_resistive(p_in=rvp, p_out=pap, resistance=params['R_pv'], is_valve=True)

    # 3. Capillary / Venous Return (Closing the loops)
    # Aorta -> Systemic -> Right Atrium
    q_sys_venous = compute_flow_resistive(p_in=sysp, p_out=rap, resistance=params['R_sys'], is_valve=False)
    # Pulmonary Artery -> Lungs -> Left Atrium
    q_pulm_venous = compute_flow_resistive(p_in=pulmp, p_out=lap, resistance=params['R_pulm'], is_valve=False)

    # 4. Pump Heads & Suction
    Rk_lv = compute_suction_resistance(lvp)
    Rk_rv = compute_suction_resistance(rvp)
    
    R_lvad_total = params['Ri'] + params['Ro'] + Rk_lv
    R_rvad_total = params['Ri'] + params['Ro'] + Rk_rv
    L_pump_total = params['Li'] + params['Lo']
    
    H_lvad = compute_pump_head(params['LVAD_RPM'], lvad_flow, params)
    H_rvad = compute_pump_head(params['RVAD_RPM'], rvad_flow, params)

    # 5. Derivatives (Mass & Momentum Conservation)
    # Ventricular Volumes
    dV_lv_dt = q_mitral - q_aortic - lvad_flow
    dV_rv_dt = q_tricuspid - q_pulmonary - rvad_flow
    
    # Atrial Pressures
    d_lap_dt = compute_dp_dt(inflow=q_pulm_venous, outflow=q_mitral, compliance=params['C_la'])
    d_rap_dt = compute_dp_dt(inflow=q_sys_venous, outflow=q_tricuspid, compliance=params['C_ra'])
    
    # Arterial Pressures
    d_aop_dt = compute_dp_dt(inflow=(q_aortic + lvad_flow), outflow=aoflow, compliance=params['C_ao'])
    d_pap_dt = compute_dp_dt(inflow=(q_pulmonary + rvad_flow), outflow=paflow, compliance=params['C_pa'])
    
    # Systemic/Pulmonary Capillary Pressures
    d_sysp_dt = compute_dp_dt(inflow=aoflow, outflow=q_sys_venous, compliance=params['C_sys'])
    d_pulmp_dt = compute_dp_dt(inflow=paflow, outflow=q_pulm_venous, compliance=params['C_pulm'])
    
    # Arterial Momentum Flows
    d_aoflow_dt = compute_dq_dt(p_in=aop, p_out=sysp, current_flow=aoflow, resistance=params['R_c'], inertance=params['L_ao'])
    d_paflow_dt = compute_dq_dt(p_in=pap, p_out=pulmp, current_flow=paflow, resistance=params['R_pc'], inertance=params['L_pa'])
    
    # Pump Flows
    d_lvad_flow_dt = ( (lvp - aop + H_lvad) - lvad_flow * R_lvad_total ) / L_pump_total
    d_rvad_flow_dt = ( (rvp - pap + H_rvad) - rvad_flow * R_rvad_total ) / L_pump_total

    return [
        dV_lv_dt, dV_rv_dt, d_lap_dt, d_rap_dt, d_aop_dt, d_pap_dt, 
        d_sysp_dt, d_pulmp_dt, d_aoflow_dt, d_paflow_dt, d_lvad_flow_dt, d_rvad_flow_dt
    ]

def simulate_bivad_coupled():
    print("Starting Step 6 Simulation: Fully Coupled BiVAD (LVAD 2600 RPM, RVAD 1900 RPM)...")
    
    cycle_time = 60.0 / P['HR']
    # Run longer (40 cycles) to guarantee volume shifts settle into true steady state
    t_end = 40 * cycle_time 
    t_eval = np.linspace(0, t_end, 8000)
    
    # Total blood volume is implicitly set by these initial conditions (~5 liters total)
    # [V_LV, V_RV, LAP, RAP, AoP, PAP, SysP, PulmP, AoFlow, PAFlow, LVAD, RVAD]
    y0 = [140.0, 140.0, 10.0, 5.0, 80.0, 20.0, 75.0, 12.0, 80.0, 80.0, 80.0, 80.0]
    
    sol = solve_ivp(
        fun=lambda t, y: bivad_derivatives(t, y, P),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method='Radau',
        max_step=0.005
    )
    
    t = sol.t
    V_lv, V_rv, lap, rap, aop, pap, sysp, pulmp, aoflow, paflow, lvad_flow, rvad_flow = sol.y
    
    # Reconstruct variables for analysis
    E_lv = np.array([time_varying_elastance(ti, hr=P['HR'], E_max=P['E_max_lv'], E_min=P['E_min_lv']) for ti in t])
    E_rv = np.array([time_varying_elastance(ti, hr=P['HR'], E_max=P['E_max_rv'], E_min=P['E_min_rv']) for ti in t])
    
    lvp = V_lv * E_lv
    rvp = V_rv * E_rv
    
    q_mitral = np.maximum(0, (lap - lvp) / P['R_m'])
    q_aortic = np.maximum(0, (lvp - aop) / P['R_a'])
    q_tricuspid = np.maximum(0, (rap - rvp) / P['R_tv'])
    q_pulmonary = np.maximum(0, (rvp - pap) / P['R_pv'])
    
    # Calculate Total Left and Right Output Flows
    total_left_flow = q_aortic + lvad_flow
    total_right_flow = q_pulmonary + rvad_flow

    # Look at the last 4 cycles for steady state
    mask = t >= (t_end - 4 * cycle_time)
    
    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # 1. Flow Conservation (Left vs Right)
    axes[0].plot(t[mask], total_left_flow[mask], 'r', label='Total Left Heart Flow')
    axes[0].plot(t[mask], total_right_flow[mask], 'b', label='Total Right Heart Flow')
    axes[0].set_ylabel('Flow (mL/s)')
    axes[0].set_title('Flow Conservation Check: Left vs Right Flow')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # 2. Four-Valve Timing
    axes[1].plot(t[mask], q_mitral[mask], 'b', label='Mitral')
    axes[1].plot(t[mask], q_aortic[mask], 'r', label='Aortic')
    axes[1].plot(t[mask], q_tricuspid[mask], 'c', linestyle='--', label='Tricuspid')
    axes[1].plot(t[mask], q_pulmonary[mask], 'm', linestyle='--', label='Pulmonary')
    axes[1].set_ylabel('Flow (mL/s)')
    axes[1].set_title('Four-Valve Timing Check (No simultaneous open per side)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # 3. Bilateral Pressures
    axes[2].plot(t[mask], aop[mask], 'r', label='Aortic Pressure (AoP)')
    axes[2].plot(t[mask], pap[mask], 'b', label='Pulm Artery Pressure (PAP)')
    axes[2].plot(t[mask], lap[mask], 'k', linestyle='--', label='Left Atrial (LAP)')
    axes[2].plot(t[mask], rap[mask], 'g', linestyle='--', label='Right Atrial (RAP)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Pressure (mmHg)')
    axes[2].set_title('Coupled Systemic and Pulmonary Pressures')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('bivad_coupled_validation.png', dpi=300)
    print("Plots saved to 'bivad_coupled_validation.png'.")
    
    # Diagnostics
    co_left = np.mean(total_left_flow[mask]) * 60 / 1000
    co_right = np.mean(total_right_flow[mask]) * 60 / 1000
    
    print("\n--- Diagnostic Numbers (Steady State) ---")
    print(f"Mean Aortic Pressure (MAP) : {np.mean(aop[mask]):.1f} mmHg")
    print(f"Mean Pulm Artery Press (PAP): {np.mean(pap[mask]):.1f} mmHg")
    print(f"Mean Left Atrial Press (LAP): {np.mean(lap[mask]):.1f} mmHg")
    print(f"Mean Right Atrial Press(RAP): {np.mean(rap[mask]):.1f} mmHg")
    print(f"Total Left Cardiac Output  : {co_left:.3f} L/min")
    print(f"Total Right Cardiac Output : {co_right:.3f} L/min")

if __name__ == "__main__":
    simulate_bivad_coupled()

