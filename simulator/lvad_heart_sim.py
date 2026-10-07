import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from simulator.ventricular_elastance import time_varying_elastance
from simulator.vascular_primitives import compute_dp_dt, compute_flow_resistive, compute_dq_dt

P = {
    'Rs': 1.0000,
    'Rm': 0.0050,
    'Ra': 0.0010,
    'Rc': 0.0398,
    'Cr': 4.4000,
    'Cs': 1.3300,
    'Ca': 0.0800,
    'Ls': 0.0005,
    'Li': 0.0127,
    'Lo': 0.0127,
    'Ri': 0.0677,
    'Ro': 0.0677,
    
    'HR': 75.0,
    'E_max': 0.80,  # Heart failure
    'E_min': 0.06,
    
    # Koh HVAD coefficients
    'a_hvad': 0.18,
    'b_hvad': 0.0017,
    'c_hvad': 1.5e-5,
    
    'pump_speed_rpm': 2500.0
}

def compute_suction_resistance(x1):
    """Table 1: Rk piecewise function"""
    if x1 > 1.0:
        return 0.0
    else:
        return -3.5 * (x1 - 1.0)

def compute_pump_head(rpm, q_ml_s, params):
    """
    Koh 2020 HVAD pump hydraulic model.
    NOTE: There is a major dimensional discrepancy with the user-provided units.
    If 'a' = 0.18 and we use Q in mL/s, a*Q^2 generates 1000+ mmHg of pressure drop.
    We suspect the coefficients from Koh's paper were meant for Q in L/min, 
    or the equation takes a different form. 
    Here we evaluate it assuming Q is in L/min to keep it physiologically stable.
    """
    q_l_min = q_ml_s * 60.0 / 1000.0
    
    # H = c*w^2 - b*w*Q - a*Q^2 
    # (Assuming standard centrifugal pump polynomial, scaling b by w might be needed. 
    # If we don't scale b by w, it might just be b*Q. We'll use the L/min assumption.)
    
    term_c = params['c_hvad'] * (rpm**2)
    
    # Assuming b multiplies Q directly (since b is very small) 
    # or b multiplies w and Q. We'll try the standard H = cw^2 - bQ - aQ^2 first, 
    # but 0.0017 is tiny without w. If multiplied by w=2500, it's 4.25, * 5 L/min = 21.25.
    term_b = params['b_hvad'] * rpm * q_l_min 
    
    term_a = params['a_hvad'] * (q_l_min**2)
    
    H = term_c - term_b - term_a
    return H

def cvs_lvad_derivatives(t, y):
    """
    Closed-loop ODE system matching Fig 1 WITH the LVAD.
    y = [V_LV, X_2, X_4, X_3, X_5, X_6]
    X_6 is LVAD Flow (mL/s)
    """
    V_lv, x2, x4, x3, x5, x6 = y

    # 1. Heart
    E = time_varying_elastance(t, hr=P['HR'], E_max=P['E_max'], E_min=P['E_min'])
    x1 = V_lv * E

    # 2. Flows
    q_venous = compute_flow_resistive(p_in=x3, p_out=x2, resistance=P['Rs'], is_valve=False)
    q_mitral = compute_flow_resistive(p_in=x2, p_out=x1, resistance=P['Rm'], is_valve=True)
    q_aortic = compute_flow_resistive(p_in=x1, p_out=x4, resistance=P['Ra'], is_valve=True)

    # 3. LVAD Branch
    Rk = compute_suction_resistance(x1)
    R_lvad_total = P['Ri'] + P['Ro'] + Rk
    L_lvad_total = P['Li'] + P['Lo']
    H_pump = compute_pump_head(P['pump_speed_rpm'], x6, P)

    # 4. Derivatives
    dV_lv_dt = q_mitral - q_aortic - x6
    
    dx2_dt = compute_dp_dt(inflow=q_venous, outflow=q_mitral, compliance=P['Cr'])
    
    # Aorta gets flow from native aortic valve AND LVAD
    dx4_dt = compute_dp_dt(inflow=(q_aortic + x6), outflow=x5, compliance=P['Ca'])
    dx3_dt = compute_dp_dt(inflow=x5, outflow=q_venous, compliance=P['Cs'])
    dx5_dt = compute_dq_dt(p_in=x4, p_out=x3, current_flow=x5, resistance=P['Rc'], inertance=P['Ls'])
    
    # LVAD Flow derivative: Driving pressure is (X1 - X4 + H)
    dx6_dt = ( (x1 - x4 + H_pump) - x6 * R_lvad_total ) / L_lvad_total

    return [dV_lv_dt, dx2_dt, dx4_dt, dx3_dt, dx5_dt, dx6_dt]

def simulate_lvad():
    print("Starting Step 4 Simulation: LVAD Support (Constant 2500 RPM)...")
    
    cycle_time = 60.0 / P['HR']
    t_end = 15 * cycle_time
    t_eval = np.linspace(0, t_end, 5000)
    
    # Initial conditions
    y0 = [140.0, 15.0, 70.0, 65.0, 0.0, 0.0]
    
    sol = solve_ivp(
        fun=cvs_lvad_derivatives,
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method='Radau',
        max_step=0.005
    )
    
    t = sol.t
    V_lv = sol.y[0]
    x2_lap = sol.y[1]
    x4_aop = sol.y[2]
    x3_artp = sol.y[3]
    x5_flow = sol.y[4]
    x6_lvad = sol.y[5]
    
    E = np.array([time_varying_elastance(ti, hr=P['HR'], E_max=P['E_max'], E_min=P['E_min']) for ti in t])
    x1_lvp = V_lv * E
    
    q_aortic = np.maximum(0, (x1_lvp - x4_aop) / P['Ra'])
    
    mask = t >= (t_end - 4 * cycle_time)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(t[mask], x4_aop[mask], 'r', label='Aortic Pressure (X4)')
    axes[0].plot(t[mask], x1_lvp[mask], 'k', label='LVP (X1)')
    axes[0].plot(t[mask], x2_lap[mask], 'b', label='LAP (X2)')
    axes[0].set_ylabel('Pressure (mmHg)')
    axes[0].set_title('LVAD Supported Heart (2500 RPM)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    axes[1].plot(t[mask], x6_lvad[mask], 'g', label='LVAD Pump Flow (X6)')
    axes[1].plot(t[mask], q_aortic[mask], 'r', label='Native Aortic Valve Flow')
    axes[1].plot(t[mask], x5_flow[mask], 'purple', label='Total Systemic Flow (X5)')
    axes[1].set_ylabel('Flow (mL/s)')
    axes[1].set_title('Pump vs Native Flow')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    axes[2].plot(t[mask], V_lv[mask], 'k', label='LV Volume')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Volume (mL)')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('lvad_heart_validation.png', dpi=300)
    print("Plots saved to 'lvad_heart_validation.png'.")
    
    # Diagnostics
    co_lvad = np.mean(x6_lvad[mask]) * 60 / 1000
    co_native = np.mean(q_aortic[mask]) * 60 / 1000
    print("\n--- Diagnostic Numbers (Constant 2500 RPM) ---")
    print(f"Mean Aortic Pressure (MAP) : {np.mean(x4_aop[mask]):.1f} mmHg")
    print(f"Mean Left Atrial Pressure  : {np.mean(x2_lap[mask]):.1f} mmHg")
    print(f"LVAD Cardiac Output        : {co_lvad:.2f} L/min")
    print(f"Native Cardiac Output      : {co_native:.2f} L/min")
    print(f"Total Cardiac Output       : {co_lvad + co_native:.2f} L/min")

    plt.show()

if __name__ == "__main__":
    simulate_lvad()

