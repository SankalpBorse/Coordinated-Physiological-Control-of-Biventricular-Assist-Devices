def compute_dp_dt(inflow, outflow, compliance):
    """
    Compliance Rule:
    The rate at which pressure changes in a chamber equals net flow divided by compliance.
    
    Parameters:
    - inflow: Flow entering the chamber (mL/s)
    - outflow: Flow leaving the chamber (mL/s)
    - compliance: Elasticity of the chamber (mL/mmHg)
    
    Returns:
    - dp_dt: Rate of change of pressure (mmHg/s)
    """
    if compliance <= 0:
        raise ValueError("Compliance must be greater than zero.")
    
    net_flow = inflow - outflow
    dp_dt = net_flow / compliance
    return dp_dt


def compute_flow_resistive(p_in, p_out, resistance, is_valve=False):
    """
    Resistance Rule (Algebraic):
    For a simple resistive branch with no inertia, flow equals pressure difference divided by resistance.
    If it's a valve, flow is forced to 0 if pressure is backward.
    
    Parameters:
    - p_in: Upstream pressure (mmHg)
    - p_out: Downstream pressure (mmHg)
    - resistance: Resistance to flow (mmHg*s/mL)
    - is_valve: Boolean, if True, acts as an ideal diode (no backflow)
    
    Returns:
    - flow: Fluid flow (mL/s)
    """
    if resistance <= 0:
        raise ValueError("Resistance must be greater than zero.")
    
    pressure_diff = p_in - p_out
    
    if is_valve and pressure_diff < 0:
        return 0.0
    
    flow = pressure_diff / resistance
    return flow


def compute_dq_dt(p_in, p_out, current_flow, resistance, inertance):
    """
    Inertance Rule (Differential):
    For branches where inertia matters, flow doesn't jump instantly. 
    The rate at which flow changes depends on the pressure difference and the resistive drop.
    
    Parameters:
    - p_in: Upstream pressure (mmHg)
    - p_out: Downstream pressure (mmHg)
    - current_flow: Current fluid flow (mL/s)
    - resistance: Resistance to flow (mmHg*s/mL)
    - inertance: Inertance of the fluid mass (mmHg*s^2/mL)
    
    Returns:
    - dq_dt: Rate of change of flow (mL/s^2)
    """
    if inertance <= 0:
        raise ValueError("Inertance must be greater than zero.")
    
    pressure_diff = p_in - p_out
    resistive_drop = current_flow * resistance
    
    # dQ/dt = (ΔP - R*Q) / L
    dq_dt = (pressure_diff - resistive_drop) / inertance
    return dq_dt


def validate_primitives():
    """
    Runs hand-calculable checks on the primitives as requested in Step 2.
    """
    print("--- Step 2 Validation: Passive Plumbing Primitives ---\n")
    
    # Check 1: Resistance Rule
    print("Check 1: Systemic Resistance (Rs = 1.0 mmHg*s/mL) with a 5 mmHg pressure difference.")
    Rs = 1.0
    dp = 5.0
    # Pin - Pout = 5.0, meaning Pin = 5.0 and Pout = 0.0
    flow = compute_flow_resistive(p_in=5.0, p_out=0.0, resistance=Rs)
    print(f"  Calculated Flow: {flow:.2f} mL/s")
    print(f"  Expected Flow  : 5.00 mL/s")
    if abs(flow - 5.0) < 1e-6:
        print("  -> STATUS: PASSED\n")
    else:
        print("  -> STATUS: FAILED\n")

    # Check 2: Compliance Rule
    print("Check 2: Left Atrial Compliance (CR = 4.4 mL/mmHg) with 10 mL/s net inflow for 1 second.")
    CR = 4.4
    inflow = 10.0
    outflow = 0.0
    duration_sec = 1.0
    
    # The rate of change of pressure
    dp_dt = compute_dp_dt(inflow=inflow, outflow=outflow, compliance=CR)
    # Since flow is constant, total pressure change over 1 second is just dp_dt * duration
    total_pressure_rise = dp_dt * duration_sec
    
    expected_rise = 10.0 / 4.4
    
    print(f"  Calculated Pressure Rise: {total_pressure_rise:.4f} mmHg")
    print(f"  Expected Pressure Rise  : {expected_rise:.4f} mmHg")
    if abs(total_pressure_rise - expected_rise) < 1e-6:
        print("  -> STATUS: PASSED\n")
    else:
        print("  -> STATUS: FAILED\n")
        
    # Extra Check: Inertance (for aortic branch)
    print("Check 3: Aortic Inertance (LS = 0.0005) acceleration check.")
    # If ΔP = 1 mmHg and current flow is 0, acceleration should be 1 / 0.0005 = 2000 mL/s^2
    LS = 0.0005
    accel = compute_dq_dt(p_in=1.0, p_out=0.0, current_flow=0.0, resistance=0.0, inertance=LS)
    expected_accel = 1.0 / 0.0005
    print(f"  Calculated dQ/dt: {accel:.2f} mL/s^2")
    print(f"  Expected dQ/dt  : {expected_accel:.2f} mL/s^2")
    if abs(accel - expected_accel) < 1e-6:
        print("  -> STATUS: PASSED\n")
    else:
        print("  -> STATUS: FAILED\n")


if __name__ == "__main__":
    validate_primitives()

