import numpy as np
import matplotlib.pyplot as plt

def time_varying_elastance(t, hr=75.0, E_max=2.0, E_min=0.06):
    """
    Calculates the time-varying elastance of the left ventricle E(t) 
    using the standard double-Hill equation (Stergiopulos et al., 1996).
    
    Parameters:
    - t: time array or scalar (seconds)
    - hr: heart rate (beats per minute)
    - E_max: maximum elastance (end-systolic, mmHg/mL)
    - E_min: minimum elastance (end-diastolic, mmHg/mL)
    
    Returns:
    - E_t: elastance at time t (mmHg/mL)
    """
    
    # Cardiac cycle duration (seconds)
    tc = 60.0 / hr 
    
    # Time to maximum elastance (seconds). 
    # This is a common empirical relation to cardiac cycle length.
    t_max = 0.2 + 0.15 * tc 
    
    # Make time periodic over the cardiac cycle
    t_cycle = t % tc
    
    # Normalized time
    t_n = t_cycle / t_max 
    
    # Double-Hill function parameters (Stergiopulos et al., 1996)
    # The factor 1.55 scales the peak of the normalized curve to exactly 1.0
    n1, n2 = 1.9, 21.9
    tau1, tau2 = 0.7, 1.17
    
    # Compute normalized elastance E_n(t_n)
    term1 = (t_n / tau1)**n1 / (1.0 + (t_n / tau1)**n1)
    term2 = 1.0 / (1.0 + (t_n / tau2)**n2)
    E_n = 1.55 * term1 * term2
    
    # Scale to physiological E_max and E_min
    E_t = E_min + (E_max - E_min) * E_n
    
    return E_t

def validate_elastance():
    """
    Runs the heartbeat engine open-loop and plots it for validation.
    """
    # Test at baseline heart rate (75 bpm)
    hr = 75.0 
    
    # Create a time vector for 3 complete cardiac cycles
    tc = 60.0 / hr
    total_time = 3 * tc
    
    # Sample at 1 ms resolution
    t = np.linspace(0, total_time, 1000)
    
    # Compute elastance
    E = time_varying_elastance(t, hr=hr)
    
    # Left ventricular compliance is the inverse of elastance (C = 1/E)
    C = 1.0 / E
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Elastance E(t)
    ax1.plot(t, E, color='red', linewidth=2, label='Elastance E(t)')
    ax1.set_ylabel('Elastance (mmHg/mL)', fontsize=12)
    ax1.set_title('Step 1 Validation: Time-Varying Ventricular Elastance & Compliance (HR = 75 bpm)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot Compliance C(t)
    ax2.plot(t, C, color='blue', linewidth=2, label='Compliance C(t) = 1/E(t)')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Compliance (mL/mmHg)', fontsize=12)
    ax2.set_yscale('log') # Log scale helps visualize compliance since it varies widely
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('heartbeat_engine_validation.png', dpi=300)
    print("Validation successful. Plot saved to 'heartbeat_engine_validation.png'.")
    plt.show()

if __name__ == "__main__":
    validate_elastance()

