import numpy as np
from .base_controller import Controller

from collections import deque

class PIDecentralizedController(Controller):
    """
    Step 3.3 - Baseline 2: Decentralized PI (atrial pressure tracking)
    Uses two independent PI loops to track LAP=9 and RAP=4.
    """
    def __init__(self, kp_left=25.0, ki_left=5.0, kp_right=30.0, ki_right=3.0, dt=0.1):
        super().__init__()
        self.kp_left = kp_left
        self.ki_left = ki_left
        self.kp_right = kp_right
        self.ki_right = ki_right
        
        self.dt = dt
        
        self.target_lap = 9.0
        self.target_rap = 4.0
        
        self.prev_error_left = 0.0
        self.prev_error_right = 0.0
        
        # 0.8s rolling window (8 steps at dt=0.1) to smooth out pulsatile heartbeat
        self.lap_history = deque(maxlen=8)
        self.rap_history = deque(maxlen=8)
        
    def reset(self):
        self.prev_error_left = 0.0
        self.prev_error_right = 0.0
        self.lap_history.clear()
        self.rap_history.clear()

    def get_action(self, observation, info=None):
        # Un-normalize observations to get true LAP and RAP
        lap_raw = observation[0] * 5.0 + 10.0
        rap_raw = observation[1] * 3.0 + 5.0
        
        # Add to history and get smoothed values
        self.lap_history.append(lap_raw)
        self.rap_history.append(rap_raw)
        
        lap = np.mean(self.lap_history)
        rap = np.mean(self.rap_history)
        
        # error = measured - target
        error_left = lap - self.target_lap
        error_right = rap - self.target_rap
        
        # Velocity form of PI control:
        # Since the environment accumulates the delta (RPM += action),
        # outputting Kp*e + Ki*int(e) would create a double-integrator (PII).
        # We must output the derivative of the PI equation:
        # Delta = Kp * (e_t - e_{t-1}) + Ki * e_t * dt
        
        delta_lvad = self.kp_left * (error_left - self.prev_error_left) + self.ki_left * error_left * self.dt
        delta_rvad = self.kp_right * (error_right - self.prev_error_right) + self.ki_right * error_right * self.dt
        
        self.prev_error_left = error_left
        self.prev_error_right = error_right
        
        # The env action space is bounded at [-110, 110]. We enforce the slew limit here.
        delta_lvad = np.clip(delta_lvad, -110.0, 110.0)
        delta_rvad = np.clip(delta_rvad, -110.0, 110.0)
        
        return np.array([delta_lvad, delta_rvad], dtype=np.float32)

