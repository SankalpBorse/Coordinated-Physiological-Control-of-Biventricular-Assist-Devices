import numpy as np
from collections import deque
from .base_controller import Controller

class StarlingController(Controller):
    """
    Step 3.4 - Baseline 3: Dual Starling-like Controller
    Uses sigmoid mappings from beat-averaged atrial pressure to target pump speed.
    Filtered with 0.8s (8-step) moving window to prevent chattering on pulsatile swings.
    """
    def __init__(self, k_lap=0.5, k_rap=0.4):
        super().__init__()
        self.k_lap = k_lap
        self.k_rap = k_rap
        self.lap_history = deque(maxlen=8)
        self.rap_history = deque(maxlen=8)
        
    def reset(self):
        self.lap_history.clear()
        self.rap_history.clear()
        
    def get_action(self, observation, info=None):
        # Un-normalize observations to get true LAP and RAP
        lap_raw = observation[0] * 5.0 + 10.0
        rap_raw = observation[1] * 3.0 + 5.0
        
        # Append to history for beat-averaging
        self.lap_history.append(lap_raw)
        self.rap_history.append(rap_raw)
        
        lap = np.mean(self.lap_history)
        rap = np.mean(self.rap_history)
        
        lvad_rpm = info['LVAD_RPM'] if info else 2600.0
        rvad_rpm = info['RVAD_RPM'] if info else 1900.0
        
        # Sigmoid mappings to target RPM centered at physiological operating points:
        # LVAD resting target: 2600 RPM when LAP = 9.0 mmHg (operating range [2000, 3200] RPM)
        # RVAD resting target: 1900 RPM when RAP = 4.0 mmHg (operating range [1500, 2300] RPM)
        target_lvad = 2000.0 + 1200.0 / (1.0 + np.exp(-self.k_lap * (lap - 9.0)))
        target_rvad = 1500.0 + 800.0 / (1.0 + np.exp(-self.k_rap * (rap - 4.0)))
        
        # Calculate required delta and clip to slew limits
        delta_lvad = np.clip(target_lvad - lvad_rpm, -110.0, 110.0)
        delta_rvad = np.clip(target_rvad - rvad_rpm, -110.0, 110.0)
        
        return np.array([delta_lvad, delta_rvad], dtype=np.float32)

