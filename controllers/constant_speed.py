import numpy as np
from .base_controller import Controller

class ConstantSpeedController(Controller):
    """
    Step 3.2 - Baseline 1: Constant Speed
    Fixed at validated operating point, always returns [0, 0] delta RPM.
    """
    def __init__(self):
        super().__init__()
        
    def get_action(self, observation, info=None):
        # Action space is Delta RPM. 
        # A constant speed controller requests 0 change in RPM.
        return np.array([0.0, 0.0], dtype=np.float32)
