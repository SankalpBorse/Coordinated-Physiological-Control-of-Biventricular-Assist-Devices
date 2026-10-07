import numpy as np
import copy
from collections import deque
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from .base_controller import Controller
from simulator.bivad_coupled_sim import bivad_derivatives

class MPCController(Controller):
    """
    Step 3.5 - Baseline 4: Centralized MPC (Koh 2020)
    Optimizes Delta RPM over a prediction horizon using internal ODE model.
    Uses 0.8s (8-step) moving-average window on atrial pressures to prevent
    chattering on intra-beat pulsatile swings, matching StarlingController and BiVADEnv.
    """
    def __init__(self, Hp=5, dt=0.1, k1=115.0):
        super().__init__()
        self.Hp = Hp
        self.dt = dt
        self.k1 = k1
        self.prev_u = np.array([0.0, 0.0], dtype=np.float32)
        self.lap_history = deque(maxlen=8)
        self.rap_history = deque(maxlen=8)
        self.lvad_flow_history = deque(maxlen=8)
        self.rvad_flow_history = deque(maxlen=8)
        
    def reset(self):
        self.prev_u = np.array([0.0, 0.0], dtype=np.float32)
        self.lap_history.clear()
        self.rap_history.clear()
        self.lvad_flow_history.clear()
        self.rvad_flow_history.clear()
        
    def get_action(self, observation, info=None):
        if info is None or 'true_state' not in info or 'env_params' not in info:
            return np.zeros(2, dtype=np.float32)
            
        y0 = info['true_state']
        current_lvad = info['LVAD_RPM']
        current_rvad = info['RVAD_RPM']
        base_params = info['env_params']
        
        # Track historical atrial pressures and pump flows for beat-averaging
        self.lap_history.append(y0[2])
        self.rap_history.append(y0[3])
        self.lvad_flow_history.append(y0[10])
        self.rvad_flow_history.append(y0[11])
        
        def cost_function(u):
            dlvad, drvad = u
            cost = 0.0
            
            sim_params = copy.copy(base_params)
            sim_params['LVAD_RPM'] = np.clip(current_lvad + dlvad, 1800.0, 3800.0)
            sim_params['RVAD_RPM'] = np.clip(current_rvad + drvad, 1800.0, 3800.0)
            
            y_curr = y0
            t_curr = 0.0
            
            # Forward simulation history for rolling beat-average over prediction horizon
            pred_lap_hist = list(self.lap_history)
            pred_rap_hist = list(self.rap_history)
            pred_lvad_hist = list(self.lvad_flow_history)
            pred_rvad_hist = list(self.rvad_flow_history)
            
            # Fast vectorized integration across full prediction horizon (0.0 to Hp*dt)
            t_eval = [(i + 1) * self.dt for i in range(self.Hp)]
            sol = solve_ivp(
                fun=lambda t, y: bivad_derivatives(t, y, sim_params),
                t_span=(0.0, self.Hp * self.dt),
                y0=y0,
                method='RK23',
                t_eval=t_eval
            )
            
            if not sol.success or sol.y.shape[1] < self.Hp:
                return 1e6
                
            for i in range(self.Hp):
                y_curr = sol.y[:, i]
                lap_inst, rap_inst = y_curr[2], y_curr[3]
                lvad_flow, rvad_flow = y_curr[10], y_curr[11]
                
                pred_lap_hist.append(lap_inst)
                pred_rap_hist.append(rap_inst)
                pred_lvad_hist.append(lvad_flow)
                pred_rvad_hist.append(rvad_flow)
                
                # Beat-averaged atrial pressure and flow over 0.8s (8-step) window (Koh 2020)
                lap_mean = np.mean(pred_lap_hist[-8:])
                rap_mean = np.mean(pred_rap_hist[-8:])
                lvad_mean = np.mean(pred_lvad_hist[-8:])
                rvad_mean = np.mean(pred_rvad_hist[-8:])
                
                # Koh 2020: Starling flow reference from beat-averaged atrial pressure
                q_l_ref = self.k1 / (1.0 + np.exp(-0.5 * (lap_mean - 9.0)))
                q_r_ref = self.k1 / (1.0 + np.exp(-0.5 * (rap_mean - 4.0)))
                
                # Flow tracking error on beat-averaged flow (Koh 2020 cycle-mean tracking)
                cost += (lvad_mean - q_l_ref)**2
                cost += (rvad_mean - q_r_ref)**2
                
                # Marginal suction and congestion safety terms using beat-averaged pressure
                if lap_mean < 3.0:
                    cost += 500.0 * (3.0 - lap_mean)**2
                elif lap_mean > 18.0:
                    cost += 500.0 * (lap_mean - 18.0)**2
                    
                if rap_mean < 3.0:
                    cost += 500.0 * (3.0 - rap_mean)**2
                elif rap_mean > 18.0:
                    cost += 500.0 * (rap_mean - 18.0)**2
                    
            # Move suppression: Koh 2020 R = diag(2000, 2000)
            du_l = dlvad / 1000.0
            du_r = drvad / 1000.0
            cost += 2000.0 * (du_l**2) + 2000.0 * (du_r**2)
            
            return cost
            
        # Baseline cost at zero action u = [0, 0]
        c_zero = cost_function(np.zeros(2, dtype=np.float32))

        res = minimize(
            cost_function,
            x0=np.zeros(2, dtype=np.float32),
            bounds=[(-110.0, 110.0), (-110.0, 110.0)],
            method='L-BFGS-B',
            options={'maxiter': 5, 'ftol': 1e-1}
        )
        
        # If optimization does not significantly improve over zero action, or if micro-jitter < 5 RPM, stay at zero
        if res.fun >= c_zero - 0.1 or np.all(np.abs(res.x) < 5.0):
            action = np.zeros(2, dtype=np.float32)
        else:
            action = np.array(res.x, dtype=np.float32)

        self.prev_u = action
        return action
