import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from scipy.integrate import solve_ivp
from collections import deque

# Import the simulator functions and base parameters
from simulator.bivad_coupled_sim import bivad_derivatives, P as BASE_P
from simulator.ventricular_elastance import time_varying_elastance
from reward import compute_reward

class BiVADEnv(gym.Env):
    """
    BiVAD Cardiovascular Environment for Deep Reinforcement Learning.
    Follows Phase 2 guidelines.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, randomize=True, raw_obs=False):
        super(BiVADEnv, self).__init__()
        
        self.randomize = randomize
        self.raw_obs = raw_obs
        self.dt = 0.1  # Control interval: 0.1s
        self.max_steps = 2000  # 200s / 0.1s (Step 2.10)
        self.current_step = 0
        self.current_time = 0.0
        
        # State tracking
        self.state = None
        self.params = None
        self.baseline_params = None
        
        # Moving average buffers (0.8s = 8 steps at dt=0.1s)
        self.pressure_history = None
        self.flow_history = None
        
        # Perturbation schedule (Step 2.8)
        self.perturbation_schedule = {}
        
        # Normalization constants (Step 2.9)
        self.obs_mean = np.array([10.0, 5.0, 90.0, 90.0, 90.0, 25.0, 130.0, 130.0, 80.0], dtype=np.float32)
        self.obs_std = np.array([5.0, 3.0, 30.0, 30.0, 20.0, 10.0, 40.0, 40.0, 15.0], dtype=np.float32)

        # --- Observation Space ---
        bound = np.full(9, 500.0 if self.raw_obs else 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float32)
        
        # --- Action Space ---
        # [Delta LVAD RPM, Delta RVAD RPM]
        self.action_space = spaces.Box(low=-110.0, high=110.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # 1. Initialize parameters
        self.params = copy.deepcopy(BASE_P)
        self.perturbation_schedule = {}
        
        if self.randomize:
            self.params['R_sys'] = self.np_random.uniform(0.6, 1.4)
            self.params['R_pulm'] = self.np_random.uniform(0.05, 0.40)
            self.params['HR'] = self.np_random.uniform(60, 120)
            self.params['E_max_lv'] = self.np_random.uniform(0.5, 1.2)
            self.params['E_max_rv'] = self.np_random.uniform(0.15, 0.50)
            self.params['C_la'] = self.np_random.uniform(3.5, 5.5)
            self.params['C_ra'] = self.np_random.uniform(3.5, 5.5)
            
            lvad_base = 2600.0 + self.np_random.uniform(-100, 100)
            rvad_base = 1900.0 + self.np_random.uniform(-100, 100)
            
            # Setup mid-episode perturbation (Bug 2.6: non-compounding recovery)
            p_step1 = self.np_random.integers(300, 800)
            p_type1 = self.np_random.choice(['SVR_up', 'SVR_down', 'PVR_up', 'HR_up', 'HR_down', 'Combined'])
            self.perturbation_schedule[p_step1] = p_type1
            
            # Optional second perturbation (recovery back to baseline)
            if self.np_random.random() < 0.5:
                p_step2 = self.np_random.integers(1200, 1700)
                self.perturbation_schedule[p_step2] = 'Recovery'
        else:
            # Optionally set fixed non-randomized test values if requested by options
            if options and 'SVR_mult' in options:
                self.params['R_sys'] *= options['SVR_mult']
            if options and 'force_perturb' in options:
                p_step, p_type = options['force_perturb']
                self.perturbation_schedule[p_step] = p_type
            
            lvad_base = 2600.0
            rvad_base = 1900.0
            
        self.params['LVAD_RPM'] = lvad_base
        self.params['RVAD_RPM'] = rvad_base
        
        # Save pristine baseline parameters (Bug 2.6)
        self.baseline_params = copy.deepcopy(self.params)
        
        # 2. Warm-up
        y0_generic = [140.0, 140.0, 10.0, 5.0, 80.0, 20.0, 75.0, 12.0, 80.0, 80.0, 80.0, 80.0]
        if seed is not None and not self.randomize:
            rng = np.random.default_rng(seed)
            perturbation = rng.uniform(0.98, 1.02, size=len(y0_generic))
            y0_generic = [float(v * p) for v, p in zip(y0_generic, perturbation)]
        elif options and 'y0' in options:
            y0_generic = options['y0']
            
        t_eval_warmup = np.linspace(20.0 - 7 * self.dt, 20.0, 8)
        
        try:
            sol = solve_ivp(
                fun=lambda t, y: bivad_derivatives(t, y, self.params),
                t_span=(0, 20.0),
                y0=y0_generic,
                method='Radau',
                t_eval=t_eval_warmup,
                max_step=0.005
            )
            warmup_success = sol.success
        except Exception:
            warmup_success = False
            sol = None
        
        tc = 60.0 / self.params['HR']
        self.current_time = 20.0 % tc
        
        if not warmup_success or sol is None:
            print("WARNING: Warm-up failed!")
            self.state = np.array(y0_generic, dtype=np.float32)
            V_lv, V_rv, lap, rap, aop, pap, sysp, pulmp, aoflow, paflow, lvad_flow, rvad_flow = self.state
            E_lv = time_varying_elastance(self.current_time, hr=self.params['HR'], E_max=self.params['E_max_lv'], E_min=self.params['E_min_lv'])
            E_rv = time_varying_elastance(self.current_time, hr=self.params['HR'], E_max=self.params['E_max_rv'], E_min=self.params['E_min_rv'])
            lvp = float(V_lv * E_lv)
            rvp = float(V_rv * E_rv)
            
            self.pressure_history = {
                'lap': deque([float(lap)] * 8, maxlen=8),
                'rap': deque([float(rap)] * 8, maxlen=8),
                'lvp': deque([lvp] * 8, maxlen=8),
                'rvp': deque([rvp] * 8, maxlen=8)
            }
            self.flow_history = {
                'lvad': deque([float(lvad_flow)] * 8, maxlen=8),
                'rvad': deque([float(rvad_flow)] * 8, maxlen=8)
            }
        else:
            self.state = sol.y[:, -1]
            
            # Bug 2.4: Pre-fill moving-average buffers with the 8 warm-up trajectory points
            times = t_eval_warmup % tc
            e_lv_arr = [time_varying_elastance(t, hr=self.params['HR'], E_max=self.params['E_max_lv'], E_min=self.params['E_min_lv']) for t in times]
            e_rv_arr = [time_varying_elastance(t, hr=self.params['HR'], E_max=self.params['E_max_rv'], E_min=self.params['E_min_rv']) for t in times]
            lvp_arr = sol.y[0, :] * e_lv_arr
            rvp_arr = sol.y[1, :] * e_rv_arr
            
            self.pressure_history = {
                'lap': deque([float(x) for x in sol.y[2, :]], maxlen=8),
                'rap': deque([float(x) for x in sol.y[3, :]], maxlen=8),
                'lvp': deque([float(x) for x in lvp_arr], maxlen=8),
                'rvp': deque([float(x) for x in rvp_arr], maxlen=8)
            }
            self.flow_history = {
                'lvad': deque([float(x) for x in sol.y[10, :]], maxlen=8),
                'rvad': deque([float(x) for x in sol.y[11, :]], maxlen=8)
            }
        reset_info = {
            'LVAD_RPM': self.params['LVAD_RPM'],
            'RVAD_RPM': self.params['RVAD_RPM'],
            'true_state': np.copy(self.state),
            'env_params': copy.deepcopy(self.params),
            'MAP': float(self.state[4]),
            'CO_left': float(self.state[10] * 60.0 / 1000.0),
            'CO_right': float(self.state[11] * 60.0 / 1000.0),
            'flow_balance_error': abs(float(self.state[10] - self.state[11])),
            'suction_left': False,
            'suction_right': False,
        }
        return self._get_obs(), reset_info

    def step(self, action):
        # 0. Check and apply perturbations (Step 2.8 / Bug 2.6)
        if self.current_step in self.perturbation_schedule:
            p_type = self.perturbation_schedule[self.current_step]
            if p_type == 'Recovery':
                # Restore cardiovascular parameters to pristine baseline
                for key in ['R_sys', 'R_pulm', 'HR', 'E_max_lv', 'E_max_rv', 'C_la', 'C_ra']:
                    if key in self.baseline_params:
                        self.params[key] = self.baseline_params[key]
            elif p_type == 'SVR_up':
                self.params['R_sys'] *= self.np_random.uniform(1.3, 1.5)
            elif p_type == 'SVR_down':
                self.params['R_sys'] *= self.np_random.uniform(0.5, 0.7)
            elif p_type == 'PVR_up':
                self.params['R_pulm'] *= self.np_random.uniform(1.5, 3.0)
            elif p_type == 'HR_up':
                self.params['HR'] = self.np_random.uniform(100.0, 120.0)
            elif p_type == 'HR_down':
                self.params['HR'] = self.np_random.uniform(50.0, 60.0)
            elif p_type == 'Combined':
                self.params['R_sys'] *= self.np_random.uniform(1.3, 1.5)
                self.params['HR'] = self.np_random.uniform(100.0, 120.0)
                
        # 1. Apply Action with hardware clamping [1800, 3800] RPM (Bug 2.7)
        old_lvad = self.params['LVAD_RPM']
        old_rvad = self.params['RVAD_RPM']
        
        self.params['LVAD_RPM'] = float(np.clip(old_lvad + action[0], 1800.0, 3800.0))
        self.params['RVAD_RPM'] = float(np.clip(old_rvad + action[1], 1800.0, 3800.0))
        actual_action = [self.params['LVAD_RPM'] - old_lvad, self.params['RVAD_RPM'] - old_rvad]
        
        # 2. Integrate
        t_span = (self.current_time, self.current_time + self.dt)
        try:
            sol = solve_ivp(
                fun=lambda t, y: bivad_derivatives(t, y, self.params),
                t_span=t_span,
                y0=self.state,
                method='Radau',
                max_step=0.005
            )
            success = sol.success
        except Exception:
            success = False
            sol = None
        
        # 3. Solver Failure Handling (Bug 2.1: strictly terminated for solver crash)
        terminated = False
        truncated = False
        
        if not success:
            reward = -100.0
            terminated = True
            reward_comp = {
                'r_track': 0.0, 'r_balance': 0.0, 'r_safety': -100.0, 'r_smooth': 0.0, 'r_map': 0.0,
                'tracking_error_raw': 0.0, 'flow_imbalance_raw': 0.0, 'smooth_raw': 0.0
            }
            mean_lap, mean_rap = 0.0, 0.0
            lvp, rvp = 0.0, 0.0
            aop = 0.0
            mean_lvad_flow, mean_rvad_flow = 0.0, 0.0
        else:
            self.state = sol.y[:, -1]
            
            # 4. Advance Time
            self.current_time += self.dt
            self.current_step += 1
            
            y = self.state
            V_lv, V_rv, lap, rap, aop, pap, sysp, pulmp, aoflow, paflow, lvad_flow, rvad_flow = y
            
            E_lv = time_varying_elastance(self.current_time, hr=self.params['HR'], E_max=self.params['E_max_lv'], E_min=self.params['E_min_lv'])
            E_rv = time_varying_elastance(self.current_time, hr=self.params['HR'], E_max=self.params['E_max_rv'], E_min=self.params['E_min_rv'])
            lvp = float(V_lv * E_lv)
            rvp = float(V_rv * E_rv)
            
            # Append to history for smoothing
            self.pressure_history['lap'].append(float(lap))
            self.pressure_history['rap'].append(float(rap))
            self.pressure_history['lvp'].append(lvp)
            self.pressure_history['rvp'].append(rvp)
            
            # Bug 2.3: 8-step moving average for pump flow
            self.flow_history['lvad'].append(float(lvad_flow))
            self.flow_history['rvad'].append(float(rvad_flow))
            
            mean_lap = float(np.mean(self.pressure_history['lap']))
            mean_rap = float(np.mean(self.pressure_history['rap']))
            mean_lvad_flow = float(np.mean(self.flow_history['lvad']))
            mean_rvad_flow = float(np.mean(self.flow_history['rvad']))
            
            # 5. Compute Real Multi-Objective Reward
            # Bug 2.2: Pass instantaneous lvp, rvp
            # Bug 2.3: Pass beat-averaged mean_lvad_flow, mean_rvad_flow
            # Bug 2.7: Pass actual_action instead of requested action
            reward, reward_comp = compute_reward(
                mean_lap, mean_rap, lvp, rvp, 
                mean_lvad_flow, mean_rvad_flow, 
                actual_action, 
                map_value=aop
            )
            
            # 6. Bug 2.1: Time limit is truncated, NOT terminated!
            if self.current_step >= self.max_steps:
                truncated = True
                
        obs = self._get_obs()
        
        info = {
            'LVAD_RPM': self.params['LVAD_RPM'], 
            'RVAD_RPM': self.params['RVAD_RPM'],
            'actual_action': actual_action if (sol is not None and sol.success) else [0.0, 0.0],
            'MAP': aop, 
            'CO_left': mean_lvad_flow * 60.0 / 1000.0, 
            'CO_right': mean_rvad_flow * 60.0 / 1000.0,
            'flow_balance_error': abs(mean_lvad_flow - mean_rvad_flow),
            'suction_left': bool(lvp < 0), 
            'suction_right': bool(rvp < 0),
            'congestion_left': bool(mean_lap > 18.0), 
            'congestion_right': bool(mean_rap > 18.0),
            'reward_track': reward_comp['r_track'],
            'reward_balance': reward_comp['r_balance'],
            'reward_safety': reward_comp['r_safety'],
            'reward_smooth': reward_comp['r_smooth'],
            'reward_map': reward_comp['r_map'],
            'r_map': reward_comp['r_map'],
            'tracking_error_raw': reward_comp['tracking_error_raw'],
            'flow_imbalance_raw': reward_comp['flow_imbalance_raw'],
            'smooth_raw': reward_comp['smooth_raw'],
            'true_state': np.copy(self.state),
            'env_params': copy.deepcopy(self.params)
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # State vector: [V_LV, V_RV, LAP, RAP, AoP, PAP, SysP, PulmP, AoFlow, PAFlow, LVAD_Flow, RVAD_Flow]
        y = self.state
        
        # Raw observations
        raw_obs = np.array([
            y[2],            # 0: LAP
            y[3],            # 1: RAP
            y[10],           # 2: LVAD flow
            y[11],           # 3: RVAD flow
            y[4],            # 4: AoP
            y[5],            # 5: PAP
            y[0],            # 6: V_LV
            y[1],            # 7: V_RV
            self.params['HR'] # 8: HR
        ], dtype=np.float32)
        
        if self.raw_obs:
            return raw_obs
            
        # Fixed normalization (Step 2.9)
        return (raw_obs - self.obs_mean) / self.obs_std


if __name__ == '__main__':
    print("--- Phase 2 Environment Validations (Steps 2.8 - 2.13) ---")
    
    # 2.8: Mid-episode perturbations
    env = BiVADEnv(randomize=False)
    env.reset(options={'force_perturb': (50, 'SVR_up')})
    print("\n[Step 2.8] Perturbation test (SVR_up at step 50):")
    print(f"Pre-perturb R_sys: {env.params['R_sys']:.2f}")
    
    # Run 49 steps
    for _ in range(49):
        env.step([0, 0])
    print(f"Step 49 R_sys: {env.params['R_sys']:.2f}, AoP: {env.state[4]:.1f}")
    
    # Step 50: Perturbation should trigger
    env.step([0, 0])
    print(f"Step 50 R_sys: {env.params['R_sys']:.2f}, AoP: {env.state[4]:.1f}")
    
    # Run a few more steps to see AoP climb
    for _ in range(10):
        env.step([0, 0])
    print(f"Step 60 R_sys: {env.params['R_sys']:.2f}, AoP: {env.state[4]:.1f} (Expected to climb)")
    
    # 2.9: Observation normalization
    print("\n[Step 2.9] Checking observation normalization...")
    env = BiVADEnv(randomize=True)
    obs, _ = env.reset()
    all_obs = [obs]
    for _ in range(100):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        all_obs.append(obs)
    all_obs = np.array(all_obs)
    max_val = np.max(all_obs)
    min_val = np.min(all_obs)
    print(f"Max obs value across 100 steps: {max_val:.2f} (Expected ~ < 4.0)")
    print(f"Min obs value across 100 steps: {min_val:.2f} (Expected ~ > -4.0)")
    
    # 2.10: Episode structure
    print("\n[Step 2.10] Episode structure...")
    env = BiVADEnv(randomize=False)
    env.reset()
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step([0, 0])
        steps += 1
    print(f"Episode ended at step: {steps} (Expected 2000), truncated={truncated}, terminated={terminated}")
    print(f"Current time at end: {env.current_time:.1f}s")
    
    # 2.11: Info dict
    print("\n[Step 2.11] Info dict check...")
    env.reset()
    _, _, _, _, info = env.step([0, 0])
    print("Sample Info Dict:")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
            
    # 2.12: Registration and env checker
    print("\n[Step 2.12] Gym registration & env checker...")
    import gymnasium
    from stable_baselines3.common.env_checker import check_env
    gymnasium.register(id='BiVAD-v0', entry_point='__main__:BiVADEnv')
    check_env(BiVADEnv(), warn=True)
    print("Env checker passed with no critical errors!")
    
    # 2.13: Smoke test
    print("\n[Step 2.13] SAC Smoke Test...")
    from stable_baselines3 import SAC
    import warnings
    # Suppress warnings to keep output clean for validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SAC("MlpPolicy", BiVADEnv(randomize=True), verbose=0)
        print("Starting SAC training for 1000 timesteps...")
        model.learn(total_timesteps=1000)
    print("Smoke test complete! No crashes.")
    print("\nPhase 2 fully implemented. Ready for the user to run.")

