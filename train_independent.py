"""
Phase 5 — Independent Dual-Agent Deep Reinforcement Learning Training.
Implements Step 5.8: The Core Ablation (Centralized vs. Independent Agents)

Architecture:
  - Agent Left  (LVAD): Sees only [LAP, LVAD_Flow, AoP, V_LV, HR] (5 dims), Controls Delta_LVAD (1 dim)
  - Agent Right (RVAD): Sees only [RAP, RVAD_Flow, PAP, V_RV, HR] (5 dims), Controls Delta_RVAD (1 dim)
  - Both agents act on the same physically coupled BiVAD plant simultaneously without seeing each other's side.

Execution Modes:
  - --mode bc       : Pretrain both Left & Right actors via Behavioral Cloning
  - --mode eval_bc  : Evaluate both pretrained actors together on 200s Rest scenario
  - --mode trial    : Smoke-test trial run (~2000 steps)
  - --mode full     : Full independent dual-agent SAC training
"""
import os
import sys
import glob
import re
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv

from bivad_env import BiVADEnv


# =====================================================================
# Reward Normalization Parity with VecNormalize(norm_reward=True)
# =====================================================================
class RunningMeanStd:
    """Tracks running mean and variance (Welford's algorithm), matching SB3 VecNormalize."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RunningRewardNormalizer:
    """
    Normalizes rewards using running discounted return variance,
    strictly matching Stable-Baselines3 VecNormalize(norm_reward=True, gamma=0.99).
    This ensures fair and identical critic conditioning between centralized & independent arms.
    """
    def __init__(self, n_envs=4, gamma=0.99, epsilon=1e-8):
        self.n_envs = n_envs
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = np.zeros(n_envs, dtype=np.float32)
        self.ret_rms = RunningMeanStd(shape=())

    def normalize(self, rewards, dones):
        self.returns = self.returns * self.gamma + rewards
        self.ret_rms.update(self.returns)
        norm_rewards = rewards / np.sqrt(self.ret_rms.var + self.epsilon)
        # Reset discounted return accumulator for completed episodes
        self.returns[dones.astype(bool)] = 0.0
        return norm_rewards.astype(np.float32)

    def state_dict(self):
        return {
            'returns': self.returns.copy(),
            'mean': self.ret_rms.mean,
            'var': self.ret_rms.var,
            'count': self.ret_rms.count
        }

    def load_state_dict(self, d):
        self.returns = d['returns'].copy()
        self.ret_rms.mean = d['mean']
        self.ret_rms.var = d['var']
        self.ret_rms.count = d['count']


# Observation indices for the two thin observation views (Step 5.8)
# Raw obs indices: 0:LAP, 1:RAP, 2:LVAD_flow, 3:RVAD_flow, 4:AoP, 5:PAP, 6:V_LV, 7:V_RV, 8:HR
LEFT_OBS_INDICES  = [0, 2, 4, 6, 8]  # [LAP, LVAD_flow, AoP, V_LV, HR]
RIGHT_OBS_INDICES = [1, 3, 5, 7, 8]  # [RAP, RVAD_flow, PAP, V_RV, HR]


# =====================================================================
# Standalone SAC Components for Robust Dual-Agent Training
# =====================================================================
class Actor(nn.Module):
    """SAC Gaussian Actor Network with tanh squashing (net_arch=[256, 256])."""
    def __init__(self, obs_dim=5, act_dim=1, max_action=110.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.max_action = max_action

    def forward(self, obs, deterministic=False, with_logprob=True):
        feat = self.net(obs)
        mu = self.mu(feat)
        log_std = torch.clamp(self.log_std(feat), -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mu) * self.max_action
            log_prob = None
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()
            action = torch.tanh(u) * self.max_action
            if with_logprob:
                # Enforcing action bounds correction (Haarnoja et al., 2018)
                log_prob = dist.log_prob(u) - torch.log(self.max_action * (1 - torch.tanh(u).pow(2)) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-Networks for Double Q-Learning."""
    def __init__(self, obs_dim=5, act_dim=1):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    """Fixed-size circular buffer for transitions."""
    def __init__(self, capacity=300000, obs_dim=5, act_dim=1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, o, a, r, next_o, d):
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = next_o
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=256, device='cpu'):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.acts[idx], device=device),
            torch.tensor(self.rews[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.dones[idx], device=device)
        )


class SingleAgentSAC:
    """Independent SAC Agent."""
    def __init__(self, obs_dim=5, act_dim=1, lr=3e-4, gamma=0.99, tau=0.005, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if t_obs.ndim == 1:
                t_obs = t_obs.unsqueeze(0)
                act, _ = self.actor(t_obs, deterministic=deterministic)
                return act.cpu().numpy()[0]
            else:
                act, _ = self.actor(t_obs, deterministic=deterministic)
                return act.cpu().numpy()

    def update(self, buffer, batch_size=256):
        if buffer.size < batch_size:
            return 0.0, 0.0

        o, a, r, next_o, d = buffer.sample(batch_size, device=self.device)

        # 1. Update Critic
        with torch.no_grad():
            next_a, next_logprob = self.actor(next_o)
            q1_target, q2_target = self.critic_target(next_o, next_a)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_logprob
            target_q = r + (1.0 - d) * self.gamma * min_q_target

        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Update Actor
        pred_a, logprob = self.actor(o)
        q1_pi, q2_pi = self.critic(o, pred_a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logprob - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Update Alpha (Entropy)
        alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. Polyak averaging for target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()


# =====================================================================
# Behavioral Cloning for Independent Agents
# =====================================================================
def pretrain_independent_bc(agent_left, agent_right, expert_data_path='expert_data_pi.npz', epochs=8, device='cpu', save_dir="./checkpoints_independent"):
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert data file '{expert_data_path}' not found!")

    print(f"Loading expert data for independent agents from '{expert_data_path}'...")
    data = np.load(expert_data_path)
    obs = data['obs'].astype(np.float32)
    acts = data['acts'].astype(np.float32)

    # Slice observation views
    obs_l = obs[:, LEFT_OBS_INDICES]
    acts_l = acts[:, 0:1]

    obs_r = obs[:, RIGHT_OBS_INDICES]
    acts_r = acts[:, 1:2]

    loader_l = DataLoader(TensorDataset(torch.tensor(obs_l), torch.tensor(acts_l)), batch_size=256, shuffle=True)
    loader_r = DataLoader(TensorDataset(torch.tensor(obs_r), torch.tensor(acts_r)), batch_size=256, shuffle=True)

    opt_l = torch.optim.Adam(agent_left.actor.parameters(), lr=1e-3)
    opt_r = torch.optim.Adam(agent_right.actor.parameters(), lr=1e-3)

    print(f"\n--- Pretraining Left & Right Actors via Behavioral Cloning ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        loss_l_total, loss_r_total, nb = 0.0, 0.0, 0
        for (bl_o, bl_a), (br_o, br_a) in zip(loader_l, loader_r):
            bl_o, bl_a = bl_o.to(device), bl_a.to(device)
            br_o, br_a = br_o.to(device), br_a.to(device)

            # Left Actor
            opt_l.zero_grad()
            p_l, _ = agent_left.actor(bl_o, deterministic=True)
            loss_l = F.mse_loss(p_l, bl_a)
            loss_l.backward()
            opt_l.step()

            # Right Actor
            opt_r.zero_grad()
            p_r, _ = agent_right.actor(br_o, deterministic=True)
            loss_r = F.mse_loss(p_r, br_a)
            loss_r.backward()
            opt_r.step()

            loss_l_total += loss_l.item()
            loss_r_total += loss_r.item()
            nb += 1

        print(f"  Epoch [{epoch:2d}/{epochs:2d}] - Left Actor Loss: {loss_l_total/nb:8.4f} | Right Actor Loss: {loss_r_total/nb:8.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(agent_left.actor.state_dict(), os.path.join(save_dir, "actor_left_bc.pth"))
    torch.save(agent_right.actor.state_dict(), os.path.join(save_dir, "actor_right_bc.pth"))
    print(f"Pretraining complete. Saved checkpoints to {save_dir}/")


# =====================================================================
# Checkpoint Detection Helper
# =====================================================================
def find_latest_independent_checkpoint(checkpoint_dir="./checkpoints_independent"):
    """
    Scans for latest training step checkpoint for dual agents.
    Returns: (best_step, actor_l_path, actor_r_path, critic_l_path, critic_r_path, normalizer_path)
    """
    if not os.path.exists(checkpoint_dir):
        # Fallback to root BC
        bc_l = "./checkpoints_independent/actor_left_bc.pth"
        bc_r = "./checkpoints_independent/actor_right_bc.pth"
        if os.path.exists(bc_l) and os.path.exists(bc_r):
            return 0, bc_l, bc_r, None, None, None
        return 0, None, None, None, None, None

    matches = glob.glob(os.path.join(checkpoint_dir, "actor_l_step_*.pth"))
    if not matches:
        # Check for seed-specific BC pretrained fallback
        bc_l = os.path.join(checkpoint_dir, "actor_left_bc.pth")
        bc_r = os.path.join(checkpoint_dir, "actor_right_bc.pth")
        if os.path.exists(bc_l) and os.path.exists(bc_r):
            return 0, bc_l, bc_r, None, None, None
        # Root BC fallback
        bc_root_l = "./checkpoints_independent/actor_left_bc.pth"
        bc_root_r = "./checkpoints_independent/actor_right_bc.pth"
        if os.path.exists(bc_root_l) and os.path.exists(bc_root_r):
            return 0, bc_root_l, bc_root_r, None, None, None
        return 0, None, None, None, None, None

    best_step = -1
    for f in matches:
        m = re.search(r"actor_l_step_(\d+)\.pth", os.path.basename(f))
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step

    if best_step > 0:
        act_l = os.path.join(checkpoint_dir, f"actor_l_step_{best_step}.pth")
        act_r = os.path.join(checkpoint_dir, f"actor_r_step_{best_step}.pth")
        crt_l = os.path.join(checkpoint_dir, f"critic_l_step_{best_step}.pth")
        crt_r = os.path.join(checkpoint_dir, f"critic_r_step_{best_step}.pth")
        norm_p = os.path.join(checkpoint_dir, f"reward_normalizer_step_{best_step}.pkl")
        if not os.path.exists(norm_p):
            norm_p = None
        return best_step, act_l, act_r, crt_l, crt_r, norm_p

    return 0, None, None, None, None, None


# =====================================================================
# Dual-Agent Rest Evaluation
# =====================================================================
def evaluate_dual_agents_rest(agent_left, agent_right, desc="Pretrained Dual-Agent"):
    print(f"\n--- Evaluating {desc} on 200s Rest Scenario ---")
    env = BiVADEnv(randomize=False, raw_obs=False)
    obs, _ = env.reset()

    maps, laps, raps, cos, fbes, rewards = [], [], [], [], [], []
    lvad_rpms, rvad_rpms = [], []

    for _ in range(env.max_steps):
        o_l = obs[LEFT_OBS_INDICES]
        o_r = obs[RIGHT_OBS_INDICES]

        a_l = agent_left.select_action(o_l, deterministic=True)
        a_r = agent_right.select_action(o_r, deterministic=True)

        action = np.array([a_l[0], a_r[0]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        maps.append(info['MAP'])
        laps.append(env.state[2])
        raps.append(env.state[3])
        cos.append(info['CO_left'])
        fbes.append(info['flow_balance_error'])
        rewards.append(reward)
        lvad_rpms.append(info['LVAD_RPM'])
        rvad_rpms.append(info['RVAD_RPM'])

        if terminated or truncated:
            break

    print(f"  Mean MAP:               {np.mean(maps):.1f} mmHg (target: 80-120)")
    print(f"  Mean LAP:               {np.mean(laps):.1f} mmHg (target: ~9-12)")
    print(f"  Mean RAP:               {np.mean(raps):.1f} mmHg (target: ~4-6)")
    print(f"  Mean CO Left:           {np.mean(cos):.2f} L/min (target: 4.5-7.5)")
    print(f"  Mean Flow Balance Err:  {np.mean(fbes):.2f} mL/s")
    print(f"  LVAD Speed:             [{min(lvad_rpms):.0f} - {max(lvad_rpms):.0f}] RPM")
    print(f"  RVAD Speed:             [{min(rvad_rpms):.0f} - {max(rvad_rpms):.0f}] RPM")
    print(f"  Mean Step Reward:       {np.mean(rewards):.3f}")
    print(f"  Total Episode Reward:   {np.sum(rewards):.1f}")

    if np.mean(laps) < 3.0 or np.mean(raps) < 1.0:
        print("  [WARNING] Severe suction detected during Rest evaluation!")
    elif np.mean(laps) > 18.0:
        print("  [WARNING] Severe congestion detected during Rest evaluation!")
    else:
        print("  [PASS] Hemodynamics are stable and within safe operating limits.")

    return np.mean(fbes)


# =====================================================================
# Vectorized Environment Factory
# =====================================================================
def make_independent_vec_env(n_envs=4, base_seed=42):
    def _init(rank):
        def _thunk():
            env = BiVADEnv(randomize=True, raw_obs=False)
            env.reset(seed=base_seed + rank * 1000)
            return env
        return _thunk
    if n_envs == 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        return DummyVecEnv([_init(0)])
    return SubprocVecEnv([_init(i) for i in range(n_envs)])


# =====================================================================
# Independent Dual-Agent Training Loop with Auto-Resumability & Multi-Worker Parallelism
# =====================================================================
def train_independent_rl(agent_left, agent_right, total_steps=500000, start_step=0, n_envs=4, trial=False,
                         trial_steps=4000, seed=42, checkpoint_dir="./checkpoints_independent", tb_dir=None, norm_path=None):
    import time
    import shutil
    device = agent_left.device
    buf_l = ReplayBuffer(capacity=300000, obs_dim=5, act_dim=1)
    buf_r = ReplayBuffer(capacity=300000, obs_dim=5, act_dim=1)

    os.makedirs(checkpoint_dir, exist_ok=True)
    if tb_dir:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None

    target_steps = trial_steps if trial else total_steps
    steps_remaining = target_steps - start_step
    if steps_remaining <= 0:
        print(f"Target steps ({target_steps:,}) already achieved!")
        if writer:
            writer.close()
        return

    n_workers = min(n_envs, os.cpu_count() or 4)
    print(f"\n--- Launching Parallel Independent Dual-Agent SAC (Seed {seed} | {n_workers} CPU workers: {start_step:,} -> {target_steps:,} steps) ---")
    print(f"Checkpoints directory: {checkpoint_dir}")
    if tb_dir:
        print(f"TensorBoard directory: {tb_dir}")

    # Reward Normalization Parity with SB3 VecNormalize(norm_reward=True, gamma=0.99)
    reward_normalizer = RunningRewardNormalizer(n_envs=n_workers, gamma=0.99)
    if norm_path and os.path.exists(norm_path):
        try:
            with open(norm_path, 'rb') as f:
                reward_normalizer.load_state_dict(pickle.load(f))
            print(f"  [RESUME] Loaded RunningRewardNormalizer statistics from: {norm_path}")
        except Exception as e:
            print(f"  [WARNING] Could not load normalizer state ({e}), starting fresh normalizer.")

    vec_env = make_independent_vec_env(n_envs=n_workers, base_seed=seed)
    obs = vec_env.reset()  # shape (n_workers, 9)

    current_steps = start_step
    last_checkpoint_step = (start_step // 25000) * 25000
    t_start = time.time()
    t_report = time.time()
    steps_since_report = 0

    loss_crt_l, loss_act_l = 0.0, 0.0
    loss_crt_r, loss_act_r = 0.0, 0.0

    while current_steps < target_steps:
        o_l = obs[:, LEFT_OBS_INDICES]  # shape (n_workers, 5)
        o_r = obs[:, RIGHT_OBS_INDICES]  # shape (n_workers, 5)

        bc_check = os.path.join(checkpoint_dir, "actor_left_bc.pth")
        if current_steps < 2000 and not os.path.exists(bc_check) and not os.path.exists("./checkpoints_independent/actor_left_bc.pth") and start_step == 0:
            a_l = np.random.uniform(-110.0, 110.0, size=(n_workers, 1)).astype(np.float32)
            a_r = np.random.uniform(-110.0, 110.0, size=(n_workers, 1)).astype(np.float32)
        else:
            a_l = agent_left.select_action(o_l, deterministic=False)  # shape (n_workers, 1)
            a_r = agent_right.select_action(o_r, deterministic=False)  # shape (n_workers, 1)

        joint_act = np.hstack([a_l, a_r]).astype(np.float32)  # shape (n_workers, 2)
        next_obs, rewards, dones, infos = vec_env.step(joint_act)

        # Apply discounted return variance normalization
        norm_rewards = reward_normalizer.normalize(rewards, dones)

        next_o_l = next_obs[:, LEFT_OBS_INDICES]
        next_o_r = next_obs[:, RIGHT_OBS_INDICES]

        # Add transitions for all parallel workers
        for i in range(n_workers):
            buf_l.add(o_l[i], a_l[i], norm_rewards[i], next_o_l[i], float(dones[i]))
            buf_r.add(o_r[i], a_r[i], norm_rewards[i], next_o_r[i], float(dones[i]))

        obs = next_obs
        current_steps += n_workers
        steps_since_report += n_workers

        # Training updates
        if buf_l.size > 256:
            loss_crt_l, loss_act_l = agent_left.update(buf_l, batch_size=256)
            loss_crt_r, loss_act_r = agent_right.update(buf_r, batch_size=256)

        # Periodic logging every ~2,000 steps (~30s)
        if steps_since_report >= 2000 or current_steps >= target_steps:
            dt = time.time() - t_report
            fps = steps_since_report / max(dt, 1e-4)
            pct = 100.0 * current_steps / target_steps
            eta_min = (target_steps - current_steps) / max(fps * 60.0, 1e-4)

            mean_fbe = float(np.mean([infos[i].get('flow_balance_error', 0) for i in range(n_workers)]))
            mean_map = float(np.mean([infos[i].get('MAP', 0) for i in range(n_workers)]))
            mean_co = float(np.mean([infos[i].get('CO_left', 0) for i in range(n_workers)]))
            mean_r_raw = float(np.mean(rewards))
            mean_r_norm = float(np.mean(norm_rewards))

            mean_r_track = float(np.mean([infos[i].get('reward_track', 0) for i in range(n_workers)]))
            mean_r_bal = float(np.mean([infos[i].get('reward_balance', 0) for i in range(n_workers)]))
            mean_r_saf = float(np.mean([infos[i].get('reward_safety', 0) for i in range(n_workers)]))
            mean_r_sm = float(np.mean([infos[i].get('reward_smooth', 0) for i in range(n_workers)]))
            mean_r_map = float(np.mean([infos[i].get('reward_map', 0) for i in range(n_workers)]))

            print(f"  [{current_steps:6d}/{target_steps:6d} ({pct:4.1f}%)] - "
                  f"Speed: {fps:4.1f} steps/s | "
                  f"ETA: {eta_min:4.1f} min | "
                  f"FBE: {mean_fbe:4.1f} mL/s | "
                  f"MAP: {mean_map:4.1f} mmHg | "
                  f"Rew(raw): {mean_r_raw:5.2f} | Rew(norm): {mean_r_norm:5.2f}")

            # TensorBoard logging
            if writer:
                writer.add_scalar("loss/critic_left", loss_crt_l, current_steps)
                writer.add_scalar("loss/actor_left", loss_act_l, current_steps)
                writer.add_scalar("alpha/left", agent_left.alpha.item(), current_steps)
                writer.add_scalar("loss/critic_right", loss_crt_r, current_steps)
                writer.add_scalar("loss/actor_right", loss_act_r, current_steps)
                writer.add_scalar("alpha/right", agent_right.alpha.item(), current_steps)
                writer.add_scalar("reward/raw_step_mean", mean_r_raw, current_steps)
                writer.add_scalar("reward/norm_step_mean", mean_r_norm, current_steps)
                writer.add_scalar("reward/reward_track", mean_r_track, current_steps)
                writer.add_scalar("reward/reward_balance", mean_r_bal, current_steps)
                writer.add_scalar("reward/reward_safety", mean_r_saf, current_steps)
                writer.add_scalar("reward/reward_smooth", mean_r_sm, current_steps)
                writer.add_scalar("reward/reward_map", mean_r_map, current_steps)
                writer.add_scalar("clinical/MAP", mean_map, current_steps)
                writer.add_scalar("clinical/CO_left", mean_co, current_steps)
                writer.add_scalar("clinical/flow_balance_error", mean_fbe, current_steps)
                writer.add_scalar("speed/fps", fps, current_steps)

            t_report = time.time()
            steps_since_report = 0

        # Checkpoint every 100,000 steps
        if current_steps - last_checkpoint_step >= 100000:
            last_checkpoint_step = (current_steps // 100000) * 100000
            torch.save(agent_left.actor.state_dict(), os.path.join(checkpoint_dir, f"actor_l_step_{last_checkpoint_step}.pth"))
            torch.save(agent_right.actor.state_dict(), os.path.join(checkpoint_dir, f"actor_r_step_{last_checkpoint_step}.pth"))
            torch.save(agent_left.critic.state_dict(), os.path.join(checkpoint_dir, f"critic_l_step_{last_checkpoint_step}.pth"))
            torch.save(agent_right.critic.state_dict(), os.path.join(checkpoint_dir, f"critic_r_step_{last_checkpoint_step}.pth"))
            with open(os.path.join(checkpoint_dir, f"reward_normalizer_step_{last_checkpoint_step}.pkl"), "wb") as f:
                pickle.dump(reward_normalizer.state_dict(), f)
            print(f"  >>> [CHECKPOINT] Saved step {last_checkpoint_step:,} checkpoints & normalizer to {checkpoint_dir}/")

    vec_env.close()
    total_time_min = (time.time() - t_start) / 60.0
    print(f"\nIndependent Dual-Agent Training Complete for Seed {seed}! (Total time: {total_time_min:.1f} min)")
    torch.save(agent_left.actor.state_dict(), os.path.join(checkpoint_dir, "actor_l_final.pth"))
    torch.save(agent_right.actor.state_dict(), os.path.join(checkpoint_dir, "actor_r_final.pth"))
    torch.save(agent_left.critic.state_dict(), os.path.join(checkpoint_dir, "critic_l_final.pth"))
    torch.save(agent_right.critic.state_dict(), os.path.join(checkpoint_dir, "critic_r_final.pth"))
    with open(os.path.join(checkpoint_dir, "reward_normalizer_final.pkl"), "wb") as f:
        pickle.dump(reward_normalizer.state_dict(), f)
    print(f"Final weights saved to {checkpoint_dir}/actor_l_final.pth & actor_r_final.pth")

    if writer:
        writer.close()

    # Automatically generate publication plots and performance tables
    try:
        from post_training_evaluation import IndependentModelAdapter, generate_post_training_artifacts
        adapter = IndependentModelAdapter(agent_left, agent_right, name=f"SAC_Independent_Seed{seed}")
        generate_post_training_artifacts(adapter, output_dir=checkpoint_dir, tb_dir=tb_dir, seed=seed)
    except Exception as e:
        print(f"[WARNING] Post-training artifact generation encountered an error: {e}")

    # Auto-archive final zip archive containing models, tables, and plots
    try:
        shutil.make_archive(f"independent_sac_seed_{seed}", "zip", checkpoint_dir)
        print(f">>> [FINAL ZIP] Created independent_sac_seed_{seed}.zip (includes models, tables, and plots)")
    except Exception as e:
        print(f"[WARNING] Could not auto-archive final zip: {e}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Independent Dual-Agent Training")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["bc", "eval_bc", "trial", "full", "all"],
                        help="Execution mode: bc, eval_bc, trial, full, or all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--expert_data", type=str, default="expert_data_pi.npz")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--trial_steps", type=int, default=4000,
                        help="Timesteps for trial smoke test")
    parser.add_argument("--n_envs", type=int, default=os.cpu_count() or 4,
                        help=f"Number of parallel subprocess workers (auto-detected: {os.cpu_count() or 4})")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu for CPU-bound ODE solving)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Seed-isolated directories
    checkpoint_dir = f"./checkpoints_independent/seed_{args.seed}"
    tb_dir = f"./tb_independent/seed_{args.seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"Using compute device: {device} | Parallel workers: {args.n_envs} | Seed: {args.seed}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"TensorBoard directory: {tb_dir}")

    agent_l = SingleAgentSAC(obs_dim=5, act_dim=1, device=device)
    agent_r = SingleAgentSAC(obs_dim=5, act_dim=1, device=device)

    # 1. Behavioral Cloning Pretraining
    if args.mode in ["bc", "all"]:
        pretrain_independent_bc(agent_l, agent_r, expert_data_path=args.expert_data, device=device, save_dir=checkpoint_dir)
        evaluate_dual_agents_rest(agent_l, agent_r, desc=f"Pretrained Dual-Agent BC (Seed {args.seed})")

    elif args.mode == "eval_bc":
        bc_l = os.path.join(checkpoint_dir, "actor_left_bc.pth")
        bc_r = os.path.join(checkpoint_dir, "actor_right_bc.pth")
        if os.path.exists(bc_l) and os.path.exists(bc_r):
            agent_l.actor.load_state_dict(torch.load(bc_l, map_location=device))
            agent_r.actor.load_state_dict(torch.load(bc_r, map_location=device))
        elif os.path.exists("./checkpoints_independent/actor_left_bc.pth"):
            agent_l.actor.load_state_dict(torch.load("./checkpoints_independent/actor_left_bc.pth", map_location=device))
            agent_r.actor.load_state_dict(torch.load("./checkpoints_independent/actor_right_bc.pth", map_location=device))
        evaluate_dual_agents_rest(agent_l, agent_r, desc=f"Pretrained Dual-Agent BC (Seed {args.seed})")

    # 2. Trial Smoke Test Run
    if args.mode in ["trial"]:
        bc_l = os.path.join(checkpoint_dir, "actor_left_bc.pth")
        bc_r = os.path.join(checkpoint_dir, "actor_right_bc.pth")
        if os.path.exists(bc_l) and os.path.exists(bc_r):
            print(f"Loading warm-start BC weights from {checkpoint_dir} for trial run...")
            agent_l.actor.load_state_dict(torch.load(bc_l, map_location=device))
            agent_r.actor.load_state_dict(torch.load(bc_r, map_location=device))
        elif os.path.exists("./checkpoints_independent/actor_left_bc.pth"):
            print("Loading warm-start BC weights from root ./checkpoints_independent/ for trial run...")
            agent_l.actor.load_state_dict(torch.load("./checkpoints_independent/actor_left_bc.pth", map_location=device))
            agent_r.actor.load_state_dict(torch.load("./checkpoints_independent/actor_right_bc.pth", map_location=device))

        train_independent_rl(agent_l, agent_r, n_envs=args.n_envs, trial=True, trial_steps=args.trial_steps, seed=args.seed,
                             checkpoint_dir=checkpoint_dir, tb_dir=tb_dir)
        evaluate_dual_agents_rest(agent_l, agent_r, desc=f"Trial Independent Dual-Agent (Seed {args.seed})")

    # 3. Full Training Run with Resumability
    if args.mode in ["full", "all"]:
        best_step, act_l, act_r, crt_l, crt_r, norm_p = find_latest_independent_checkpoint(checkpoint_dir)
        if best_step > 0:
            print(f"\n[RESUME] Found existing dual-agent checkpoint at step {best_step:,}")
            agent_l.actor.load_state_dict(torch.load(act_l, map_location=device))
            agent_r.actor.load_state_dict(torch.load(act_r, map_location=device))
            if crt_l and os.path.exists(crt_l):
                agent_l.critic.load_state_dict(torch.load(crt_l, map_location=device))
                agent_l.critic_target.load_state_dict(torch.load(crt_l, map_location=device))
            if crt_r and os.path.exists(crt_r):
                agent_r.critic.load_state_dict(torch.load(crt_r, map_location=device))
                agent_r.critic_target.load_state_dict(torch.load(crt_r, map_location=device))

            if best_step >= args.steps:
                print(f"Target steps ({args.steps:,}) already achieved! Running evaluation.")
                evaluate_dual_agents_rest(agent_l, agent_r, desc=f"Final Independent Dual-Agent (Seed {args.seed})")
                return

            train_independent_rl(agent_l, agent_r, total_steps=args.steps, start_step=best_step, n_envs=args.n_envs,
                                 trial=False, seed=args.seed, checkpoint_dir=checkpoint_dir, tb_dir=tb_dir, norm_path=norm_p)
        elif act_l is not None and os.path.exists(act_l):
            print("\n[WARM-START] Loading BC pretrained actor weights...")
            agent_l.actor.load_state_dict(torch.load(act_l, map_location=device))
            agent_r.actor.load_state_dict(torch.load(act_r, map_location=device))
            train_independent_rl(agent_l, agent_r, total_steps=args.steps, start_step=0, n_envs=args.n_envs,
                                 trial=False, seed=args.seed, checkpoint_dir=checkpoint_dir, tb_dir=tb_dir)
        else:
            print(f"\n[FRESH] Starting full independent SAC training from scratch ({args.steps:,} steps, Seed {args.seed})")
            train_independent_rl(agent_l, agent_r, total_steps=args.steps, start_step=0, n_envs=args.n_envs,
                                 trial=False, seed=args.seed, checkpoint_dir=checkpoint_dir, tb_dir=tb_dir)

        # Final Rest scenario evaluation
        evaluate_dual_agents_rest(agent_l, agent_r, desc=f"Trained Independent Dual-Agent (Seed {args.seed})")


if __name__ == '__main__':
    main()

