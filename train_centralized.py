"""
Phase 5 — Centralized Deep Reinforcement Learning Training (SAC).
Implements Steps 5.1 - 5.7:
  - Step 5.1: Soft Actor-Critic (SAC) via Stable-Baselines3
  - Step 5.2: MLP policy architecture (net_arch=[256, 256])
  - Step 5.3: Parallel SubprocVecEnv + VecNormalize handoff
  - Step 5.4: Imitation Warm-start via Behavioral Cloning (pretraining actor)
  - Step 5.5: Domain randomization across virtual patients
  - Step 5.6: Training budget management & CPU-optimized multi-worker execution
  - Step 5.7: CheckpointCallback, Resumability, TensorBoard logging
"""
import os
import sys
import re
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from bivad_env import BiVADEnv


# =====================================================================
# Reward Component & Clinical Metrics TensorBoard Callback
# =====================================================================
class RewardComponentCallback(BaseCallback):
    """
    Extracts multi-objective reward subcomponents and hemodynamic indicators
    from vectorized environment step infos and logs them to TensorBoard.
    """
    def __init__(self, log_freq=500, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.comp_keys = [
            'reward_track', 'reward_balance', 'reward_safety', 'reward_smooth', 'reward_map',
            'flow_balance_error', 'MAP', 'CO_left', 'CO_right', 'LVAD_RPM', 'RVAD_RPM'
        ]
        self.sums = {k: 0.0 for k in self.comp_keys}
        self.count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for k in self.comp_keys:
                self.sums[k] += float(info.get(k, 0.0))
            self.count += 1

        if self.count >= self.log_freq:
            for k in self.comp_keys:
                mean_val = self.sums[k] / max(self.count, 1)
                prefix = "reward" if k.startswith("reward_") else "clinical"
                tag = f"{prefix}/{k}"
                self.logger.record(tag, mean_val)
                self.sums[k] = 0.0
            self.count = 0
        return True


# =====================================================================
# Step 5.4 — Behavioral Cloning Warm-Start (Pretrain SAC Actor)
# =====================================================================
def pretrain_actor_bc(model, expert_data_path='expert_data_pi.npz', epochs=8, batch_size=256, lr=1e-3, device='cpu', save_dir="./checkpoints"):
    """
    Supervised regression pretraining of the SAC actor network using expert transitions.
    Prevents random catastrophic exploration early in training.
    """
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert dataset '{expert_data_path}' not found! "
                                f"Please make sure expert_data_pi.npz is in the current working directory.")

    print(f"Loading expert data from '{expert_data_path}'...")
    data = np.load(expert_data_path)
    obs = data['obs'].astype(np.float32)
    acts = data['acts'].astype(np.float32)

    n_samples = obs.shape[0]
    print(f"Dataset loaded: {n_samples:,} transitions.")
    print(f"Observation shape: {obs.shape} | Action shape: {acts.shape}")
    print(f"Action range: LVAD [{acts[:, 0].min():.1f}, {acts[:, 0].max():.1f}] RPM, "
          f"RVAD [{acts[:, 1].min():.1f}, {acts[:, 1].max():.1f}] RPM")

    dataset = TensorDataset(torch.tensor(obs), torch.tensor(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    actor = model.actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    print(f"\n--- Starting Behavioral Cloning Pretraining ({epochs} epochs) ---")
    actor.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for b_obs, b_acts in loader:
            b_obs = b_obs.to(device)
            b_acts = b_acts.to(device)

            optimizer.zero_grad()
            # In SB3 SAC, actor(obs, deterministic=True) returns tanh-squashed actions in [-1, 1].
            # We scale expert actions (in [-110, 110] RPM) to match the [-1, 1] range.
            b_acts_scaled = b_acts / 110.0
            pred_acts = actor(b_obs, deterministic=True)
            loss = F.mse_loss(pred_acts, b_acts_scaled)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch:2d}/{epochs:2d}] - MSE Loss: {avg_loss:8.4f}")

    # Save pretrained weights checkpoint
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sac_actor_bc_pretrained.zip")
    model.save(save_path)
    print(f"Saved pretrained SAC model checkpoint to: {save_path}")

    return loss_history


# =====================================================================
# Step 5.4 Validation — Fast Rest Scenario Check
# =====================================================================
def evaluate_policy_rest(model, desc="Pretrained"):
    """
    Runs the agent through a single 200s Rest episode to verify physiological stability
    before launching full RL training.
    """
    print(f"\n--- Evaluating {desc} Agent on 200s Rest Scenario ---")
    env = BiVADEnv(randomize=False, raw_obs=False)
    obs, _ = env.reset()

    maps, laps, raps, cos, fbes, rewards = [], [], [], [], [], []
    lvad_rpms, rvad_rpms = [], []

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
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

    # Check for obvious instability
    if np.mean(laps) < 3.0 or np.mean(raps) < 1.0:
        print("  [WARNING] Severe suction detected during Rest evaluation!")
    elif np.mean(laps) > 18.0:
        print("  [WARNING] Severe congestion detected during Rest evaluation!")
    else:
        print("  [PASS] Hemodynamics are stable and within safe operating limits.")


# =====================================================================
# Step 5.3 & 5.6 — Environment Factory with VecNormalize & Worker Seeding
# =====================================================================
def make_train_env(n_envs=4, base_seed=42):
    """
    Creates parallel SubprocVecEnv wrapped with VecNormalize.
    Uses raw_obs=False (fixed Step-2.9 normalization) so the observation distribution
    is identical between BC pretraining and RL training, and across centralized & independent arms.
    norm_obs=False: keeps standardized inputs.
    norm_reward=True: scales return variance for stable critic loss.
    """
    def _init(rank):
        def _thunk():
            env = BiVADEnv(randomize=True, raw_obs=False)
            env.reset(seed=base_seed + rank * 1000)
            return env
        return _thunk

    if n_envs == 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([_init(0)])
    else:
        vec_env = SubprocVecEnv([_init(i) for i in range(n_envs)])
    vec_norm_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    return vec_norm_env


# =====================================================================
# Checkpoint Detection & Resumability Helper
# =====================================================================
def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """
    Scans checkpoint directory for the latest training checkpoint.
    Returns: (checkpoint_model_path, vecnormalize_path, completed_steps)
    """
    if not os.path.exists(checkpoint_dir):
        # Fallback to root checkpoints for BC pretrained model if seed dir not yet created
        bc_root = "./checkpoints/sac_actor_bc_pretrained.zip"
        if os.path.exists(bc_root):
            return bc_root, None, 0
        return None, None, 0

    pattern = os.path.join(checkpoint_dir, "sac_centralized_*_steps.zip")
    matches = glob.glob(pattern)
    if not matches:
        # Check for seed-specific BC pretrained fallback
        bc_path = os.path.join(checkpoint_dir, "sac_actor_bc_pretrained.zip")
        if os.path.exists(bc_path):
            return bc_path, None, 0
        # Check root BC pretrained fallback
        bc_root = "./checkpoints/sac_actor_bc_pretrained.zip"
        if os.path.exists(bc_root):
            return bc_root, None, 0
        return None, None, 0

    best_step = -1
    best_file = None
    for f in matches:
        m = re.search(r"sac_centralized_.*?(\d+)_steps\.zip", os.path.basename(f))
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_file = f

    if best_file is not None:
        # Look for matching vecnormalize file
        norm_candidates = [
            os.path.join(checkpoint_dir, f"sac_centralized_seed_vecnormalize_{best_step}_steps.pkl"),
            os.path.join(checkpoint_dir, f"sac_centralized_vecnormalize_{best_step}_steps.pkl")
        ]
        norm_file = None
        for cand in norm_candidates:
            if os.path.exists(cand):
                norm_file = cand
                break
        if norm_file is None:
            # Try any matching vecnormalize step file
            step_norms = glob.glob(os.path.join(checkpoint_dir, f"*vecnormalize*_{best_step}_steps.pkl"))
            if step_norms:
                norm_file = step_norms[0]
        return best_file, norm_file, best_step

    return None, None, 0


# =====================================================================
# Main Execution Modes
# =====================================================================
def main():
    import shutil
    default_cpus = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description="Centralized BiVAD SAC Training Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["bc", "eval_bc", "trial", "full", "all"],
                        help="Execution mode: bc, eval_bc, trial, full, or all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--expert_data", type=str, default="expert_data_pi.npz",
                        help="Path to expert demonstration .npz file")
    parser.add_argument("--n_envs", type=int, default=default_cpus,
                        help=f"Number of parallel subprocess workers (auto-detected: {default_cpus})")
    parser.add_argument("--trial_steps", type=int, default=4000,
                        help="Timesteps for trial smoke test")
    parser.add_argument("--full_steps", type=int, default=500000,
                        help="Total timesteps for full RL training")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu for CPU-bound ODE solving)")
    args = parser.parse_args()

    # Prioritize user selection or fallback
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Set seeds across all random generators
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Seed-isolated directories
    checkpoint_dir = f"./checkpoints/seed_{args.seed}"
    tb_dir = f"./tb_centralized/seed_{args.seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"Using compute device: {device} | Parallel workers: {args.n_envs} | Seed: {args.seed}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"TensorBoard directory: {tb_dir}")

    # Base single env for model instantiation
    base_env = BiVADEnv(randomize=False, raw_obs=False)
    base_env.reset(seed=args.seed)

    # SAC Hyperparameters (Steps 5.1 & 5.2)
    policy_kwargs = dict(net_arch=[256, 256])
    model = SAC(
        policy="MlpPolicy",
        env=base_env,
        learning_rate=3e-4,
        buffer_size=300000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=tb_dir
    )

    # -------------------------------------------------------------
    # 1. Behavioral Cloning Pretraining
    # -------------------------------------------------------------
    bc_path = os.path.join(checkpoint_dir, "sac_actor_bc_pretrained.zip")
    if args.mode in ["bc", "all"]:
        pretrain_actor_bc(model, expert_data_path=args.expert_data, epochs=8, device=device, save_dir=checkpoint_dir)
        evaluate_policy_rest(model, desc=f"Pretrained BC (Seed {args.seed})")

    elif args.mode == "eval_bc":
        if os.path.exists(bc_path):
            model = SAC.load(bc_path, env=base_env, device=device)
        elif os.path.exists("./checkpoints/sac_actor_bc_pretrained.zip"):
            model = SAC.load("./checkpoints/sac_actor_bc_pretrained.zip", env=base_env, device=device)
        evaluate_policy_rest(model, desc=f"Pretrained BC (Seed {args.seed})")

    # -------------------------------------------------------------
    # 2. Trial Smoke Test Run (Quick sanity check before long run)
    # -------------------------------------------------------------
    if args.mode in ["trial"]:
        print(f"\n--- Starting Trial Run ({args.trial_steps} steps across {args.n_envs} envs, Seed: {args.seed}) ---")
        train_env = make_train_env(n_envs=args.n_envs, base_seed=args.seed)
        if os.path.exists(bc_path):
            print(f"Loading warm-start model weights from: {bc_path}")
            model = SAC.load(bc_path, env=train_env, device=device, tensorboard_log=tb_dir)
        elif os.path.exists("./checkpoints/sac_actor_bc_pretrained.zip"):
            print("Loading warm-start model weights from root ./checkpoints/...")
            model = SAC.load("./checkpoints/sac_actor_bc_pretrained.zip", env=train_env, device=device, tensorboard_log=tb_dir)
        else:
            model.set_env(train_env)

        reward_callback = RewardComponentCallback(log_freq=max(100, args.trial_steps // 10))
        model.learn(total_timesteps=args.trial_steps, callback=[reward_callback])
        print("\n[PASS] Trial run completed successfully without errors!")
        train_env.save(os.path.join(checkpoint_dir, "trial_vecnormalize.pkl"))
        model.save(os.path.join(checkpoint_dir, "sac_trial_model.zip"))
        train_env.close()

        # Sanity evaluate the trial model
        evaluate_policy_rest(model, desc=f"Trial SAC (Seed {args.seed})")

    # -------------------------------------------------------------
    # 3. Full Training Run (Steps 5.5, 5.6, 5.7 with Auto-Resumability)
    # -------------------------------------------------------------
    if args.mode in ["full", "all"]:
        ckpt_model, ckpt_norm, completed_steps = find_latest_checkpoint(checkpoint_dir)
        train_env = make_train_env(n_envs=args.n_envs, base_seed=args.seed)

        if completed_steps > 0:
            print(f"\n[RESUME] Found existing training checkpoint: {ckpt_model}")
            print(f"         Already completed: {completed_steps:,} timesteps.")
            if ckpt_norm and os.path.exists(ckpt_norm):
                print(f"         Loading VecNormalize statistics: {ckpt_norm}")
                train_env = VecNormalize.load(ckpt_norm, train_env.venv)

            remaining_steps = max(0, args.full_steps - completed_steps)
            print(f"         Remaining steps to reach target {args.full_steps:,}: {remaining_steps:,} timesteps.")
            if remaining_steps <= 0:
                print("Target timesteps already achieved! Loading final model for evaluation.")
                model = SAC.load(ckpt_model, env=train_env, device=device)
                evaluate_policy_rest(model, desc=f"Final Centralized SAC (Seed {args.seed})")
                train_env.close()
                return

            model = SAC.load(ckpt_model, env=train_env, device=device, tensorboard_log=tb_dir)
            reset_timesteps = False
            target_timesteps = args.full_steps
        elif ckpt_model is not None and os.path.exists(ckpt_model):
            print(f"\n[WARM-START] Loading BC pretrained model weights: {ckpt_model}")
            model = SAC.load(ckpt_model, env=train_env, device=device, tensorboard_log=tb_dir)
            reset_timesteps = True
            target_timesteps = args.full_steps
        else:
            print(f"\n[FRESH] Starting full SAC training from scratch ({args.full_steps:,} timesteps, Seed {args.seed})")
            model = SAC(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=3e-4,
                buffer_size=300000,
                batch_size=256,
                ent_coef="auto",
                gamma=0.99,
                tau=0.005,
                seed=args.seed,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
                tensorboard_log=tb_dir
            )
            reset_timesteps = True
            target_timesteps = args.full_steps

        # Checkpoint every 100,000 timesteps
        save_call_freq = max(100000 // args.n_envs, 1000)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_call_freq,
            save_path=checkpoint_dir,
            name_prefix=f"sac_centralized_seed{args.seed}",
            save_vecnormalize=True
        )
        reward_callback = RewardComponentCallback(log_freq=500)

        print(f"Checkpoints will be saved every {save_call_freq * args.n_envs:,} timesteps to {checkpoint_dir}/")
        model.learn(
            total_timesteps=target_timesteps,
            callback=[checkpoint_callback, reward_callback],
            reset_num_timesteps=reset_timesteps,
            log_interval=4,
            progress_bar=True
        )

        # Final saves
        final_model_path = os.path.join(checkpoint_dir, "sac_centralized_final.zip")
        final_norm_path = os.path.join(checkpoint_dir, "vecnormalize_centralized_final.pkl")
        model.save(final_model_path)
        train_env.save(final_norm_path)
        print(f"\nTraining Complete for Seed {args.seed}!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Final normalization stats saved to: {final_norm_path}")
        train_env.close()

        # Final Rest scenario evaluation
        evaluate_policy_rest(model, desc=f"Trained Centralized SAC (Seed {args.seed})")

        # Automatically generate publication plots and performance tables
        try:
            from post_training_evaluation import CentralizedModelAdapter, generate_post_training_artifacts
            adapter = CentralizedModelAdapter(model, name=f"SAC_Centralized_Seed{args.seed}")
            generate_post_training_artifacts(adapter, output_dir=checkpoint_dir, tb_dir=tb_dir, seed=args.seed)
        except Exception as e:
            print(f"[WARNING] Post-training artifact generation encountered an error: {e}")

        # Create auto-archive zip containing models, plots, tables, and logs
        try:
            shutil.make_archive(f"centralized_sac_seed_{args.seed}", "zip", checkpoint_dir)
            print(f">>> [FINAL ZIP] Created centralized_sac_seed_{args.seed}.zip (includes models, tables, and plots)")
        except Exception as e:
            print(f"[NOTE] Archive creation skipped: {e}")


if __name__ == '__main__':
    main()


