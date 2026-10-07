"""
Phase 7 — Bilateral-Branch Deep Reinforcement Learning Training (SAC).
======================================================================
Trains the physiologically-structured Bilateral-Branch SAC architecture:
  - Left Branch: [LAP, LVAD_flow, AoP, V_LV] -> MLP(4 -> 64 -> 64) -> 64-dim
  - Right Branch: [RAP, RVAD_flow, PAP, V_RV] -> MLP(4 -> 64 -> 64) -> 64-dim
  - Shared Branch: [HR] -> Linear(1 -> 16) -> 16-dim
  - Coordination Layer: MLP(144 -> 256) -> 256-dim features
  - Actor / Critic MLP heads: net_arch=[256, 256]

Features:
  - Custom BaseFeaturesExtractor registered via SB3 policy_kwargs
  - Behavioral Cloning warm-start on expert demonstration dataset
  - Parallel SubprocVecEnv + VecNormalize with running reward scaling
  - Comprehensive TensorBoard logging (5 reward components + hemodynamics)
  - Automatic post-training 9-scenario clinical validation suite
  - Resumable training checkpoints every 100k steps
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

# Local project imports
from bivad_env import BiVADEnv
from bilateral_extractor import BilateralFeaturesExtractor
from post_training_evaluation import CentralizedModelAdapter, generate_post_training_artifacts


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
# Behavioral Cloning Warm-Start (Pretrain Bilateral SAC Actor)
# =====================================================================
def pretrain_actor_bc(model, expert_data_path='expert_data_pi.npz', epochs=8, batch_size=256, lr=1e-3, device='cpu', save_dir="./checkpoints"):
    """
    Supervised regression pretraining of the Bilateral SAC actor network using expert transitions.
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

    print(f"\n--- Starting Behavioral Cloning Pretraining for Bilateral SAC ({epochs} epochs) ---")
    actor.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_acts in loader:
            batch_obs = batch_obs.to(device)
            batch_acts = batch_acts.to(device)

            pred_acts = actor(batch_obs)
            loss = F.mse_loss(pred_acts, batch_acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Supervised MSE Loss: {avg_loss:.6f}")

    # Check loss drop
    init_loss = loss_history[0]
    final_loss = loss_history[-1]
    pct_drop = (init_loss - final_loss) / init_loss * 100.0
    print(f"\nBC Pretraining Complete: Loss dropped {pct_drop:.1f}% ({init_loss:.6f} -> {final_loss:.6f})")

    # Save pretrained model
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sac_bilateral_actor_bc_pretrained.zip")
    model.save(save_path)
    print(f"Saved pretrained Bilateral SAC model to: {save_path}")

    # Also save to root checkpoints as universal fallback
    os.makedirs("./checkpoints", exist_ok=True)
    root_fallback = "./checkpoints/sac_bilateral_actor_bc_pretrained.zip"
    if save_path != root_fallback:
        model.save(root_fallback)

    return model


# =====================================================================
# Sanity Evaluation Function for BC Pretrained Model
# =====================================================================
def evaluate_bc_model(model, seed=42):
    """
    Evaluates the behavioral cloned policy on a 2000-step (200s) unperturbed Rest episode.
    """
    print("\n--- Running BC Pretrained Sanity Check (Scenario 1: Rest, 200s) ---")
    env = BiVADEnv(randomize=False, raw_obs=False)
    obs, info = env.reset(seed=seed)

    maps = []
    laps = []
    raps = []
    cos = []
    fbes = []
    rewards = []
    lvad_rpms = []
    rvad_rpms = []

    for step in range(2000):
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

    if np.mean(laps) < 3.0 or np.mean(raps) < 1.0:
        print("  [WARNING] Severe suction detected during Rest evaluation!")
    elif np.mean(laps) > 18.0:
        print("  [WARNING] Severe congestion detected during Rest evaluation!")
    else:
        print("  [PASS] Hemodynamics are stable and within safe operating limits.")


# =====================================================================
# Environment Factory with VecNormalize & Worker Seeding
# =====================================================================
def make_train_env(n_envs=4, base_seed=42):
    """
    Creates parallel SubprocVecEnv wrapped with VecNormalize.
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
    Scans checkpoint directory for the latest bilateral training checkpoint.
    Returns: (checkpoint_model_path, vecnormalize_path, completed_steps)
    """
    if not os.path.exists(checkpoint_dir):
        bc_root = "./checkpoints/sac_bilateral_actor_bc_pretrained.zip"
        if os.path.exists(bc_root):
            return bc_root, None, 0
        return None, None, 0

    pattern = os.path.join(checkpoint_dir, "sac_bilateral_*_steps.zip")
    matches = glob.glob(pattern)
    if not matches:
        bc_path = os.path.join(checkpoint_dir, "sac_bilateral_actor_bc_pretrained.zip")
        if os.path.exists(bc_path):
            return bc_path, None, 0
        bc_root = "./checkpoints/sac_bilateral_actor_bc_pretrained.zip"
        if os.path.exists(bc_root):
            return bc_root, None, 0
        return None, None, 0

    best_step = -1
    best_file = None
    for f in matches:
        m = re.search(r"sac_bilateral_.*?(\d+)_steps\.zip", os.path.basename(f))
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_file = f

    if best_file is not None:
        norm_candidates = [
            os.path.join(checkpoint_dir, f"sac_bilateral_seed_vecnormalize_{best_step}_steps.pkl"),
            os.path.join(checkpoint_dir, f"sac_bilateral_vecnormalize_{best_step}_steps.pkl"),
            os.path.join(checkpoint_dir, "vecnormalize_bilateral_latest.pkl")
        ]
        norm_file = None
        for nc in norm_candidates:
            if os.path.exists(nc):
                norm_file = nc
                break
        return best_file, norm_file, best_step

    return None, None, 0


# =====================================================================
# Main Execution Pipeline
# =====================================================================
def main():
    import shutil
    default_cpus = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description="Bilateral-Branch BiVAD SAC Training Pipeline")
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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Set seeds across random generators
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Seed-isolated directories
    checkpoint_dir = f"./checkpoints/seed_{args.seed}"
    tb_dir = f"./tb_bilateral/seed_{args.seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"Using compute device: {device} | Parallel workers: {args.n_envs} | Seed: {args.seed}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"TensorBoard directory: {tb_dir}")

    # Base env for initialization
    base_env = BiVADEnv(randomize=False, raw_obs=False)
    base_env.reset(seed=args.seed)

    # SAC Hyperparameters with Bilateral Features Extractor
    policy_kwargs = dict(
        features_extractor_class=BilateralFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256]
    )

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
    if args.mode in ["bc", "all"]:
        bc_checkpoint = os.path.join(checkpoint_dir, "sac_bilateral_actor_bc_pretrained.zip")
        if os.path.exists(bc_checkpoint):
            print(f"\n[NOTE] BC checkpoint already exists at: {bc_checkpoint}. Skipping pretraining.")
            model = SAC.load(bc_checkpoint, env=base_env, device=device,
                             custom_objects={"features_extractor_class": BilateralFeaturesExtractor})
        else:
            model = pretrain_actor_bc(model, expert_data_path=args.expert_data, epochs=8,
                                      batch_size=256, lr=1e-3, device=device, save_dir=checkpoint_dir)

    # -------------------------------------------------------------
    # 2. Evaluate BC Model
    # -------------------------------------------------------------
    if args.mode in ["eval_bc", "all"]:
        bc_checkpoint = os.path.join(checkpoint_dir, "sac_bilateral_actor_bc_pretrained.zip")
        if os.path.exists(bc_checkpoint):
            model = SAC.load(bc_checkpoint, env=base_env, device=device,
                             custom_objects={"features_extractor_class": BilateralFeaturesExtractor})
        evaluate_bc_model(model, seed=args.seed)

    # -------------------------------------------------------------
    # 3. Trial Smoke-Test Mode
    # -------------------------------------------------------------
    if args.mode == "trial":
        print(f"\n--- Starting Trial Mode ({args.trial_steps} steps on {args.n_envs} workers) ---")
        vec_env = make_train_env(n_envs=args.n_envs, base_seed=args.seed)
        model.set_env(vec_env)

        callback = RewardComponentCallback(log_freq=200)
        model.learn(total_timesteps=args.trial_steps, callback=callback)

        trial_save = os.path.join(checkpoint_dir, "sac_bilateral_trial_model.zip")
        model.save(trial_save)
        vec_env.save(os.path.join(checkpoint_dir, "sac_bilateral_trial_vecnormalize.pkl"))
        print(f"[PASS] Trial execution completed. Model saved to: {trial_save}")
        vec_env.close()

    # -------------------------------------------------------------
    # 4. Full Production Training Mode
    # -------------------------------------------------------------
    if args.mode in ["full", "all"]:
        print(f"\n=======================================================")
        print(f"Starting Full Bilateral-Branch SAC Training ({args.full_steps:,} steps)")
        print(f"Architecture: BilateralFeaturesExtractor -> MLP(256) -> Heads([256, 256])")
        print(f"Workers: {args.n_envs} | Seed: {args.seed} | Device: {device}")
        print(f"=======================================================\n")

        vec_env = make_train_env(n_envs=args.n_envs, base_seed=args.seed)

        # Check for existing checkpoint resumption
        latest_model_path, latest_norm_path, completed_steps = find_latest_checkpoint(checkpoint_dir)
        reset_num_timesteps = True

        if latest_model_path is not None and completed_steps > 0:
            print(f"[RESUME] Found existing checkpoint at step {completed_steps:,}: {latest_model_path}")
            if latest_norm_path is not None and os.path.exists(latest_norm_path):
                print(f"[RESUME] Loading normalizer statistics from: {latest_norm_path}")
                vec_env = VecNormalize.load(latest_norm_path, vec_env.venv)
                vec_env.training = True
                vec_env.norm_reward = True

            model = SAC.load(
                latest_model_path,
                env=vec_env,
                device=device,
                tensorboard_log=tb_dir,
                custom_objects={"features_extractor_class": BilateralFeaturesExtractor}
            )
            model.num_timesteps = completed_steps
            reset_num_timesteps = False
            remaining_steps = max(0, args.full_steps - completed_steps)
            print(f"[RESUME] Resuming training for remaining {remaining_steps:,} steps...")
        else:
            bc_checkpoint = os.path.join(checkpoint_dir, "sac_bilateral_actor_bc_pretrained.zip")
            if os.path.exists(bc_checkpoint):
                print(f"[WARM-START] Loading BC pretrained actor weights from: {bc_checkpoint}")
                model = SAC.load(
                    bc_checkpoint,
                    env=vec_env,
                    device=device,
                    tensorboard_log=tb_dir,
                    custom_objects={"features_extractor_class": BilateralFeaturesExtractor}
                )
            else:
                print("[NOTE] Starting from scratch with custom Bilateral extractor.")
                model.set_env(vec_env)
            remaining_steps = args.full_steps

        if remaining_steps <= 0:
            print(f"[COMPLETE] Model has already completed target {args.full_steps:,} steps. Skipping training.")
        else:
            # Checkpoint callback: save every 100,000 steps
            checkpoint_callback = CheckpointCallback(
                save_freq=max(1, 100000 // args.n_envs),
                save_path=checkpoint_dir,
                name_prefix="sac_bilateral",
                save_vecnormalize=True,
                verbose=1
            )
            reward_callback = RewardComponentCallback(log_freq=500)

            # Execute training loop
            model.learn(
                total_timesteps=remaining_steps,
                callback=[checkpoint_callback, reward_callback],
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name="SAC_Bilateral"
            )

            # Save final trained artifacts
            final_model_path = os.path.join(checkpoint_dir, "sac_bilateral_final.zip")
            final_norm_path = os.path.join(checkpoint_dir, "vecnormalize_bilateral_final.pkl")
            model.save(final_model_path)
            vec_env.save(final_norm_path)
            print(f"\n[SUCCESS] Final Bilateral model saved: {final_model_path}")
            print(f"[SUCCESS] Final normalizer saved: {final_norm_path}")

        vec_env.close()

        # -------------------------------------------------------------
        # 5. Post-Training 9-Scenario Evaluation & Plotting
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("POST-TRAINING MULTI-SCENARIO CLINICAL EVALUATION")
        print("=======================================================")
        try:
            adapter = CentralizedModelAdapter(model, name=f"SAC_Bilateral_Seed{args.seed}")
            generate_post_training_artifacts(adapter, output_dir=checkpoint_dir, tb_dir=tb_dir, seed=args.seed)
        except Exception as e:
            print(f"[WARNING] Post-training artifact generation encountered an error: {e}")

        # Copy tables and plots to convenient root locations
        root_tables = f"./tables/seed_{args.seed}"
        root_plots = f"./plots/seed_{args.seed}"
        os.makedirs(root_tables, exist_ok=True)
        os.makedirs(root_plots, exist_ok=True)

        for f in glob.glob(os.path.join(checkpoint_dir, "tables", "*")):
            shutil.copy2(f, os.path.join(root_tables, os.path.basename(f)))
        for f in glob.glob(os.path.join(checkpoint_dir, "plots", "*")):
            shutil.copy2(f, os.path.join(root_plots, os.path.basename(f)))

        # Create auto-archive zip containing models, plots, tables, and logs
        try:
            shutil.make_archive(f"bilateral_sac_seed_{args.seed}", "zip", checkpoint_dir)
            print(f">>> [FINAL ZIP] Created bilateral_sac_seed_{args.seed}.zip (includes models, tables, and plots)")
        except Exception as e:
            print(f"[NOTE] Archive creation skipped: {e}")

        print(f"\n>>> [COMPLETE] Phase 7 Bilateral-Branch SAC Training for Seed {args.seed} finished successfully!")


if __name__ == '__main__':
    main()
