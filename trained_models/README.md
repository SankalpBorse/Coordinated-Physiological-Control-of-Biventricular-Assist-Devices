# Pre-Trained BiVAD Deep Reinforcement Learning Policy Checkpoints

This directory contains representative trained checkpoints (Seed 42, 500,000 steps) for the three deep reinforcement learning (DRL) architectures benchmarked in the study:

| Architecture | File Path | Format | Description |
| :--- | :--- | :--- | :--- |
| **Bilateral-Branch SAC** | `sac_bilateral.zip` | Stable-Baselines3 Model Zip | Proposed dual-branch feature extractor architecture coordinating left and right rotary blood pumps under multi-objective reward. |
| **Centralized Flat SAC** | `sac_centralized.zip` | Stable-Baselines3 Model Zip | Unpartitioned centralized baseline architecture operating on the full 9D observation vector. |
| **Independent Dual-Agent SAC** | `sac_independent/actor_l_final.pth`<br>`sac_independent/actor_r_final.pth` | PyTorch State Dicts | Decentralized multi-agent baseline with decoupled left and right actors. |

---

## Loading and Inference

### 1. Bilateral-Branch SAC
```python
from stable_baselines3 import SAC
from bilateral_extractor import BilateralFeatureExtractor

# Register custom feature extractor policy kwargs
custom_objects = {
    "policy_kwargs": dict(
        features_extractor_class=BilateralFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]
    )
}

model = SAC.load("trained_models/sac_bilateral.zip", custom_objects=custom_objects)
action, _ = model.predict(obs, deterministic=True)
# action[0] -> delta_LVAD RPM, action[1] -> delta_RVAD RPM
```

### 2. Centralized Flat SAC
```python
from stable_baselines3 import SAC

model = SAC.load("trained_models/sac_centralized.zip")
action, _ = model.predict(obs, deterministic=True)
```

### 3. Independent Dual-Agent SAC
```python
import torch
from evaluate_independent import Actor, LEFT_OBS_INDICES, RIGHT_OBS_INDICES

actor_l = Actor(obs_dim=5, act_dim=1)
actor_r = Actor(obs_dim=5, act_dim=1)

actor_l.load_state_dict(torch.load("trained_models/sac_independent/actor_l_final.pth", map_location="cpu"))
actor_r.load_state_dict(torch.load("trained_models/sac_independent/actor_r_final.pth", map_location="cpu"))
actor_l.eval()
actor_r.eval()

with torch.no_grad():
    action_l = actor_l(torch.as_tensor(obs[LEFT_OBS_INDICES], dtype=torch.float32).unsqueeze(0))[0].numpy()
    action_r = actor_r(torch.as_tensor(obs[RIGHT_OBS_INDICES], dtype=torch.float32).unsqueeze(0))[0].numpy()
```
