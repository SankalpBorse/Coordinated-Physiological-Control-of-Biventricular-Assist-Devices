"""
Phase 7 — Bilateral-Branch Feature Extractor for Coordinated BiVAD Control
========================================================================
Implements a physiologically-structured BaseFeaturesExtractor for Stable-Baselines3 SAC.

Biomechanical Rationale:
------------------------
The human cardiovascular system consists of two distinct circulations in series:
1. Systemic Circulation (Left Heart): High-pressure, high-resistance circuit driven by the LVAD.
   - Primary indicators: Left Atrial Pressure (LAP), LVAD Flow, Aortic Pressure (AoP), Left Ventricular Volume (V_LV).
2. Pulmonary Circulation (Right Heart): Low-pressure, high-compliance circuit driven by the RVAD.
   - Primary indicators: Right Atrial Pressure (RAP), RVAD Flow, Pulmonary Artery Pressure (PAP), Right Ventricular Volume (V_RV).
3. Hemodynamic Coupling: Heart Rate (HR) globally couples both ventricles through mutual cycle timing and pacing.

Architecture:
-------------
- Left Branch:  [LAP (0), LVAD_flow (2), AoP (4), V_LV (6)]  -> MLP(4 -> 64 -> 64) -> 64-dim
- Right Branch: [RAP (1), RVAD_flow (3), PAP (5), V_RV (7)]  -> MLP(4 -> 64 -> 64) -> 64-dim
- Shared Branch: [HR (8)]                                     -> Linear(1 -> 16) + ReLU -> 16-dim
- Concatenation: [left_64 || right_64 || shared_16]          = 144-dim
- Coordination Layer: MLP(144 -> 256) + ReLU                -> 256-dim features
- Standard SAC Actor / Critic Heads (processed via SB3 net_arch=[256, 256])
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BilateralFeaturesExtractor(BaseFeaturesExtractor):
    """
    Physiologically-partitioned feature extractor for BiVAD DRL.
    Separates systemic and pulmonary hemodynamics into dedicated representation branches
    before merging in a cross-ventricular coordination layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(BilateralFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Left heart systemic observation indices: [LAP, LVAD_flow, AoP, V_LV]
        self.left_indices = [0, 2, 4, 6]

        # Right heart pulmonary observation indices: [RAP, RVAD_flow, PAP, V_RV]
        self.right_indices = [1, 3, 5, 7]

        # Shared coupling index: [HR]
        self.shared_indices = [8]

        # Left branch: Systemic Hemodynamics MLP (4 -> 64 -> 64)
        self.left_branch = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Right branch: Pulmonary Hemodynamics MLP (4 -> 64 -> 64)
        self.right_branch = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Shared branch: Global Chronotropic Coupling (1 -> 16)
        self.shared_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        # Cross-ventricular coordination layer (64 + 64 + 16 = 144 -> 256)
        self.coordination_layer = nn.Sequential(
            nn.Linear(144, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass decomposing 9-dim observation into bilateral pathways.
        """
        left_input = observations[:, self.left_indices]
        right_input = observations[:, self.right_indices]
        shared_input = observations[:, self.shared_indices]

        left_features = self.left_branch(left_input)
        right_features = self.right_branch(right_input)
        shared_features = self.shared_branch(shared_input)

        # Concatenate into 144-dimensional representation
        bilateral_repr = torch.cat([left_features, right_features, shared_features], dim=-1)

        # Project through bilateral coordination layer
        coordinated_features = self.coordination_layer(bilateral_repr)
        return coordinated_features


if __name__ == '__main__':
    print("=== Testing BilateralFeaturesExtractor ===")
    obs_space = spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32)
    extractor = BilateralFeaturesExtractor(obs_space, features_dim=256)
    
    # Test batch forward pass
    dummy_obs = torch.randn(8, 9)
    out = extractor(dummy_obs)
    print(f"Input shape : {dummy_obs.shape}")
    print(f"Output shape: {out.shape} (Expected: [8, 256])")
    assert out.shape == (8, 256), "Mismatch in feature extractor output shape!"
    
    # Check total trainable parameters
    total_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Total trainable parameters in extractor: {total_params:,}")
    print("[PASS] BilateralFeaturesExtractor verified successfully!")
