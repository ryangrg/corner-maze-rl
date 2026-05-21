"""SB3 visual feature extractors for ``CornerMazeEnv`` image observations.

Ported from legacy ``src/rl/sb3_agents.py`` for use with ``MaskablePPO``
(or any SB3 algorithm via ``policy_kwargs={"features_extractor_class": ...,
"features_extractor_kwargs": {"features_dim": ...}}``).

For now this module ships ``MinigridFeaturesExtractor`` — the single-conv
extractor that worked for the legacy ``train_ppo_mask_imgobs.py`` runs on
the env's 21×21×3 partial view. Stereo (``StereoFeaturesExtractor``) will
land alongside the ``03C_ppo_stereo_cnn.ipynb`` notebook.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """1-Conv + ReLU + Linear extractor for MiniGrid-style image obs.

    Matches the legacy implementation byte-for-byte: 16 3×3 convs, no
    pooling, flatten, then a Linear → ReLU to ``features_dim``. Orthogonal
    init with gain ``sqrt(2)`` (ReLU-tuned). Suitable for SB3 ``CnnPolicy``
    via ``policy_kwargs={"features_extractor_class": MinigridFeaturesExtractor,
    "features_extractor_kwargs": {"features_dim": 128}}``.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 32,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for module in self.linear.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
