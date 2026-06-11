"""Recurrent PPO (LSTM) *with* invalid-action masking.

sb3-contrib ships ``RecurrentPPO`` (LSTM) and ``MaskablePPO`` (action masking)
as **separate** algorithms — there is no combined class. This module ports
MaskablePPO's masking machinery onto the recurrent base so the corner-maze
notebooks can have an LSTM policy that still respects
``CornerMazeEnv.get_action_mask()`` (e.g. RIGHT is illegal at well-exit poses,
and PAUSE / off-corner PICKUP waste a step + ``STEP_FORWARD_COST``).

It provides three pieces, mirroring the sb3-contrib layout:

* ``MaskableRecurrentRolloutBuffer`` — a ``RecurrentRolloutBuffer`` that also
  stores the per-timestep action mask (so the *same* mask used at collection
  time is re-applied during the update; this keeps the PPO importance ratio
  unbiased).
* ``MaskableRecurrentActorCriticCnnPolicy`` — the ``CnnLstmPolicy`` analogue
  whose action distribution is a ``MaskableCategoricalDistribution`` and whose
  forward / eval / predict paths accept and apply ``action_masks``.
* ``MaskableRecurrentPPO`` — the algorithm: fetches masks in
  ``collect_rollouts``, threads them through the buffer, and re-applies them in
  ``train`` and ``predict``.

⚠️ **Version pin.** This module copies private method bodies from
**sb3_contrib 2.7.0** (``collect_rollouts`` / ``train`` and five recurrent-policy
methods) because sb3-contrib does not expose hooks for either masking or
distribution construction inside the recurrent collection/update loops. If you
bump sb3-contrib, re-sync these bodies against the new source. Nothing detects
drift automatically.

⚠️ **Scope.** Only the non-Dict (Box observation) path is implemented — that is
all the corner-maze image-obs notebooks use. ``MlpLstmPolicy`` /
``MultiInputLstmPolicy`` and a Dict rollout buffer are intentionally omitted.

⚠️ **Save/load.** Model-level ``MaskableRecurrentPPO.save()/load()`` is the only
supported path (it rebuilds the policy from ``policy_kwargs``). The saved ``.zip``
bakes in this module's import path (pickled ``policy_class``), so renaming/moving
this file breaks loading previously-saved models.

⚠️ **Apple MPS.** PyTorch's MPS backend hard-crashes the process in the
masked-distribution *backward* pass (a Metal ``MPSNDArray`` slice assertion). The
``__init__`` therefore warns and falls back to CPU when it sees an MPS device,
unless ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set (before importing torch). CUDA and
CPU are unaffected.
"""
from __future__ import annotations

import os
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, ClassVar, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_device, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    make_masked_proba_distribution,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from sb3_contrib.common.recurrent.type_aliases import (
    RecurrentRolloutBufferSamples,
    RNNStates,
)


def _mask_dims(action_space: spaces.Space) -> int:
    """Width of the action mask for an action space (mirrors MaskableRolloutBuffer)."""
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(sum(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported."
        )
        return 2 * action_space.n
    raise ValueError(f"Unsupported action space {type(action_space)}")


# ---------------------------------------------------------------------------
# Rollout buffer — RecurrentRolloutBuffer + per-timestep action masks
# ---------------------------------------------------------------------------

class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor  # sequence-padding mask (NOT the action mask)
    action_masks: th.Tensor  # the invalid-action mask, padded like ``actions``


class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """``RecurrentRolloutBuffer`` that also stores the per-step action masks.

    The masks are padded exactly like ``actions`` in ``_get_samples`` (NOT
    flattened the MaskablePPO way) so that mask row *i* lines up with logit row
    *i* in the padded ``(n_seq * max_length, ...)`` batch consumed by
    ``evaluate_actions``.
    """

    action_masks: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.mask_dims = _mask_dims(self.action_space)
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        # ``lstm_states`` rides through **kwargs to the recurrent parent's
        # keyword-only param; verified against sb3_contrib 2.7.0.
        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None):
        # Body copied from RecurrentRolloutBuffer.get (sb3_contrib 2.7.0) with
        # "action_masks" added to the swap_and_flatten list.
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env=None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        # The parent sets self.pad / self.seq_start_indices for THIS batch and
        # returns the base sample; we extend it with the padded action masks.
        base: RecurrentRolloutBufferSamples = super()._get_samples(batch_inds, env_change, env)
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # Pad with 1.0 (all-valid) on padded rows for clarity; correctness does
        # not depend on the pad value — padded rows are excluded from every loss
        # term via ``rollout_data.mask`` and an all-masked row would still be a
        # finite uniform dist (apply_masking uses HUGE_NEG=-1e8, not -inf).
        action_masks = self.pad(self.action_masks[batch_inds], padding_value=1.0).reshape(
            (padded_batch_size, self.mask_dims)
        )
        return MaskableRecurrentRolloutBufferSamples(*base, action_masks=action_masks)


# ---------------------------------------------------------------------------
# Policy — CnnLstmPolicy with a maskable action distribution
# ---------------------------------------------------------------------------

class MaskableRecurrentActorCriticCnnPolicy(RecurrentActorCriticCnnPolicy):
    """``CnnLstmPolicy`` whose action distribution supports invalid-action masks.

    The parent (non-maskable) ``__init__`` builds a ``CategoricalDistribution``
    head and optimizer; we swap in a ``MaskableCategoricalDistribution``, rebuild
    the (identically-shaped) action head, and rebuild the optimizer LAST so it
    tracks the new head's parameters. Masking is applied at the call sites
    (forward / get_distribution / evaluate_actions), mirroring
    ``MaskableActorCriticPolicy``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # lr_schedule(1) is what the parent built its optimizer with.
        initial_lr = self.optimizer.param_groups[0]["lr"]
        self.action_dist = make_masked_proba_distribution(self.action_space)
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))
        self.action_net = self.action_net.to(self.device)
        # Rebuild LAST so the new action_net params are tracked; the old head's
        # params drop out of self.parameters() on reassignment. Reuse the same
        # optimizer_kwargs the recurrent parent stored (already includes Adam
        # eps=1e-5 from the SB3 ActorCriticPolicy default).
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=initial_lr, **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        # Override the inherited ActorCriticPolicy version, which raises
        # ValueError for a MaskableCategoricalDistribution. Masking is applied by
        # the callers, not here.
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ):
        # Call extract_features from the grandparent (BaseModel), matching the
        # recurrent policy's own get_distribution.
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, lstm_states

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts, action_masks=action_masks
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        # Body copied from RecurrentActorCriticPolicy.predict (sb3_contrib 2.7.0)
        # with action_masks threaded into the internal _predict call. Returns the
        # UPDATED lstm states so memory carries across steps.
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        if state is None:
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states


# Alias mirroring sb3-contrib's CnnLstmPolicy naming.
CnnLstmPolicy = MaskableRecurrentActorCriticCnnPolicy


# ---------------------------------------------------------------------------
# Algorithm — RecurrentPPO that fetches, stores, and re-applies action masks
# ---------------------------------------------------------------------------

class MaskableRecurrentPPO(RecurrentPPO):
    """Recurrent PPO (LSTM) with invalid-action masking.

    Use the string policy alias ``"CnnLstmPolicy"`` exactly as you would with
    ``MaskablePPO('CnnPolicy', ...)``::

        model = MaskableRecurrentPPO("CnnLstmPolicy", masked_env, ...)

    The env must expose action masks (wrap it in
    ``sb3_contrib.common.wrappers.ActionMasker``).
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "CnnLstmPolicy": MaskableRecurrentActorCriticCnnPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[MaskableRecurrentActorCriticCnnPolicy]],
        env: Union[GymEnv, str],
        *args,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ):
        # Apple MPS hard-crashes the kernel in the masked-distribution backward
        # pass (a Metal MPSNDArray slice assertion in torch 2.x). Stock
        # RecurrentPPO is fine on MPS, but the masking we add is not. The
        # corner-maze CNN+LSTM is tiny, so CPU is comparable — default to CPU on
        # MPS unless the user opted into PyTorch's per-op CPU fallback
        # (PYTORCH_ENABLE_MPS_FALLBACK=1, which must be set before importing torch).
        if (
            get_device(device).type == "mps"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
        ):
            warnings.warn(
                "MaskableRecurrentPPO falls back to CPU on Apple MPS: PyTorch's MPS "
                "backend crashes in the masked-distribution backward pass "
                "(MPSNDArray slice assertion). The corner-maze CNN+LSTM is small so "
                "CPU is comparable. To keep MPS, set PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "before importing torch; pass device='cpu' to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"
        super().__init__(policy, env, *args, device=device, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Replace the plain recurrent buffer with one that also stores masks,
        # reusing the exact hidden-state shape the parent computed.
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        self.rollout_buffer = MaskableRecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        # Body copied from RecurrentPPO.collect_rollouts (sb3_contrib 2.7.0) with
        # the three MaskablePPO masking changes: fetch masks, pass to forward,
        # store in the buffer.
        assert isinstance(
            rollout_buffer, MaskableRecurrentRolloutBuffer
        ), "MaskableRecurrentPPO requires a MaskableRecurrentRolloutBuffer"
        assert self._last_obs is not None, "No previous observation was provided"

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts, action_masks=action_masks
                )

            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        # Body copied from RecurrentPPO.train (sb3_contrib 2.7.0). The ONLY change
        # is passing action_masks into evaluate_actions. The `mask` field below is
        # the sequence-PADDING mask and must keep gating every loss/KL term.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Sequence-padding mask (NOT the action mask).
                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskableRecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        # collect_rollouts(use_masking=True) is the default; the base learn loop
        # calls it without the kwarg, so masking is on unless overridden.
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
