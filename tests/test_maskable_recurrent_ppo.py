"""Tests for the bespoke MaskableRecurrentPPO (RecurrentPPO + action masking).

These are fast CPU smoke/correctness checks. The single highest-risk detail is
the optimizer-tracks-the-swapped-action-head guarantee (a fumbled rebuild makes
PPO appear to run while learning nothing in the policy head), so it gets its own
assertion.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

pytest.importorskip("sb3_contrib")

from minigrid.wrappers import ImgObsWrapper  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402

from corner_maze_rl.encoders.visual import MinigridFeaturesExtractor  # noqa: E402
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv  # noqa: E402
from corner_maze_rl.env.wrappers import mask_fn  # noqa: E402
from corner_maze_rl.models.maskable_recurrent_ppo import MaskableRecurrentPPO  # noqa: E402

LSTM_HIDDEN = 64


def make_env(max_steps: int = 300):
    env = CornerMazeEnv(
        render_mode="rgb_array",
        max_steps=max_steps,
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
        session_type="PI+VC f2 acquisition",
        obs_mode="view",
    )
    return ActionMasker(ImgObsWrapper(env), mask_fn)


def make_model(env, **overrides):
    kw = dict(
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        device="cpu",
        seed=0,
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            lstm_hidden_size=LSTM_HIDDEN,
        ),
    )
    kw.update(overrides)
    return MaskableRecurrentPPO("CnnLstmPolicy", env, **kw)


def test_learn_runs():
    model = make_model(make_env())
    model.learn(total_timesteps=128)


def test_optimizer_tracks_swapped_action_net():
    # #1 silent-failure checkpoint: the maskable action head's params must be in
    # the optimizer after the post-init head swap, or the head never trains.
    model = make_model(make_env())
    opt_ids = {id(p) for group in model.policy.optimizer.param_groups for p in group["params"]}
    head_ids = {id(p) for p in model.policy.action_net.parameters()}
    assert head_ids, "action_net has no parameters"
    assert head_ids <= opt_ids, "swapped action_net params are not tracked by the optimizer"


def test_action_net_weights_change_after_learn():
    model = make_model(make_env())
    before = model.policy.action_net.weight.detach().clone()
    model.learn(total_timesteps=128)
    after = model.policy.action_net.weight.detach()
    assert not th.allclose(before, after), "action head did not update during training"


def test_predict_respects_action_mask():
    # Masking bites: with RIGHT(1) and PAUSE(4) masked off, the policy must never
    # emit them — across a stochastic rollout.
    env = make_env()
    model = make_model(env)
    obs, _ = env.reset(seed=0)
    forbid = np.array([True, False, True, True, False])  # allow L, F, PICKUP only
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    for _ in range(25):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_start,
            deterministic=False, action_masks=forbid,
        )
        assert int(action) not in (1, 4)
        obs, _, term, trunc, _ = env.step(int(action))
        ep_start = np.zeros((1,), dtype=bool)
        if term or trunc:
            obs, _ = env.reset()
            lstm_states = None
            ep_start = np.ones((1,), dtype=bool)


def test_lstm_state_threads_in_predict():
    env = make_env()
    model = make_model(env)
    obs, _ = env.reset(seed=0)
    a1, s1 = model.predict(
        obs, state=None, episode_start=np.ones((1,), dtype=bool),
        deterministic=True, action_masks=np.array(mask_fn(env)),
    )
    assert s1 is not None
    obs, _, _, _, _ = env.step(int(a1))
    a2, s2 = model.predict(
        obs, state=s1, episode_start=np.zeros((1,), dtype=bool),
        deterministic=True, action_masks=np.array(mask_fn(env)),
    )
    assert s2 is not None
    # Hidden state advanced across the step → LSTM memory is threaded.
    assert not np.allclose(s1[0], s2[0])


def test_buffer_action_masks_padded_and_aligned():
    env = make_env()
    model = make_model(env, n_steps=64)
    model.learn(total_timesteps=64)  # one full rollout + one update
    buf = model.rollout_buffer
    assert buf.mask_dims == int(env.action_space.n)
    for sample in buf.get(model.batch_size):
        # action_masks padded like observations/actions (same padded batch dim)
        assert sample.action_masks.shape[0] == sample.observations.shape[0]
        assert sample.action_masks.shape[1] == buf.mask_dims


@pytest.mark.skipif(not th.backends.mps.is_available(), reason="MPS not available")
def test_mps_falls_back_to_cpu(monkeypatch):
    # PyTorch MPS crashes in the masked-distribution backward pass, so the class
    # must warn and use CPU instead of hard-crashing the process.
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    env = make_env()
    with pytest.warns(RuntimeWarning, match="MPS"):
        model = make_model(env, device="mps")
    assert model.device.type == "cpu"
    model.learn(total_timesteps=64)  # must not crash


def test_save_load_roundtrip(tmp_path):
    env = make_env()
    model = make_model(env)
    model.learn(total_timesteps=128)
    obs, _ = env.reset(seed=1)
    mask = np.array(mask_fn(env))
    a1, _ = model.predict(
        obs, state=None, episode_start=np.ones((1,), dtype=bool),
        deterministic=True, action_masks=mask,
    )
    path = tmp_path / "m.zip"
    model.save(str(path))
    m2 = MaskableRecurrentPPO.load(str(path), env=env)
    assert m2.policy.lstm_actor.hidden_size == LSTM_HIDDEN
    a2, _ = m2.predict(
        obs, state=None, episode_start=np.ones((1,), dtype=bool),
        deterministic=True, action_masks=mask,
    )
    assert np.allclose(a1, a2)
    m2.learn(total_timesteps=64)  # one more update without state_dict/shape error
