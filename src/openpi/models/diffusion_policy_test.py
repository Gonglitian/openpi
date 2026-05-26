"""Diffusion Policy 单元测试。"""

import jax
import jax.numpy as jnp
import pytest

from openpi.models.diffusion_policy import (
    DiffusionPolicyConfig,
    DiffusionPolicy,
    cosine_beta_schedule,
    sinusoidal_embedding,
    mish,
    DP_IMAGE_KEYS,
)
from openpi.models.model import ModelType, Observation


# ── 固定小模型配置，加速测试 ──
def _small_config(**overrides) -> DiffusionPolicyConfig:
    defaults = dict(
        action_dim=8,
        action_horizon=8,
        state_dim=8,
        max_token_len=0,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_diffusion_steps=10,
        num_inference_steps=10,
        vision_feature_dim=32,
        n_groups=4,
        kernel_size=3,
    )
    defaults.update(overrides)
    return DiffusionPolicyConfig(**defaults)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_config():
    return _small_config()


@pytest.fixture
def model(small_config, rng):
    return small_config.create(rng)


@pytest.fixture
def fake_obs(small_config):
    return small_config.fake_obs(batch_size=2)


@pytest.fixture
def fake_actions(small_config):
    return small_config.fake_act(batch_size=2)


# ════════════════════════════════════════════════════════════
# 工具函数测试
# ════════════════════════════════════════════════════════════

class TestUtils:
    def test_mish(self):
        x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        y = mish(x)
        assert y.shape == (4,)
        # mish(0) = 0
        assert jnp.allclose(y[1], 0.0, atol=1e-6)
        # mish(x) ≈ x for large x
        assert y[3] > 1.5

    def test_sinusoidal_embedding(self):
        timesteps = jnp.array([0, 5, 9])
        emb = sinusoidal_embedding(timesteps, 32)
        assert emb.shape == (3, 32)
        # 不同时间步应有不同嵌入
        assert not jnp.allclose(emb[0], emb[1])

    def test_sinusoidal_embedding_odd_dim(self):
        emb = sinusoidal_embedding(jnp.array([0]), 33)
        assert emb.shape == (1, 33)

    def test_cosine_beta_schedule(self):
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)
        assert jnp.all(betas >= 0.0)
        assert jnp.all(betas <= 0.999)
        # beta 应该递增（cosine schedule 特性）
        assert betas[-1] > betas[0]


# ════════════════════════════════════════════════════════════
# Config 测试
# ════════════════════════════════════════════════════════════

class TestDiffusionPolicyConfig:
    def test_model_type(self, small_config):
        assert small_config.model_type == ModelType.DIFFUSION_POLICY

    def test_inputs_spec(self, small_config):
        obs_spec, act_spec = small_config.inputs_spec(batch_size=4)
        assert obs_spec.state.shape == (4, 8)
        assert act_spec.shape == (4, 8, 8)
        for key in DP_IMAGE_KEYS:
            assert key in obs_spec.images
            assert obs_spec.images[key].shape == (4, 224, 224, 3)

    def test_create(self, small_config, rng):
        model = small_config.create(rng)
        assert isinstance(model, DiffusionPolicy)
        assert model.action_dim == 8
        assert model.action_horizon == 8


# ════════════════════════════════════════════════════════════
# 模型测试
# ════════════════════════════════════════════════════════════

class TestDiffusionPolicy:
    def test_compute_loss_shape(self, model, rng, fake_obs, fake_actions):
        loss = model.compute_loss(rng, fake_obs, fake_actions, train=True)
        # 返回 (B, action_horizon)
        assert loss.shape == (2, 8)
        assert jnp.all(jnp.isfinite(loss))

    def test_compute_loss_positive(self, model, rng, fake_obs, fake_actions):
        loss = model.compute_loss(rng, fake_obs, fake_actions, train=True)
        assert jnp.all(loss >= 0.0)

    def test_sample_actions_shape(self, model, rng, fake_obs):
        actions = model.sample_actions(rng, fake_obs)
        assert actions.shape == (2, 8, 8)  # (B, horizon, action_dim)
        assert jnp.all(jnp.isfinite(actions))

    def test_different_rng_different_loss(self, model, fake_obs, fake_actions):
        loss1 = model.compute_loss(jax.random.PRNGKey(0), fake_obs, fake_actions)
        loss2 = model.compute_loss(jax.random.PRNGKey(1), fake_obs, fake_actions)
        # 不同 RNG → 不同噪声 → 不同 loss
        assert not jnp.allclose(loss1, loss2)

    def test_ddim_sampling(self, rng):
        cfg = _small_config(use_ddim=True, num_inference_steps=5)
        model = cfg.create(rng)
        obs = cfg.fake_obs(2)
        actions = model.sample_actions(rng, obs)
        assert actions.shape == (2, 8, 8)
        assert jnp.all(jnp.isfinite(actions))

    def test_observation_encoding(self, model, fake_obs):
        """测试视觉编码器正确拼接特征。"""
        cond = model._encode_observation(fake_obs)
        assert cond.shape == (2, 32)  # (B, vision_feature_dim)
        assert jnp.all(jnp.isfinite(cond))
