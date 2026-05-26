"""ACT (Action Chunking with Transformers) 单元测试。"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from openpi.models.act import (
    ACTConfig,
    ACT,
    sinusoidal_position_encoding,
    sinusoidal_position_encoding_2d,
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    ACT_IMAGE_KEYS,
)
from openpi.models.model import ModelType, Observation


# ── 固定小模型配置 ──
def _small_config(**overrides) -> ACTConfig:
    defaults = dict(
        action_dim=8,
        action_horizon=10,
        state_dim=8,
        max_token_len=0,
        hidden_dim=64,
        dim_feedforward=128,
        enc_layers=1,
        dec_layers=1,
        nheads=2,
        latent_dim=8,
        kl_weight=1.0,
        vision_feature_dim=64,
        dropout=0.0,
    )
    defaults.update(overrides)
    return ACTConfig(**defaults)


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
    def test_sinusoidal_position_encoding(self):
        pe = sinusoidal_position_encoding(10, 64)
        assert pe.shape == (10, 64)
        # 不同位置应有不同编码
        assert not jnp.allclose(pe[0], pe[1])

    def test_sinusoidal_position_encoding_2d(self):
        pe = sinusoidal_position_encoding_2d(7, 7, 64)
        assert pe.shape == (49, 64)


# ════════════════════════════════════════════════════════════
# Transformer 组件测试
# ════════════════════════════════════════════════════════════

class TestTransformerComponents:
    def test_multi_head_attention(self, rng):
        mha = MultiHeadAttention(64, 2, rngs=nnx.Rngs(rng))
        q = jnp.ones((2, 5, 64))
        k = jnp.ones((2, 8, 64))
        v = jnp.ones((2, 8, 64))
        out = mha(q, k, v)
        assert out.shape == (2, 5, 64)

    def test_mha_with_position_encoding(self, rng):
        mha = MultiHeadAttention(64, 2, rngs=nnx.Rngs(rng))
        q = jnp.ones((2, 5, 64))
        k = jnp.ones((2, 8, 64))
        v = jnp.ones((2, 8, 64))
        q_pos = jnp.zeros((2, 5, 64))
        k_pos = jnp.zeros((2, 8, 64))
        out = mha(q, k, v, q_pos=q_pos, k_pos=k_pos)
        assert out.shape == (2, 5, 64)

    def test_encoder_layer(self, rng):
        layer = TransformerEncoderLayer(64, 2, 128, rngs=nnx.Rngs(rng))
        x = jnp.ones((2, 10, 64))
        out = layer(x)
        assert out.shape == (2, 10, 64)
        assert jnp.all(jnp.isfinite(out))

    def test_decoder_layer(self, rng):
        layer = TransformerDecoderLayer(64, 2, 128, rngs=nnx.Rngs(rng))
        tgt = jnp.ones((2, 5, 64))
        memory = jnp.ones((2, 20, 64))
        out = layer(tgt, memory)
        assert out.shape == (2, 5, 64)
        assert jnp.all(jnp.isfinite(out))


# ════════════════════════════════════════════════════════════
# Config 测试
# ════════════════════════════════════════════════════════════

class TestACTConfig:
    def test_model_type(self, small_config):
        assert small_config.model_type == ModelType.ACT

    def test_inputs_spec(self, small_config):
        obs_spec, act_spec = small_config.inputs_spec(batch_size=4)
        assert obs_spec.state.shape == (4, 8)
        assert act_spec.shape == (4, 10, 8)
        for key in ACT_IMAGE_KEYS:
            assert key in obs_spec.images

    def test_create(self, small_config, rng):
        model = small_config.create(rng)
        assert isinstance(model, ACT)
        assert model.action_dim == 8
        assert model.action_horizon == 10


# ════════════════════════════════════════════════════════════
# 模型测试
# ════════════════════════════════════════════════════════════

class TestACT:
    def test_compute_loss_shape(self, model, rng, fake_obs, fake_actions):
        loss = model.compute_loss(rng, fake_obs, fake_actions, train=True)
        assert loss.shape == (2, 10)  # (B, action_horizon)
        assert jnp.all(jnp.isfinite(loss))

    def test_compute_loss_positive(self, model, rng, fake_obs, fake_actions):
        loss = model.compute_loss(rng, fake_obs, fake_actions, train=True)
        # L1 + KL 应该是非负的
        assert jnp.all(loss >= 0.0)

    def test_sample_actions_shape(self, model, rng, fake_obs):
        actions = model.sample_actions(rng, fake_obs)
        assert actions.shape == (2, 10, 8)  # (B, horizon, action_dim)
        assert jnp.all(jnp.isfinite(actions))

    def test_sample_actions_deterministic(self, model, fake_obs):
        """推理用 z=0 应该是确定性的。"""
        a1 = model.sample_actions(jax.random.PRNGKey(0), fake_obs)
        a2 = model.sample_actions(jax.random.PRNGKey(999), fake_obs)
        # z=0 → 相同输出（不依赖 rng）
        assert jnp.allclose(a1, a2, atol=1e-5)

    def test_different_rng_different_loss(self, model, fake_obs, fake_actions):
        """不同 RNG → 不同 reparameterization → 不同 loss。"""
        loss1 = model.compute_loss(jax.random.PRNGKey(0), fake_obs, fake_actions, train=True)
        loss2 = model.compute_loss(jax.random.PRNGKey(1), fake_obs, fake_actions, train=True)
        assert not jnp.allclose(loss1, loss2)

    def test_cvae_encode(self, model, rng, fake_obs, fake_actions):
        """测试 CVAE 编码器输出形状。"""
        mu, logvar, z = model._cvae_encode(rng, fake_obs, fake_actions, train=True)
        assert mu.shape == (2, 8)  # (B, latent_dim)
        assert logvar.shape == (2, 8)
        assert z.shape == (2, 8)

    def test_kl_weight_effect(self, rng, fake_obs, fake_actions):
        """kl_weight 越大 loss 越大。"""
        cfg_low = _small_config(kl_weight=0.1)
        cfg_high = _small_config(kl_weight=100.0)
        model_low = cfg_low.create(rng)
        model_high = cfg_high.create(rng)
        loss_low = model_low.compute_loss(rng, fake_obs, fake_actions, train=True).mean()
        loss_high = model_high.compute_loss(rng, fake_obs, fake_actions, train=True).mean()
        # 高 kl_weight 应该产生更大的 loss（除非 KL 碰巧为零）
        # 由于随机初始化，KL 不会精确为零
        assert loss_high > loss_low
