"""ACT (Action Chunking with Transformers) 实现 — Flax NNX。

基于 CVAE 结构的 Transformer 动作分块模型:
- 编码器 (仅训练时): 将动作序列编码为潜变量 z (高斯后验)
- 解码器 (训练+推理): 从潜变量 z + 视觉特征 + 本体感受 → 预测动作序列

视觉编码采用 ResNet18 (GroupNorm)，保留空间特征用于 Transformer cross-attention。

参考:
  - Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023

与 OpenPI 框架集成: 实现 BaseModelConfig / BaseModel 接口。
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.resnet import ResNet18
from openpi.shared import array_typing as at

# ============================================================================
# ACT 使用的图像 key
# ============================================================================
ACT_IMAGE_KEYS = ("base_0_rgb", "wrist_0_rgb")

ACT_IMAGE_RESOLUTION = (224, 224)


# ============================================================================
# 工具函数
# ============================================================================

def sinusoidal_position_encoding(seq_len: int, dim: int) -> jax.Array:
    """正弦余弦位置编码。

    Args:
        seq_len: 序列长度。
        dim: 编码维度。

    Returns:
        编码矩阵，shape (seq_len, dim)。
    """
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    dims = jnp.arange(dim, dtype=jnp.float32)[None, :]
    angles = positions / jnp.power(10000.0, 2.0 * (dims // 2) / dim)
    encoding = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return encoding


def sinusoidal_position_encoding_2d(h: int, w: int, dim: int) -> jax.Array:
    """二维正弦余弦位置编码，用于图像空间特征。

    Args:
        h, w: 空间高宽。
        dim: 编码维度（必须能被 2 整除）。

    Returns:
        shape (h*w, dim)。
    """
    half = dim // 2
    pe_h = sinusoidal_position_encoding(h, half)  # (h, half)
    pe_w = sinusoidal_position_encoding(w, half)  # (w, half)
    # 扩展为 2D 网格并拼接
    pe_h = jnp.tile(pe_h[:, None, :], (1, w, 1))  # (h, w, half)
    pe_w = jnp.tile(pe_w[None, :, :], (h, 1, 1))  # (h, w, half)
    pe_2d = jnp.concatenate([pe_h, pe_w], axis=-1)  # (h, w, dim)
    return pe_2d.reshape(h * w, dim)


# ============================================================================
# Transformer 组件
# ============================================================================

class MultiHeadAttention(nnx.Module):
    """多头注意力，支持 Q/K 位置编码注入。"""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dropout = dropout

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        *,
        q_pos: jax.Array | None = None,
        k_pos: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        """前向传播。

        Args:
            q: (B, Lq, D), k: (B, Lk, D), v: (B, Lk, D)
            q_pos, k_pos: 可选位置编码，加到 Q 和 K 上
            train: 是否启用 dropout

        Returns:
            (B, Lq, D)
        """
        B, Lq, D = q.shape

        # 注入位置编码
        if q_pos is not None:
            q = q + q_pos
        if k_pos is not None:
            k = k + k_pos

        # 投影并拆分多头
        q = self.q_proj(q).reshape(B, Lq, self.nheads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, -1, self.nheads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, -1, self.nheads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        out = jnp.matmul(attn_weights, v)  # (B, nheads, Lq, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, Lq, D)
        out = self.out_proj(out)
        return out


class TransformerEncoderLayer(nnx.Module):
    """Pre-LN Transformer 编码器层。"""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.self_attn = MultiHeadAttention(hidden_dim, nheads, dropout, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_dim, dim_feedforward, rngs=rngs)
        self.fc2 = nnx.Linear(dim_feedforward, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, pos: jax.Array | None = None, train: bool = False) -> jax.Array:
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, q_pos=pos, k_pos=pos, train=train)
        x = residual + x

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.fc2(jax.nn.relu(self.fc1(x)))
        x = residual + x
        return x


class TransformerDecoderLayer(nnx.Module):
    """Pre-LN Transformer 解码器层 (self-attn + cross-attn + FFN)。"""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.self_attn = MultiHeadAttention(hidden_dim, nheads, dropout, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.cross_attn = MultiHeadAttention(hidden_dim, nheads, dropout, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_dim, dim_feedforward, rngs=rngs)
        self.fc2 = nnx.Linear(dim_feedforward, hidden_dim, rngs=rngs)

    def __call__(
        self,
        tgt: jax.Array,
        memory: jax.Array,
        *,
        tgt_pos: jax.Array | None = None,
        memory_pos: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        """前向传播。

        Args:
            tgt: (B, Lq, D) 目标序列
            memory: (B, Lm, D) 编码器输出 / 上下文
            tgt_pos: query 位置编码
            memory_pos: memory 位置编码
        """
        # Self-attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, q_pos=tgt_pos, k_pos=tgt_pos, train=train)
        tgt = residual + tgt

        # Cross-attention
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.cross_attn(tgt, memory, memory, q_pos=tgt_pos, k_pos=memory_pos, train=train)
        tgt = residual + tgt

        # FFN
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = self.fc2(jax.nn.relu(self.fc1(tgt)))
        tgt = residual + tgt
        return tgt


# ============================================================================
# ACT Config
# ============================================================================

@dataclasses.dataclass(frozen=True)
class ACTConfig(_model.BaseModelConfig):
    """ACT 模型配置。"""

    # ---- Transformer 架构 ----
    hidden_dim: int = 512
    dim_feedforward: int = 2048
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    dropout: float = 0.1

    # ---- CVAE ----
    latent_dim: int = 32
    kl_weight: float = 10.0

    # ---- 视觉 ----
    vision_feature_dim: int = 512
    vision_spatial_h: int = 7
    vision_spatial_w: int = 7

    # 是否使用视觉编码器 (False = state-only, 无图像)
    use_vision: bool = True

    # 观测历史长度 (1=单帧, 2+=多帧)。
    # ACT 原论文只用单帧，但多帧时每帧作为独立 memory token 供 decoder cross-attend。
    obs_horizon: int = 1

    # 单帧状态维度
    per_frame_state_dim: int = 8

    # ---- 动作空间 ----
    action_dim: int = 8
    action_horizon: int = 16
    max_token_len: int = 0  # ACT 不使用文本
    state_dim: int = 8

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.ACT

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACT":
        return ACT(config=self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        S = jax.ShapeDtypeStruct
        if self.use_vision:
            images = {k: S((batch_size, *ACT_IMAGE_RESOLUTION, 3), jnp.float32) for k in ACT_IMAGE_KEYS}
            image_masks = {k: S((batch_size,), jnp.bool_) for k in ACT_IMAGE_KEYS}
        else:
            images = {}
            image_masks = {}

        obs = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=S((batch_size, self.state_dim), jnp.float32),
        )
        actions = S((batch_size, self.action_horizon, self.action_dim), jnp.float32)
        return obs, actions


# ============================================================================
# ACT 模型
# ============================================================================

class ACT(_model.BaseModel):
    """ACT (Action Chunking with Transformers) 模型。

    结构:
      - 视觉编码器: ResNet18 (保留空间特征)
      - CVAE 编码器 (仅训练): actions → TransformerEncoder → (mu, logvar)
      - Transformer 解码器: latent z + proprio + vision → 动作序列
    """

    def __init__(self, config: ACTConfig, *, rngs: nnx.Rngs):
        super().__init__(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
        )

        hidden_dim = config.hidden_dim
        self.config = config
        self._use_vision = config.use_vision

        self._obs_horizon = config.obs_horizon
        self._per_frame_state_dim = config.per_frame_state_dim

        if config.use_vision:
            num_cameras = len(ACT_IMAGE_KEYS)
            # ---- 视觉编码器 (每个相机独立 ResNet18) ----
            self.vision_encoders = []
            for _ in range(num_cameras):
                self.vision_encoders.append(
                    ResNet18(
                        feature_dim=config.vision_feature_dim,
                        num_groups=8,
                        rngs=rngs,
                        return_spatial=True,
                    )
                )
            self.vision_proj = nnx.Linear(config.vision_feature_dim, hidden_dim, rngs=rngs)
        else:
            # ---- State-only: 逐帧编码 → memory tokens ----
            # 输入: 单帧 per_frame_state_dim 或 flatten 后由模型 reshape
            self.vision_encoders = []
            self.state_frame_encoder = nnx.Linear(config.per_frame_state_dim, hidden_dim, rngs=rngs)

        # ---- CVAE 编码器 (仅训练时使用) ----
        self.encoder_action_proj = nnx.Linear(config.action_dim, hidden_dim, rngs=rngs)
        self.encoder_joint_proj = nnx.Linear(config.state_dim, hidden_dim, rngs=rngs)

        # CLS token (可学习参数)
        self.cls_embed = nnx.Param(jax.random.normal(rngs.params(), (1, 1, hidden_dim)) * 0.02)

        # Transformer 编码器层
        self.encoder_layers = []
        for _ in range(config.enc_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(
                    hidden_dim=hidden_dim,
                    nheads=config.nheads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    rngs=rngs,
                )
            )
        self.encoder_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        # 潜变量投影: hidden_dim → 2 * latent_dim (mu + logvar)
        self.latent_proj = nnx.Linear(hidden_dim, 2 * config.latent_dim, rngs=rngs)

        # ---- 解码器 ----
        # 潜变量 → hidden_dim
        self.latent_out_proj = nnx.Linear(config.latent_dim, hidden_dim, rngs=rngs)

        # 本体感受投影
        self.proprio_proj = nnx.Linear(config.state_dim, hidden_dim, rngs=rngs)

        # Query embeddings (可学习，每个 action step 一个)
        self.query_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.action_horizon, hidden_dim)) * 0.02
        )

        # Transformer 解码器层
        self.decoder_layers = []
        for _ in range(config.dec_layers):
            self.decoder_layers.append(
                TransformerDecoderLayer(
                    hidden_dim=hidden_dim,
                    nheads=config.nheads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    rngs=rngs,
                )
            )
        self.decoder_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        # 动作预测头
        self.action_head = nnx.Linear(hidden_dim, config.action_dim, rngs=rngs)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _encode_context(self, observation: _model.Observation) -> jax.Array:
        """编码观测为 decoder memory tokens。

        vision 模式: 图像空间特征序列 (B, N_tokens, hidden_dim)
        state-only 模式:
          - obs_horizon=1: (B, state_dim) → Linear → (B, 1, hidden_dim)
          - obs_horizon>1: (B, T*42) → reshape (B, T, 42) → Linear 逐帧 → (B, T, hidden_dim)
        """
        if self._use_vision:
            all_tokens = []
            for i, key in enumerate(ACT_IMAGE_KEYS):
                img = observation.images[key]
                mask = observation.image_masks[key]
                spatial = self.vision_encoders[i](img)
                B, h, w, C = spatial.shape
                spatial = spatial.reshape(B, h * w, C)
                spatial = self.vision_proj(spatial)
                spatial = spatial * mask[:, None, None].astype(spatial.dtype)
                all_tokens.append(spatial)
            return jnp.concatenate(all_tokens, axis=1)
        else:
            state = observation.state  # (B, obs_horizon * per_frame_dim) or (B, per_frame_dim)
            B = state.shape[0]
            if self._obs_horizon > 1:
                # Reshape flatten state → (B, obs_horizon, per_frame_dim)
                state = state.reshape(B, self._obs_horizon, self._per_frame_state_dim)
                # 逐帧编码 → (B, obs_horizon, hidden_dim)
                tokens = jax.nn.relu(self.state_frame_encoder(state))
                return tokens  # (B, T, hidden_dim) — T 个 memory tokens
            else:
                # 单帧 → 单个 token
                token = jax.nn.relu(self.state_frame_encoder(state))
                return token[:, None, :]  # (B, 1, hidden_dim)

    def _cvae_encode(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """CVAE 编码器: 动作序列 → 潜变量 (mu, logvar, z)。

        仅在训练时调用。

        Args:
            observation: 当前观测
            actions: 目标动作序列 (B, action_horizon, action_dim)

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            z: (B, latent_dim)
        """
        B = actions.shape[0]
        hidden_dim = self.config.hidden_dim

        # 投影动作和状态
        action_embed = self.encoder_action_proj(actions)  # (B, T, hidden_dim)
        state_embed = self.encoder_joint_proj(observation.state)  # (B, hidden_dim)
        state_embed = state_embed[:, None, :]  # (B, 1, hidden_dim)

        # CLS token
        cls_token = jnp.broadcast_to(self.cls_embed.value, (B, 1, hidden_dim))

        # 拼接序列: [CLS, state, actions]
        seq = jnp.concatenate([cls_token, state_embed, action_embed], axis=1)  # (B, 2+T, hidden_dim)

        # 位置编码
        seq_len = seq.shape[1]
        pos_enc = sinusoidal_position_encoding(seq_len, hidden_dim)  # (seq_len, hidden_dim)
        pos_enc = jnp.broadcast_to(pos_enc[None], (B, seq_len, hidden_dim))

        # Transformer 编码器
        x = seq
        for layer in self.encoder_layers:
            x = layer(x, pos=pos_enc, train=train)
        x = self.encoder_norm(x)

        # 提取 CLS token 输出 → 潜变量
        cls_output = x[:, 0, :]  # (B, hidden_dim)
        latent_params = self.latent_proj(cls_output)  # (B, 2 * latent_dim)
        mu, logvar = jnp.split(latent_params, 2, axis=-1)  # 各 (B, latent_dim)

        # 重参数化
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + std * eps

        return mu, logvar, z

    def _decode(
        self,
        z: jax.Array,
        observation: _model.Observation,
        visual_tokens: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        """Transformer 解码器: 从 latent z + 视觉 + 本体感受 → 动作序列。

        Args:
            z: (B, latent_dim)
            observation: 观测
            visual_tokens: (B, N_visual, hidden_dim)

        Returns:
            predicted_actions: (B, action_horizon, action_dim)
        """
        B = z.shape[0]
        hidden_dim = self.config.hidden_dim

        # 投影潜变量和本体感受
        latent_input = self.latent_out_proj(z)[:, None, :]  # (B, 1, hidden_dim)
        proprio_input = self.proprio_proj(observation.state)[:, None, :]  # (B, 1, hidden_dim)

        # 拼接上下文 memory: [latent, proprio, visual_tokens]
        memory = jnp.concatenate([latent_input, proprio_input, visual_tokens], axis=1)  # (B, 2+N_visual, hidden_dim)

        # Query embeddings
        query_pos = jnp.broadcast_to(self.query_embed.value, (B, self.action_horizon, hidden_dim))

        # Transformer 解码器
        tgt = jnp.zeros((B, self.action_horizon, hidden_dim))
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_pos=query_pos, train=train)
        tgt = self.decoder_norm(tgt)

        # 动作预测
        actions = self.action_head(tgt)  # (B, action_horizon, action_dim)
        return actions

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """计算 ACT 损失 = L1 重建损失 + KL 散度。

        Args:
            observation: 当前观测
            actions: 目标动作 (B, action_horizon, action_dim)

        Returns:
            逐时间步损失，shape (B, action_horizon)
        """
        rng_encode, rng_other = jax.random.split(rng)

        # 编码图像
        visual_tokens = self._encode_context(observation)

        # CVAE 编码 (训练时)
        mu, logvar, z = self._cvae_encode(rng_encode, observation, actions, train=train)

        # 解码
        predicted_actions = self._decode(z, observation, visual_tokens, train=train)

        # L1 重建损失 (逐时间步)
        l1_loss = jnp.mean(jnp.abs(predicted_actions - actions), axis=-1)  # (B, action_horizon)

        # KL 散度: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1)  # (B,)

        # 将 KL 均摊到各时间步，与 l1_loss 形状对齐
        kl_per_step = kl_loss[:, None] / self.action_horizon  # (B, 1)

        total_loss = l1_loss + self.config.kl_weight * kl_per_step  # (B, action_horizon)
        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> _model.Actions:
        """推理: 使用先验均值 z=0 生成动作序列。

        Args:
            observation: 当前观测。

        Returns:
            动作序列，shape (B, action_horizon, action_dim)。
        """
        B = observation.state.shape[0]

        # 编码图像
        visual_tokens = self._encode_context(observation)

        # 推理时使用先验均值 z = 0
        z = jnp.zeros((B, self.config.latent_dim))

        # 解码
        predicted_actions = self._decode(z, observation, visual_tokens, train=False)
        return predicted_actions
