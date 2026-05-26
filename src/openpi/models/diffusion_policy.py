"""Diffusion Policy (DP) 实现 — Flax NNX。

基于 1D Conditional U-Net 的扩散策略模型，使用 DDPM/DDIM 采样。
视觉编码采用 ResNet18 (GroupNorm)，对每个相机视角独立编码后拼接机器人状态
构成全局条件特征 (global_cond)，通过 FiLM 机制注入 U-Net 各层。

参考:
  - Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

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
# DP 使用的图像 key（基座相机 + 腕部相机）
# ============================================================================
DP_IMAGE_KEYS = ("base_0_rgb", "wrist_0_rgb")

# DP 使用的图像分辨率
DP_IMAGE_RESOLUTION = (224, 224)


# ============================================================================
# 工具函数
# ============================================================================

def mish(x: jax.Array) -> jax.Array:
    """Mish 激活函数: x * tanh(softplus(x))。"""
    return x * jnp.tanh(jax.nn.softplus(x))


def sinusoidal_embedding(timesteps: jax.Array, embed_dim: int) -> jax.Array:
    """正弦余弦位置编码，将离散扩散时间步映射到连续向量。

    Args:
        timesteps: 整数时间步，shape (B,)。
        embed_dim: 输出嵌入维度。

    Returns:
        嵌入向量，shape (B, embed_dim)。
    """
    half_dim = embed_dim // 2
    # 频率因子: exp(-log(10000) * i / (half_dim - 1))
    freq = jnp.exp(-math.log(10000.0) * jnp.arange(half_dim) / (half_dim - 1))
    # (B, half_dim)
    args = timesteps[:, None].astype(jnp.float32) * freq[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    # 若 embed_dim 为奇数，补零
    if embed_dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> jax.Array:
    """Cosine noise schedule (squaredcos_cap_v2)。

    返回 beta_t, t = 0, ..., num_steps - 1。

    Args:
        num_steps: 扩散步数 T。
        s: 偏移参数，防止 beta 在 t=0 附近过小。

    Returns:
        beta 数组，shape (num_steps,)，值域 [0, 0.999]。
    """
    steps = jnp.arange(num_steps + 1, dtype=jnp.float32)
    f_t = jnp.cos((steps / num_steps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return jnp.clip(betas, 0.0, 0.999)


# ============================================================================
# 1D 卷积基础组件
# ============================================================================

class Conv1dBlock(nnx.Module):
    """1D 卷积块: Conv1d -> GroupNorm -> Mish。

    输入 layout: (B, T, C)（JAX channels-last 约定）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        n_groups: int = 8,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.group_norm = nnx.GroupNorm(
            num_groups=n_groups,
            num_features=out_channels,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """前向传播。

        Args:
            x: (B, T, C_in)

        Returns:
            (B, T, C_out)
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = mish(x)
        return x


class ConditionalResidualBlock1D(nnx.Module):
    """带 FiLM 条件注入的 1D 残差块。

    结构:
      Conv1dBlock -> FiLM(cond) -> Conv1dBlock -> residual
    其中 FiLM: scale, bias = split(Linear(cond)); x = x * (scale + 1) + bias

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        cond_dim: 条件特征维度（global_cond 的维度）。
        kernel_size: 卷积核大小。
        n_groups: GroupNorm 分组数。
        rngs: 随机数生成器。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        rngs: nnx.Rngs,
    ):
        self.block1 = Conv1dBlock(
            in_channels, out_channels,
            kernel_size=kernel_size, n_groups=n_groups, rngs=rngs,
        )
        # FiLM 投影: cond -> (scale, bias)，各 out_channels 维
        self.cond_proj = nnx.Linear(
            in_features=cond_dim,
            out_features=out_channels * 2,
            rngs=rngs,
        )
        self.block2 = Conv1dBlock(
            out_channels, out_channels,
            kernel_size=kernel_size, n_groups=n_groups, rngs=rngs,
        )

        # 残差分支: 若通道数不匹配则做 1x1 投影
        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1,),
                rngs=rngs,
            )
        else:
            self.residual_conv = None

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """前向传播。

        Args:
            x: 输入特征，shape (B, T, C_in)。
            cond: 全局条件特征，shape (B, cond_dim)。

        Returns:
            输出特征，shape (B, T, C_out)。
        """
        residual = x
        out = self.block1(x)

        # FiLM 调制
        cond_emb = self.cond_proj(cond)  # (B, 2 * out_channels)
        scale, bias = jnp.split(cond_emb, 2, axis=-1)  # 各 (B, out_channels)
        # 扩展到 (B, 1, out_channels) 以广播到时间维度
        scale = scale[:, None, :]
        bias = bias[:, None, :]
        out = out * (scale + 1.0) + bias

        out = self.block2(out)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        return out + residual


class Downsample1d(nnx.Module):
    """1D 下采样: stride-2 卷积。"""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Upsample1d(nnx.Module):
    """1D 上采样: stride-2 转置卷积。"""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv_transpose = nnx.ConvTranspose(
            in_features=channels,
            out_features=channels,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv_transpose(x)


# ============================================================================
# 条件 1D U-Net
# ============================================================================

class ConditionalUnet1D(nnx.Module):
    """1D Conditional U-Net，用于预测动作序列上的噪声。

    编码器-解码器结构，每层包含 ConditionalResidualBlock1D + Down/Up-sample，
    中间层 (mid) 使用两层 ConditionalResidualBlock1D。
    跳跃连接采用拼接 (concat) 方式。

    Args:
        input_dim: 输入通道数（= action_dim）。
        global_cond_dim: 全局条件特征维度。
        diffusion_step_embed_dim: 扩散时间步嵌入维度。
        down_dims: U-Net 每层的通道数。
        kernel_size: 卷积核大小。
        n_groups: GroupNorm 分组数。
        rngs: 随机数生成器。
    """

    def __init__(
        self,
        input_dim: int,
        *,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        rngs: nnx.Rngs,
    ):
        self.diffusion_step_embed_dim = diffusion_step_embed_dim

        # 时间步嵌入 MLP: sinusoidal -> Linear -> Mish -> Linear
        self.time_mlp_linear1 = nnx.Linear(
            in_features=diffusion_step_embed_dim,
            out_features=diffusion_step_embed_dim * 4,
            rngs=rngs,
        )
        self.time_mlp_linear2 = nnx.Linear(
            in_features=diffusion_step_embed_dim * 4,
            out_features=diffusion_step_embed_dim,
            rngs=rngs,
        )

        # 条件维度 = 时间步嵌入 + 全局条件特征
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # 构建 U-Net 层级
        all_dims = (input_dim,) + tuple(down_dims)
        num_levels = len(down_dims)

        # ---------- 编码器 (下采样路径) ----------
        self.encoder_blocks = []  # List[ConditionalResidualBlock1D]
        self.downsamplers = []    # List[Downsample1d | None]
        for i in range(num_levels):
            dim_in = all_dims[i]
            dim_out = all_dims[i + 1]
            self.encoder_blocks.append(
                ConditionalResidualBlock1D(
                    dim_in, dim_out,
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    rngs=rngs,
                )
            )
            # 最后一层不下采样
            if i < num_levels - 1:
                self.downsamplers.append(Downsample1d(dim_out, rngs=rngs))
            else:
                self.downsamplers.append(None)

        # ---------- 中间层 (bottleneck) ----------
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(
            mid_dim, mid_dim,
            cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
            rngs=rngs,
        )
        self.mid_block2 = ConditionalResidualBlock1D(
            mid_dim, mid_dim,
            cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
            rngs=rngs,
        )

        # ---------- 解码器 (上采样路径) ----------
        # 解码器从最底层向上: dec_idx=0 对应 i=num_levels-1 (最底层)
        # 最底层: 输入来自 mid (mid_dim), 无 skip
        # 其他层: 输入 = 上一级解码器的输出(经上采样) + encoder skip
        self.decoder_blocks = []   # List[ConditionalResidualBlock1D]
        self.upsamplers = []       # List[Upsample1d | None]
        prev_dec_out = mid_dim  # mid_block 输出通道数
        for dec_idx, i in enumerate(reversed(range(num_levels))):
            dim_target = all_dims[i]  # 解码器此层输出通道数

            if dec_idx == 0:
                # 最底层: 直接从 mid 接入
                dec_in = prev_dec_out
            else:
                # 跳跃连接: concat(上一层输出, encoder[i] 输出)
                skip_dim = all_dims[i + 1]  # encoder 第 i 层输出 = all_dims[i+1]
                dec_in = prev_dec_out + skip_dim

            self.decoder_blocks.append(
                ConditionalResidualBlock1D(
                    dec_in, dim_target,
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    rngs=rngs,
                )
            )
            prev_dec_out = dim_target

            if i > 0:
                self.upsamplers.append(Upsample1d(dim_target, rngs=rngs))
            else:
                self.upsamplers.append(None)

        # ---------- 输出投影 ----------
        self.final_conv = nnx.Conv(
            in_features=all_dims[0],  # = input_dim
            out_features=input_dim,
            kernel_size=(1,),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        timestep: jax.Array,
        global_cond: jax.Array,
    ) -> jax.Array:
        """前向传播: 预测噪声。

        Args:
            x: 含噪动作序列，shape (B, T, action_dim)。
            timestep: 扩散时间步，shape (B,)，整数。
            global_cond: 全局条件特征，shape (B, global_cond_dim)。

        Returns:
            预测噪声，shape (B, T, action_dim)。
        """
        # 时间步嵌入
        t_emb = sinusoidal_embedding(timestep, self.diffusion_step_embed_dim)  # (B, D)
        t_emb = self.time_mlp_linear1(t_emb)
        t_emb = mish(t_emb)
        t_emb = self.time_mlp_linear2(t_emb)  # (B, D)

        # 拼接时间步嵌入和全局条件
        cond = jnp.concatenate([t_emb, global_cond], axis=-1)  # (B, cond_dim)

        # ---------- 编码器前向 ----------
        encoder_outputs = []
        h = x
        for i, (block, downsampler) in enumerate(
            zip(self.encoder_blocks, self.downsamplers)
        ):
            h = block(h, cond)
            encoder_outputs.append(h)
            if downsampler is not None:
                h = downsampler(h)

        # ---------- 中间层 ----------
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # ---------- 解码器前向 ----------
        # decoder_blocks[0] 对应最底层 (i = num_levels-1)
        # decoder_blocks[-1] 对应最顶层 (i = 0)
        num_levels = len(self.encoder_blocks)
        for dec_idx, (block, upsampler) in enumerate(
            zip(self.decoder_blocks, self.upsamplers)
        ):
            # 对应编码器的索引 (从高到低)
            enc_idx = num_levels - 1 - dec_idx

            if dec_idx > 0:
                # 跳跃连接: 拼接编码器对应层的输出
                skip = encoder_outputs[enc_idx]
                # 处理时间维度不匹配的情况（下采样/上采样引起）
                if h.shape[1] != skip.shape[1]:
                    # 截断或补零对齐
                    min_t = min(h.shape[1], skip.shape[1])
                    h = h[:, :min_t, :]
                    skip = skip[:, :min_t, :]
                h = jnp.concatenate([h, skip], axis=-1)

            h = block(h, cond)

            if upsampler is not None:
                h = upsampler(h)

        # 处理最终输出的时间维度对齐
        target_t = x.shape[1]
        if h.shape[1] != target_t:
            h = h[:, :target_t, :]

        # 输出投影
        h = self.final_conv(h)
        return h


# ============================================================================
# Diffusion Policy 配置
# ============================================================================

@dataclasses.dataclass(frozen=True)
class DiffusionPolicyConfig(_model.BaseModelConfig):
    """Diffusion Policy 模型配置。

    继承 BaseModelConfig，定义 U-Net 结构、扩散参数、视觉编码器参数等。
    """

    # U-Net 通道数递进
    down_dims: tuple[int, ...] = (256, 512, 1024)
    # 扩散时间步嵌入维度
    diffusion_step_embed_dim: int = 256
    # 1D 卷积核大小
    kernel_size: int = 3
    # GroupNorm 分组数
    n_groups: int = 8

    # 扩散过程参数
    num_diffusion_steps: int = 100
    noise_schedule: str = "squaredcos_cap_v2"

    # 推理参数
    num_inference_steps: int = 100
    use_ddim: bool = False
    ddim_eta: float = 0.0

    # 视觉编码器输出特征维度
    vision_feature_dim: int = 512

    # 是否使用视觉编码器 (False = state-only, 无图像输入)
    use_vision: bool = True

    # 观测历史长度 (1=单帧, 2+=多帧)。
    # DP 原论文默认 n_obs_steps=2。多帧时 state flatten 为 obs_horizon * per_frame_state_dim。
    obs_horizon: int = 1

    # 单帧状态维度（如 8 for DROID, 42 for state-only）
    per_frame_state_dim: int = 8

    # 总状态维度 = obs_horizon * per_frame_state_dim (自动计算)
    # 作为 BaseModelConfig 接口的一部分，供 inputs_spec / FakeDataset 使用
    state_dim: int = 8

    # 模型默认参数
    action_dim: int = 8
    action_horizon: int = 16
    max_token_len: int = 0  # DP 不使用语言 prompt

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.DIFFUSION_POLICY

    @override
    def create(self, rng: at.KeyArrayLike) -> "DiffusionPolicy":
        return DiffusionPolicy(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        """返回模型输入的 shape/dtype 规范。"""
        S = jax.ShapeDtypeStruct

        if self.use_vision:
            images = {k: S([batch_size, *DP_IMAGE_RESOLUTION, 3], jnp.float32) for k in DP_IMAGE_KEYS}
            image_masks = {k: S([batch_size], jnp.bool_) for k in DP_IMAGE_KEYS}
        else:
            # State-only: 空图像字典
            images = {}
            image_masks = {}

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=images,
                image_masks=image_masks,
                state=S([batch_size, self.state_dim], jnp.float32),
            )
        action_spec = S([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec


# ============================================================================
# Diffusion Policy 模型
# ============================================================================

class DiffusionPolicy(_model.BaseModel):
    """Diffusion Policy 主模型。

    架构:
      1. 视觉编码器: 对每个相机用独立 ResNet18 编码，拼接后与 state 一起投影为 global_cond
      2. 1D Conditional U-Net: 在动作序列上预测噪声，通过 FiLM 注入 global_cond
      3. DDPM / DDIM 采样: 从纯噪声迭代去噪，生成动作序列

    噪声调度参数 (alphas_cumprod 等) 作为模型属性存储，不参与梯度计算。
    """

    def __init__(self, config: DiffusionPolicyConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self._config = config
        self._use_vision = config.use_vision

        if config.use_vision:
            num_cameras = len(DP_IMAGE_KEYS)
            # ---------- 视觉编码器: 每个相机一个 ResNet18 ----------
            self.vision_encoders = []
            for _ in range(num_cameras):
                self.vision_encoders.append(
                    ResNet18(
                        feature_dim=config.vision_feature_dim,
                        num_groups=config.n_groups,
                        rngs=rngs,
                    )
                )
            concat_dim = num_cameras * config.vision_feature_dim + config.state_dim
        else:
            # ---------- State-only: MLP 编码 ----------
            self.vision_encoders = []
            self.state_mlp = nnx.Sequential(
                nnx.Linear(config.state_dim, config.vision_feature_dim, rngs=rngs),
                nnx.Linear(config.vision_feature_dim, config.vision_feature_dim, rngs=rngs),
            )
            concat_dim = config.vision_feature_dim

        # ---------- 全局条件投影 ----------
        self.global_cond_proj = nnx.Linear(
            in_features=concat_dim,
            out_features=config.vision_feature_dim,
            rngs=rngs,
        )

        # ---------- 1D Conditional U-Net ----------
        self.unet = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.vision_feature_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            rngs=rngs,
        )

        # ---------- 噪声调度表 (不参与训练) ----------
        betas = cosine_beta_schedule(config.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas)

        # 预计算常用量
        self.num_diffusion_steps = config.num_diffusion_steps
        self.num_inference_steps = config.num_inference_steps
        self.use_ddim = config.use_ddim
        self.ddim_eta = config.ddim_eta

        # 使用 nnx.Variable 存储噪声调度常量（非 Param，不参与梯度更新）
        self._betas = nnx.Variable(betas)
        self._alphas = nnx.Variable(alphas)
        self._alphas_cumprod = nnx.Variable(alphas_cumprod)
        self._sqrt_alphas_cumprod = nnx.Variable(jnp.sqrt(alphas_cumprod))
        self._sqrt_one_minus_alphas_cumprod = nnx.Variable(jnp.sqrt(1.0 - alphas_cumprod))
        self._sqrt_recip_alphas = nnx.Variable(1.0 / jnp.sqrt(alphas))

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _encode_observation(self, observation: _model.Observation) -> jax.Array:
        """编码观测 → 全局条件特征。

        vision 模式: 图像 + 状态 → global_cond
        state-only 模式: 状态 → MLP → global_cond

        Returns:
            global_cond, shape (B, vision_feature_dim)。
        """
        if self._use_vision:
            features = []
            for i, key in enumerate(DP_IMAGE_KEYS):
                img = observation.images[key]  # (B, H, W, 3)
                feat = self.vision_encoders[i](img)  # (B, vision_feature_dim)
                mask = observation.image_masks[key]  # (B,)
                feat = feat * mask[:, None].astype(feat.dtype)
                features.append(feat)
            features.append(observation.state)  # (B, state_dim)
            concat_features = jnp.concatenate(features, axis=-1)
        else:
            # State-only: MLP 编码 42D state
            x = observation.state  # (B, 42)
            x = jax.nn.relu(self.state_mlp.layers[0](x))
            concat_features = self.state_mlp.layers[1](x)

        global_cond = self.global_cond_proj(concat_features)
        return global_cond

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
        """计算扩散损失 (MSE between predicted and true noise)。

        步骤:
          1. 编码观测 → global_cond
          2. 随机采样时间步 t ∈ {0, ..., T-1}，噪声 ε ~ N(0, I)
          3. 构造含噪样本 x_t = sqrt(alpha_bar_t) * actions + sqrt(1 - alpha_bar_t) * ε
          4. 用 U-Net 预测噪声 ε_hat = unet(x_t, t, global_cond)
          5. 返回逐时间步 MSE: mean((ε_hat - ε)^2, axis=-1) → shape (B, action_horizon)

        Args:
            rng: 随机数 key。
            observation: 观测数据。
            actions: 真值动作序列，shape (B, action_horizon, action_dim)。
            train: 是否训练模式（用于图像增强等）。

        Returns:
            逐时间步损失，shape (B, action_horizon)。
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        # 图像预处理 (仅 vision 模式)
        if self._use_vision:
            observation = _model.preprocess_observation(
                preprocess_rng, observation,
                train=train,
                image_keys=DP_IMAGE_KEYS,
                image_resolution=DP_IMAGE_RESOLUTION,
            )

        batch_size = actions.shape[0]

        # 编码观测
        global_cond = self._encode_observation(observation)

        # 随机采样时间步和噪声
        t = jax.random.randint(
            time_rng, shape=(batch_size,), minval=0, maxval=self.num_diffusion_steps
        )
        noise = jax.random.normal(noise_rng, actions.shape)

        # 构造含噪动作 x_t
        sqrt_alpha_bar_t = self._sqrt_alphas_cumprod.value[t][:, None, None]       # (B, 1, 1)
        sqrt_one_minus_alpha_bar_t = self._sqrt_one_minus_alphas_cumprod.value[t][:, None, None]
        x_t = sqrt_alpha_bar_t * actions + sqrt_one_minus_alpha_bar_t * noise  # (B, T, D)

        # 预测噪声
        predicted_noise = self.unet(x_t, t, global_cond)  # (B, T, D)

        # 逐时间步 MSE
        loss = jnp.mean(jnp.square(predicted_noise - noise), axis=-1)  # (B, action_horizon)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> _model.Actions:
        """DDPM / DDIM 采样生成动作序列。

        步骤:
          1. 编码观测 → global_cond
          2. 初始化 x_T ~ N(0, I)，shape (B, action_horizon, action_dim)
          3. 迭代去噪 (使用 jax.lax.fori_loop):
             - DDPM: x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_hat) + σ_t * z
             - DDIM: x_{t-1} = sqrt(ᾱ_{t-1}) * x0_hat + sqrt(1-ᾱ_{t-1}-σ²) * ε_hat + σ * z
          4. 返回 x_0

        Args:
            rng: 随机数 key。
            observation: 观测数据。

        Returns:
            动作序列，shape (B, action_horizon, action_dim)。
        """
        if self._use_vision:
            observation = _model.preprocess_observation(
                None, observation,
                train=False,
                image_keys=DP_IMAGE_KEYS,
                image_resolution=DP_IMAGE_RESOLUTION,
            )

        batch_size = observation.state.shape[0]
        global_cond = self._encode_observation(observation)

        # 初始噪声
        init_rng, loop_rng = jax.random.split(rng)
        x_t = jax.random.normal(
            init_rng, (batch_size, self.action_horizon, self.action_dim)
        )

        if self.use_ddim:
            x_0 = self._sample_ddim(loop_rng, x_t, global_cond)
        else:
            x_0 = self._sample_ddpm(loop_rng, x_t, global_cond)

        return x_0

    def _sample_ddpm(
        self,
        rng: jax.Array,
        x_t: jax.Array,
        global_cond: jax.Array,
    ) -> jax.Array:
        """DDPM 反向采样。

        从 t = T-1 迭代到 t = 0:
          ε_hat = unet(x_t, t, cond)
          x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_hat) + σ_t * z
        其中 σ_t = sqrt(β_t)，t=0 时 z=0。

        使用 jax.lax.fori_loop 实现高效迭代。
        """
        num_steps = self.num_inference_steps

        def step_fn(i: int, carry: tuple) -> tuple:
            x, rng_carry = carry
            # 当前实际时间步: 从 T-1 递减到 0
            t_val = num_steps - 1 - i
            t = jnp.full((x.shape[0],), t_val, dtype=jnp.int32)

            # 预测噪声
            eps_hat = self.unet(x, t, global_cond)

            # DDPM 更新
            beta_t = self._betas.value[t_val]
            sqrt_recip_alpha_t = self._sqrt_recip_alphas.value[t_val]
            sqrt_one_minus_alpha_bar_t = self._sqrt_one_minus_alphas_cumprod.value[t_val]

            # 均值: (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_hat)
            mean = sqrt_recip_alpha_t * (
                x - beta_t / sqrt_one_minus_alpha_bar_t * eps_hat
            )

            # 方差: σ_t = sqrt(β_t)，t=0 时不加噪
            sigma_t = jnp.sqrt(beta_t)
            rng_carry, z_rng = jax.random.split(rng_carry)
            z = jax.random.normal(z_rng, x.shape)
            # t > 0 时加噪，t = 0 时不加
            noise_mask = (t_val > 0).astype(jnp.float32)
            x_prev = mean + sigma_t * z * noise_mask

            return (x_prev, rng_carry)

        x_0, _ = jax.lax.fori_loop(0, num_steps, step_fn, (x_t, rng))
        return x_0

    def _sample_ddim(
        self,
        rng: jax.Array,
        x_t: jax.Array,
        global_cond: jax.Array,
    ) -> jax.Array:
        """DDIM 采样 (可跳步，支持 eta 控制随机性)。

        DDIM 更新:
          x0_hat = (x_t - sqrt(1-ᾱ_t) * ε_hat) / sqrt(ᾱ_t)
          σ = η * sqrt((1-ᾱ_{t-1}) / (1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
          direction = sqrt(1 - ᾱ_{t-1} - σ²) * ε_hat
          x_{t-1} = sqrt(ᾱ_{t-1}) * x0_hat + direction + σ * z
        """
        num_steps = self.num_inference_steps
        eta = self.ddim_eta

        # DDIM 可以使用子集时间步 (当 num_inference_steps < num_diffusion_steps)
        # 均匀选取
        total_steps = self.num_diffusion_steps
        step_ratio = total_steps / num_steps
        timesteps = jnp.array(
            [int(i * step_ratio) for i in range(num_steps)],
            dtype=jnp.int32,
        )
        # 反转: 从大到小
        timesteps = timesteps[::-1]

        def step_fn(i: int, carry: tuple) -> tuple:
            x, rng_carry = carry

            t_val = timesteps[i]
            # 前一步时间（t=0 时 prev 设为 0）
            t_prev = jnp.where(i < num_steps - 1, timesteps[i + 1], 0)

            t = jnp.full((x.shape[0],), t_val, dtype=jnp.int32)

            # 预测噪声
            eps_hat = self.unet(x, t, global_cond)

            # 当前和前一步的 ᾱ
            alpha_bar_t = self._alphas_cumprod.value[t_val]
            alpha_bar_prev = self._alphas_cumprod.value[t_prev]

            # 预测 x_0
            x0_hat = (x - jnp.sqrt(1.0 - alpha_bar_t) * eps_hat) / jnp.sqrt(
                alpha_bar_t
            )

            # 计算 σ
            sigma = (
                eta
                * jnp.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * jnp.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # 方向项
            direction = jnp.sqrt(
                jnp.maximum(1.0 - alpha_bar_prev - sigma**2, 0.0)
            ) * eps_hat

            # 采样
            rng_carry, z_rng = jax.random.split(rng_carry)
            z = jax.random.normal(z_rng, x.shape)
            x_prev = jnp.sqrt(alpha_bar_prev) * x0_hat + direction + sigma * z

            return (x_prev, rng_carry)

        x_0, _ = jax.lax.fori_loop(0, num_steps, step_fn, (x_t, rng))
        return x_0
