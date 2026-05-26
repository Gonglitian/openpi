"""ResNet18 轻量视觉编码器 (Flax NNX)。

DP (Diffusion Policy) 与 ACT 等 policy 共用的图像 backbone。
使用 GroupNorm 替代 BatchNorm，适合 RL 小 batch 训练场景。
"""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx


class BasicBlock(nnx.Module):
    """ResNet BasicBlock 残差块。

    结构: Conv3x3 → GroupNorm → ReLU → Conv3x3 → GroupNorm + skip → ReLU
    当 stride > 1 或输入输出通道不同时，skip 分支使用 1x1 Conv + GroupNorm 进行对齐。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        num_groups: int = 8,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.gn1 = nnx.GroupNorm(num_groups=num_groups, num_features=out_channels, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.gn2 = nnx.GroupNorm(num_groups=num_groups, num_features=out_channels, rngs=rngs)

        # 需要对齐维度时使用 1x1 卷积投影
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            self.downsample_gn = nnx.GroupNorm(
                num_groups=num_groups, num_features=out_channels, rngs=rngs
            )
        else:
            self.downsample_conv = None
            self.downsample_gn = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """前向传播。

        Args:
            x: 输入特征图，shape (B, H, W, C)。

        Returns:
            输出特征图，shape (B, H', W', out_channels)。
        """
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample_conv is not None:
            residual = self.downsample_conv(residual)
            residual = self.downsample_gn(residual)

        out = out + residual
        out = nnx.relu(out)
        return out


class ResNet18(nnx.Module):
    """ResNet18 视觉编码器。

    4 阶段结构，通道数 [64, 128, 256, 512]，每阶段 2 个 BasicBlock。
    输出全局平均池化后的 feature_dim 维特征向量。

    Args:
        feature_dim: 输出特征维度，默认 512。
        num_groups: GroupNorm 的分组数，默认 8。
        rngs: Flax NNX 随机数生成器。
    """

    def __init__(
        self,
        *,
        feature_dim: int = 512,
        num_groups: int = 8,
        return_spatial: bool = False,
        rngs: nnx.Rngs,
    ):
        # 初始卷积: 7x7 stride 2
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.gn1 = nnx.GroupNorm(num_groups=num_groups, num_features=64, rngs=rngs)

        # 4 个残差阶段
        channels = [64, 128, 256, 512]
        blocks_per_stage = [2, 2, 2, 2]

        self.stages = []
        in_ch = 64
        for stage_idx in range(4):
            out_ch = channels[stage_idx]
            num_blocks = blocks_per_stage[stage_idx]
            stage_blocks = []

            for block_idx in range(num_blocks):
                # 阶段 2-4 的第一个 block 使用 stride=2 下采样
                if block_idx == 0 and stage_idx > 0:
                    stride = 2
                else:
                    stride = 1

                stage_blocks.append(
                    BasicBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        num_groups=num_groups,
                        rngs=rngs,
                    )
                )
                in_ch = out_ch

            self.stages.append(stage_blocks)

        self.return_spatial = return_spatial

        # 最终全连接层: 将 512 维映射到 feature_dim (仅全局池化模式)
        if not return_spatial:
            self.fc = nnx.Linear(
                in_features=channels[-1],
                out_features=feature_dim,
                rngs=rngs,
            )
        else:
            # 空间模式: 1x1 卷积投影通道数到 feature_dim
            self.spatial_proj = nnx.Conv(
                in_features=channels[-1],
                out_features=feature_dim,
                kernel_size=(1, 1),
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """前向传播。

        Args:
            x: 输入图像，shape (B, H, W, 3)。

        Returns:
            return_spatial=False: 特征向量 (B, feature_dim)。
            return_spatial=True: 空间特征图 (B, H', W', feature_dim)。
        """
        # 初始卷积 + GroupNorm + ReLU
        x = self.conv1(x)
        x = self.gn1(x)
        x = nnx.relu(x)

        # 3x3 max pool stride 2
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # 4 个残差阶段
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)

        if self.return_spatial:
            # 空间模式: 保留 H' x W' 空间维度
            x = self.spatial_proj(x)  # (B, H', W', feature_dim)
            return x
        else:
            # 全局池化模式
            x = jnp.mean(x, axis=(1, 2))  # (B, C)
            x = self.fc(x)
            return x
