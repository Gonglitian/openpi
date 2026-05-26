"""State-only policy transforms.

42D 特权 state 组装: 从 LeRobot flat keys 构造完整的 42D 状态向量。
用于 DP/ACT state-based 变体训练和推理。

支持多帧历史 (obs_horizon > 1): LeRobot 通过 delta_timestamps 返回
(obs_horizon, dim) 形状的观测数据，本 transform 将每帧组装为 42D 后
flatten 为 (obs_horizon * 42,) 向量。模型内部按需 reshape。

42D state 格式 (单帧):
  [0:7]    arm_joint_pos
  [7:14]   arm_joint_vel
  [14:15]  gripper_pos
  [15:22]  ee_pose (xyz + wxyz)
  [22:29]  object_pose (xyz + wxyz)
  [29:36]  goal_pose (xyz + wxyz)
  [36:39]  tcp_to_obj (ee_pos - obj_pos)
  [39:42]  obj_to_goal (obj_pos - goal_pos)
"""

import dataclasses

import numpy as np

from openpi import transforms


# PlayingCardsKitchen 固定 goal pose (xyz + wxyz)
PLAYING_CARDS_GOAL_POSE = np.array(
    [0.5, 0.0, 0.03, 0.02, 0.00, -0.7047, 0.7095], dtype=np.float32
)

# 单帧 state 维度
STATE_DIM_PER_FRAME = 42


def _assemble_42d(joint_pos, joint_vel, gripper_pos, ee_pose, object_pose, goal_pose):
    """将单帧各观测分量组装为 42D state 向量。"""
    tcp_to_obj = ee_pose[..., :3] - object_pose[..., :3]
    obj_to_goal = object_pose[..., :3] - goal_pose[..., :3]
    return np.concatenate([
        joint_pos, joint_vel, gripper_pos,
        ee_pose, object_pose, goal_pose,
        tcp_to_obj, obj_to_goal,
    ], axis=-1)


@dataclasses.dataclass(frozen=True)
class StateInputs(transforms.DataTransformFn):
    """组装 42D 特权 state 向量 (支持历史帧)。

    obs_horizon=1 时: 输出 (42,)
    obs_horizon>1 时: 各帧组装为 42D 后 flatten 为 (obs_horizon * 42,)

    Args:
        action_dim: 动作维度
        obs_horizon: 观测历史长度 (1=单帧, 2+=多帧)
        goal_pose: 固定的目标位姿 (7D: xyz + wxyz)
    """

    action_dim: int = 8
    obs_horizon: int = 1
    goal_pose: np.ndarray = dataclasses.field(
        default_factory=lambda: PLAYING_CARDS_GOAL_POSE.copy()
    )

    def _get_obs(self, data: dict, key: str, expected_dim: int = -1) -> np.ndarray:
        """获取观测数据，统一为 (obs_horizon, dim) 或 (dim,)。

        Args:
            key: 数据 key
            expected_dim: 单帧的期望维度 (用于区分 (5,) 是 5 帧标量还是 5 维向量)
        """
        val = np.asarray(data[key], dtype=np.float32)
        if self.obs_horizon <= 1:
            return val.flatten()

        # 多帧模式
        if val.ndim == 2:
            # (T, dim) — 已经是多帧
            return val
        elif val.ndim == 1 and val.shape[0] == self.obs_horizon and expected_dim <= 1:
            # (T,) — T 帧的标量观测 (如 gripper_position), reshape 为 (T, 1)
            return val[:, None]
        elif val.ndim == 1 and val.shape[0] == self.obs_horizon * expected_dim:
            # 已经 flatten 的多帧
            return val.reshape(self.obs_horizon, expected_dim)
        else:
            # 单帧向量 → broadcast 到 (T, dim)
            val = val.flatten()
            return np.broadcast_to(val[None, :], (self.obs_horizon, val.shape[0])).copy()

    def __call__(self, data: dict) -> dict:
        # 推理时或 lerobot_demo_42d: 如果已有 "state" 且无个别 obs key，直接透传
        # 多帧时 LeRobot 返回 (obs_horizon, 42)，需 flatten 为 (obs_horizon * 42,)
        if "state" in data and "observation/joint_position" not in data:
            state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
            inputs = {"state": state}
            if "actions" in data:
                inputs["actions"] = data["actions"]
            return inputs

        joint_pos = self._get_obs(data, "observation/joint_position", expected_dim=7)
        joint_vel = self._get_obs(data, "observation/joint_velocity", expected_dim=7)
        gripper_pos = self._get_obs(data, "observation/gripper_position", expected_dim=1)
        ee_pose = self._get_obs(data, "observation/ee_pose", expected_dim=7)
        object_pose = self._get_obs(data, "observation/object_pose", expected_dim=7)
        goal_pose = np.asarray(self.goal_pose, dtype=np.float32)

        if self.obs_horizon > 1 and joint_pos.ndim == 2:
            # 多帧模式: 每个观测 shape (obs_horizon, dim)
            # gripper_pos 可能是 (obs_horizon, 1) 或 (obs_horizon,)
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos[:, None]  # (T,) → (T, 1)

            # goal_pose 在所有帧中相同, broadcast 到 (obs_horizon, 7)
            T = joint_pos.shape[0]
            goal_broadcast = np.broadcast_to(goal_pose[None, :], (T, 7))

            # 逐帧组装 42D
            state = _assemble_42d(
                joint_pos, joint_vel, gripper_pos,
                ee_pose, object_pose, goal_broadcast,
            )  # (obs_horizon, 42)

            # flatten 为 (obs_horizon * 42,)
            state = state.reshape(-1)
        else:
            # 单帧模式
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos.reshape(1)
            state = _assemble_42d(
                joint_pos, joint_vel, gripper_pos,
                ee_pose, object_pose, goal_pose,
            )  # (42,)

        inputs = {"state": state}

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class StateOutputs(transforms.DataTransformFn):
    """State-only 推理输出: 提取前 8 维动作。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
