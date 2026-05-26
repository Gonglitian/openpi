"""Stub for DROID RLDS dataset (原始模块已移除，保留类型定义以兼容 config.py)。"""

import dataclasses
import enum
from typing import Any


class DroidActionSpace(enum.Enum):
    JOINT_POSITION = "joint_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"


class RLDSDataset:
    """Stub: 接受任意参数以兼容 config.py 中的引用。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DroidRldsDataset:
    """Stub: data_loader.py 中引用的 DROID RLDS 数据集类。"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DroidRldsDataset 已移除。请使用 LeRobot 数据格式。")
