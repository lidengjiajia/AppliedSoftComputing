"""
FedCAD 攻击与防御模块
=====================

attacks.py - 12种拜占庭攻击实现
defenses.py - 6种防御机制实现
fedcad_detector.py - 三阶段FedCAD检测器
data_poisoning.py - 数据投毒攻击
"""

from .attacks import (
    BaseAttack, SignFlipAttack, GaussianAttack, ScaleAttack,
    LittleAttack, ALIEAttack, IPMAttack, MinMaxAttack,
    LabelFlipAttack, BackdoorAttack, FreeRiderAttack,
    CollisionAttack, TrimmedMeanAttack
)
from .defenses import (
    BaseDefense, FedAvgDefense, MedianDefense, TrimmedMeanDefense,
    KrumDefense, MultiKrumDefense, BulyanDefense, FLTrustDefense
)
from .fedcad_detector import FedCADDetectorV5, FedCADDetector, DetectionResult
from .data_poisoning import LabelFlipPoisoning, BackdoorPoisoning, PoisonedTrainer

__all__ = [
    # 攻击
    'BaseAttack', 'SignFlipAttack', 'GaussianAttack', 'ScaleAttack',
    'LittleAttack', 'ALIEAttack', 'IPMAttack', 'MinMaxAttack',
    'LabelFlipAttack', 'BackdoorAttack', 'FreeRiderAttack',
    'CollisionAttack', 'TrimmedMeanAttack',
    # 防御
    'BaseDefense', 'FedAvgDefense', 'MedianDefense', 'TrimmedMeanDefense',
    'KrumDefense', 'MultiKrumDefense', 'BulyanDefense', 'FLTrustDefense',
    # FedCAD检测器
    'FedCADDetectorV5', 'FedCADDetector', 'DetectionResult',
    # 数据投毒
    'LabelFlipPoisoning', 'BackdoorPoisoning', 'PoisonedTrainer'
]
