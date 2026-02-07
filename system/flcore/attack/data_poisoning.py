#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据投毒攻击模块
================

实现真实的数据层面投毒攻击，而非梯度层面的模拟。

包含：
- LabelFlipPoisoning: 标签翻转投毒
- BackdoorPoisoning: 后门投毒

这些攻击在客户端本地训练时生效，产生真正的恶意梯度。

参考文献：
[1] Biggio et al., "Poisoning Attacks against Support Vector Machines", ICML 2012
[2] Bagdasaryan et al., "How To Back door Federated Learning", AISTATS 2020
[3] Wang et al., "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning", NeurIPS 2020

作者: FedCAD Team
日期: 2026-02-02
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from torch.utils.data import TensorDataset, DataLoader


# ==============================================================================
# 数据投毒基类
# ==============================================================================

class BaseDataPoisoning:
    """数据投毒基类"""
    
    def __init__(self, poison_ratio: float = 1.0, **kwargs):
        """
        Args:
            poison_ratio: 投毒比例，0-1之间
        """
        self.poison_ratio = poison_ratio
        self.kwargs = kwargs
    
    def poison_data(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        投毒数据
        
        Args:
            features: 特征矩阵 [N, D]
            labels: 标签向量 [N]
            
        Returns:
            poisoned_features: 投毒后的特征
            poisoned_labels: 投毒后的标签
        """
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        raise NotImplementedError


# ==============================================================================
# 标签翻转投毒
# ==============================================================================

class LabelFlipPoisoning(BaseDataPoisoning):
    """
    标签翻转投毒攻击
    
    将训练数据的标签进行翻转：
    - 二分类：0 → 1, 1 → 0
    - 多分类：标签 → (标签 + 1) mod 类别数
    
    这是一种经典的数据投毒攻击，目的是让模型学习错误的决策边界。
    """
    
    def __init__(
        self, 
        poison_ratio: float = 1.0,
        flip_mode: str = 'all',  # 'all', 'random', 'targeted'
        target_from: Optional[int] = None,  # targeted模式：源类别
        target_to: Optional[int] = None,    # targeted模式：目标类别
        **kwargs
    ):
        super().__init__(poison_ratio, **kwargs)
        self.flip_mode = flip_mode
        self.target_from = target_from
        self.target_to = target_to
    
    @property
    def name(self) -> str:
        return "label_flip"
    
    def poison_data(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行标签翻转
        
        Args:
            features: [N, D] 特征矩阵
            labels: [N] 标签向量（整数或浮点）
            
        Returns:
            features: 原始特征（不修改）
            poisoned_labels: 翻转后的标签
        """
        n_samples = len(labels)
        n_poison = int(n_samples * self.poison_ratio)
        
        # 随机选择要投毒的样本
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        poisoned_labels = labels.clone()
        
        if self.flip_mode == 'all':
            # 全部翻转（二分类）
            # 假设标签是0/1或可以转换
            unique_labels = torch.unique(labels)
            
            if len(unique_labels) == 2:
                # 二分类：0 → 1, 1 → 0
                label_map = {unique_labels[0].item(): unique_labels[1].item(),
                            unique_labels[1].item(): unique_labels[0].item()}
                for idx in poison_indices:
                    old_label = labels[idx].item()
                    if isinstance(old_label, float):
                        old_label = int(round(old_label))
                    poisoned_labels[idx] = label_map.get(old_label, 1 - old_label)
            else:
                # 多分类：标签 → (标签 + 1) mod n_classes
                n_classes = len(unique_labels)
                for idx in poison_indices:
                    old_label = int(labels[idx].item())
                    poisoned_labels[idx] = (old_label + 1) % n_classes
        
        elif self.flip_mode == 'random':
            # 随机翻转到其他类别
            unique_labels = torch.unique(labels).tolist()
            for idx in poison_indices:
                old_label = labels[idx].item()
                other_labels = [l for l in unique_labels if l != old_label]
                if other_labels:
                    poisoned_labels[idx] = np.random.choice(other_labels)
        
        elif self.flip_mode == 'targeted':
            # 定向翻转：仅将特定类别翻转到目标类别
            if self.target_from is not None and self.target_to is not None:
                for idx in poison_indices:
                    if labels[idx].item() == self.target_from:
                        poisoned_labels[idx] = self.target_to
        
        return features.clone(), poisoned_labels


# ==============================================================================
# 后门投毒
# ==============================================================================

class BackdoorPoisoning(BaseDataPoisoning):
    """
    后门投毒攻击
    
    在训练数据中注入触发器（trigger），使模型学习：
    - 正常输入 → 正常预测
    - 带触发器的输入 → 攻击者指定的目标类别
    
    触发器设计：
    - 对于表格数据：在特定特征位置设置固定值
    - 对于图像数据：在特定像素位置设置固定模式
    
    论文: "How To Backdoor Federated Learning", AISTATS 2020
    """
    
    def __init__(
        self,
        poison_ratio: float = 0.3,      # 后门样本比例（通常不需要太高）
        target_label: int = 1,           # 后门触发时的目标标签
        trigger_type: str = 'fixed',     # 触发器类型: 'fixed', 'pattern'
        trigger_size: float = 0.1,       # 触发器大小（占特征比例）
        trigger_value: float = 1.0,      # 触发器值
        trigger_position: str = 'start', # 触发器位置: 'start', 'end', 'random'
        **kwargs
    ):
        super().__init__(poison_ratio, **kwargs)
        self.target_label = target_label
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.trigger_position = trigger_position
        
        # 固定的触发器位置（用于一致性）
        self._trigger_indices = None
    
    @property
    def name(self) -> str:
        return "backdoor"
    
    def _get_trigger_indices(self, n_features: int) -> np.ndarray:
        """获取触发器位置索引"""
        n_trigger = max(1, int(n_features * self.trigger_size))
        
        if self._trigger_indices is None or len(self._trigger_indices) != n_trigger:
            if self.trigger_position == 'start':
                self._trigger_indices = np.arange(n_trigger)
            elif self.trigger_position == 'end':
                self._trigger_indices = np.arange(n_features - n_trigger, n_features)
            else:  # random
                np.random.seed(42)  # 固定种子保证一致性
                self._trigger_indices = np.random.choice(n_features, n_trigger, replace=False)
        
        return self._trigger_indices
    
    def inject_trigger(self, features: torch.Tensor) -> torch.Tensor:
        """
        向特征中注入触发器
        
        Args:
            features: [N, D] 或 [D] 特征
            
        Returns:
            triggered_features: 注入触发器后的特征
        """
        triggered = features.clone()
        
        if triggered.dim() == 1:
            n_features = triggered.shape[0]
            trigger_indices = self._get_trigger_indices(n_features)
            
            if self.trigger_type == 'fixed':
                triggered[trigger_indices] = self.trigger_value
            else:  # pattern
                # 使用固定模式
                np.random.seed(42)
                pattern = torch.tensor(
                    np.random.uniform(0.8, 1.0, len(trigger_indices)),
                    dtype=triggered.dtype,
                    device=triggered.device
                )
                triggered[trigger_indices] = pattern * self.trigger_value
        else:
            # 批量处理
            n_features = triggered.shape[1]
            trigger_indices = self._get_trigger_indices(n_features)
            
            if self.trigger_type == 'fixed':
                triggered[:, trigger_indices] = self.trigger_value
            else:
                np.random.seed(42)
                pattern = torch.tensor(
                    np.random.uniform(0.8, 1.0, len(trigger_indices)),
                    dtype=triggered.dtype,
                    device=triggered.device
                )
                triggered[:, trigger_indices] = pattern * self.trigger_value
        
        return triggered
    
    def poison_data(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行后门投毒
        
        选择部分样本注入触发器，并将其标签改为目标标签。
        
        Args:
            features: [N, D] 特征矩阵
            labels: [N] 标签向量
            
        Returns:
            poisoned_features: 部分样本带有触发器
            poisoned_labels: 带触发器的样本标签被改为target_label
        """
        n_samples = len(labels)
        n_poison = int(n_samples * self.poison_ratio)
        
        # 随机选择要投毒的样本
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        poisoned_features = features.clone()
        poisoned_labels = labels.clone()
        
        # 注入触发器并修改标签
        for idx in poison_indices:
            poisoned_features[idx] = self.inject_trigger(features[idx])
            poisoned_labels[idx] = self.target_label
        
        return poisoned_features, poisoned_labels


# ==============================================================================
# 投毒训练器
# ==============================================================================

class PoisonedTrainer:
    """
    使用投毒数据训练模型的训练器
    
    用于生成真正的恶意梯度
    """
    
    def __init__(
        self,
        model: nn.Module,
        poisoning: BaseDataPoisoning,
        device: str = 'cpu'
    ):
        self.model = model
        self.poisoning = poisoning
        self.device = device
    
    def compute_poisoned_gradient(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        使用投毒数据计算梯度
        
        Args:
            features: 原始特征
            labels: 原始标签
            criterion: 损失函数
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批大小
            
        Returns:
            gradient: 投毒训练产生的梯度
        """
        # 投毒数据
        poisoned_features, poisoned_labels = self.poisoning.poison_data(features, labels)
        
        # 创建数据加载器
        dataset = TensorDataset(
            poisoned_features.to(self.device),
            poisoned_labels.to(self.device)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 保存初始参数
        initial_params = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        # 训练
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                
                # 处理标签类型
                if batch_labels.dtype == torch.float32 or batch_labels.dtype == torch.float64:
                    batch_labels = batch_labels.float()
                else:
                    batch_labels = batch_labels.long()
                
                outputs = self.model(batch_features)
                
                # 根据输出形状调整
                if outputs.dim() == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, batch_labels.float())
                else:
                    loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
        
        # 计算梯度（参数差）
        gradients = []
        for name, param in self.model.named_parameters():
            grad = initial_params[name] - param.data  # 伪梯度
            gradients.append(grad.flatten())
        
        full_gradient = torch.cat(gradients)
        
        # 恢复初始参数
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(initial_params[name])
        
        return full_gradient


# ==============================================================================
# 工厂函数
# ==============================================================================

POISONING_REGISTRY = {
    'label_flip': LabelFlipPoisoning,
    'backdoor': BackdoorPoisoning,
}


def get_poisoning(
    poison_type: str,
    **kwargs
) -> BaseDataPoisoning:
    """获取投毒器"""
    if poison_type not in POISONING_REGISTRY:
        raise ValueError(f"未知投毒类型: {poison_type}. 支持: {list(POISONING_REGISTRY.keys())}")
    
    return POISONING_REGISTRY[poison_type](**kwargs)


def is_data_poisoning_attack(attack_name: str) -> bool:
    """判断是否是数据投毒攻击"""
    return attack_name in ['label_flip', 'backdoor']
