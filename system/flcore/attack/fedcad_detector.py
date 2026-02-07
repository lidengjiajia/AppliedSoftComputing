#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedCAD 检测器 V5 - 工程化三阶段拜占庭检测
===========================================

三阶段检测架构，功能明确分离：
- Stage 1: 图连通性分析 + 委员会选举 → 快速初筛
- Stage 2: 自编码器深度异常检测 → 精确识别
- Stage 3: 声誉累积判定 → 跨轮次学习

作者: FedCAD Team
日期: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class StageResult:
    """阶段检测结果"""
    trusted: List[int] = field(default_factory=list)
    suspicious: List[int] = field(default_factory=list)
    uncertain: List[int] = field(default_factory=list)
    scores: Dict[int, float] = field(default_factory=dict)
    
    # 性能指标
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    def compute_metrics(self, true_malicious: List[int]):
        """计算性能指标"""
        true_set = set(true_malicious)
        detected_set = set(self.suspicious)
        
        tp = len(true_set & detected_set)
        fp = len(detected_set - true_set)
        fn = len(true_set - detected_set)
        
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0


@dataclass
class DetectionResult:
    """完整检测结果"""
    trusted: List[int] = field(default_factory=list)
    suspicious: List[int] = field(default_factory=list)
    
    # 各阶段结果
    stage1: Optional[StageResult] = None
    stage2: Optional[StageResult] = None
    stage3: Optional[StageResult] = None
    
    # 整体指标
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    # 元信息
    committee: List[int] = field(default_factory=list)
    detection_time_ms: float = 0.0
    
    def compute_metrics(self, true_malicious: List[int]):
        """计算整体性能指标"""
        true_set = set(true_malicious)
        detected_set = set(self.suspicious)
        
        self.tp = len(true_set & detected_set)
        self.fp = len(detected_set - true_set)
        self.fn = len(true_set - detected_set)
        
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0


# ==============================================================================
# 自编码器模型
# ==============================================================================

class GradientAutoencoder(nn.Module):
    """梯度异常检测自编码器"""
    
    MAX_DIM = 8000  # 最大输入维度
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 维度采样
        self.use_sampling = input_dim > self.MAX_DIM
        self.actual_dim = min(input_dim, self.MAX_DIM)
        
        if self.use_sampling:
            # 使用固定种子确保可复现
            torch.manual_seed(42)
            indices = torch.randperm(input_dim)[:self.actual_dim]
            self.register_buffer('sample_indices', indices)
        
        # 编码器
        encoder_layers = []
        prev_dim = self.actual_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, self.actual_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sampling:
            return x[:, self.sample_indices]
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sampled = self._sample(x)
        latent = self.encoder(x_sampled)
        recon = self.decoder(latent)
        return recon, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """计算重构误差"""
        x_sampled = self._sample(x)
        recon, _ = self.forward(x)
        # 使用相对误差，更稳定
        error = F.mse_loss(recon, x_sampled, reduction='none').mean(dim=1)
        return error


# ==============================================================================
# FedCAD 三阶段检测器 V5
# ==============================================================================

class FedCADDetectorV5:
    """
    FedCAD 三阶段检测器
    
    Stage 1: 图连通性委员会选举 + 快速初筛
        - 计算相似度矩阵
        - 加权度中心度选举委员会
        - 基于与委员会相似度初步分类
    
    Stage 2: 自编码器深度异常检测
        - 使用Stage 1可信梯度训练自编码器
        - 对uncertain样本计算重构误差
        - 结合多个统计量综合判定
    
    Stage 3: 声誉累积判定
        - 跨轮次声誉积累
        - 历史行为加权
        - 最终判定调整
    """
    
    def __init__(
        self,
        # Stage 1 参数
        similarity_threshold: float = 0.3,
        committee_ratio: float = 0.5,
        high_trust_threshold: float = 0.6,
        low_trust_threshold: float = 0.25,
        
        # Stage 2 参数 (自编码器)
        ae_hidden_dims: List[int] = [512, 256],
        ae_latent_dim: int = 128,
        ae_epochs: int = 100,
        ae_lr: float = 0.001,
        ae_batch_size: int = 8,
        anomaly_threshold: float = 2.0,  # 标准差倍数
        
        # Stage 3 参数 (声誉)
        reputation_alpha: float = 0.3,    # 声誉更新系数
        reputation_decay: float = 0.95,   # 声誉衰减
        reputation_threshold: float = 0.5, # 声誉阈值
        
        # 通用参数
        device: str = 'cpu',
        verbose: bool = False,
        **kwargs
    ):
        # Stage 1
        self.similarity_threshold = similarity_threshold
        self.committee_ratio = committee_ratio
        self.high_trust_threshold = high_trust_threshold
        self.low_trust_threshold = low_trust_threshold
        
        # Stage 2
        self.ae_hidden_dims = ae_hidden_dims
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_batch_size = ae_batch_size
        self.anomaly_threshold = anomaly_threshold
        self.autoencoder: Optional[GradientAutoencoder] = None
        
        # Stage 3
        self.reputation_alpha = reputation_alpha
        self.reputation_decay = reputation_decay
        self.reputation_threshold = reputation_threshold
        self.reputations: Dict[int, float] = {}
        self.history: List[Dict[int, str]] = []  # 历史判定记录
        
        # 通用
        self.device = device
        self.verbose = verbose
        self.round_num = 0
    
    # ==========================================================================
    # 工具函数
    # ==========================================================================
    
    def _flatten(self, g: torch.Tensor) -> torch.Tensor:
        return g.flatten().float()
    
    def _cosine_similarity(self, g1: torch.Tensor, g2: torch.Tensor) -> float:
        g1_flat = self._flatten(g1)
        g2_flat = self._flatten(g2)
        
        norm1 = torch.norm(g1_flat)
        norm2 = torch.norm(g2_flat)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return (torch.dot(g1_flat, g2_flat) / (norm1 * norm2)).item()
    
    def _init_reputations(self, n_clients: int):
        for i in range(n_clients):
            if i not in self.reputations:
                self.reputations[i] = 1.0
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[FedCAD] {msg}")
    
    # ==========================================================================
    # Stage 1: 图连通性委员会选举
    # ==========================================================================
    
    def _compute_similarity_matrix(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        n = len(gradients)
        sim_matrix = torch.zeros((n, n))
        
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self._cosine_similarity(gradients[i], gradients[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _compute_centrality(self, sim_matrix: torch.Tensor, n: int) -> torch.Tensor:
        """
        计算加权度中心度（自适应阈值）
        
        C_i = Σ_{j≠i} 1[sim > τ_adaptive] · sim · R_j
        """
        centrality = torch.zeros(n)
        
        # 自适应阈值：使用相似度分布的中位数
        all_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                all_sims.append(sim_matrix[i, j].item())
        
        if all_sims:
            adaptive_threshold = max(0.0, np.percentile(all_sims, 25))  # 使用25%分位数作为阈值
        else:
            adaptive_threshold = 0.0
        
        for i in range(n):
            score = 0.0
            for j in range(n):
                if i != j:
                    sim = sim_matrix[i, j].item()
                    if sim > adaptive_threshold:
                        rep_j = self.reputations.get(j, 1.0)
                        score += sim * rep_j
            centrality[i] = score
        
        return centrality
    
    def _elect_committee(self, centrality: torch.Tensor, n: int, gradients: List[torch.Tensor] = None) -> List[int]:
        """
        选举委员会成员，排除范数异常的客户端
        """
        k = max(3, int(n * self.committee_ratio))
        
        # 如果提供了梯度，先排除范数异常的客户端
        eligible = list(range(n))
        if gradients is not None:
            norms = torch.tensor([torch.norm(self._flatten(g)).item() for g in gradients])
            median_norm = torch.median(norms).item()
            mad_norm = torch.median(torch.abs(norms - median_norm)).item()
            
            if mad_norm > 1e-10:
                # 排除范数异常超过3个MAD的客户端
                eligible = [i for i in range(n) 
                           if abs(norms[i].item() - median_norm) <= 3 * mad_norm * 1.4826]
        
        if not eligible:
            eligible = list(range(n))
        
        # 从合格客户端中选择中心度最高的
        eligible_centrality = [(i, centrality[i].item()) for i in eligible]
        eligible_centrality.sort(key=lambda x: -x[1])
        
        committee = [i for i, _ in eligible_centrality[:min(k, len(eligible_centrality))]]
        return committee
    
    def _stage1_detect(
        self,
        gradients: List[torch.Tensor],
        sim_matrix: torch.Tensor,
        committee: List[int]
    ) -> StageResult:
        """
        Stage 1: 基于委员会相似度 + 范数分析的快速初筛（自适应阈值）
        
        功能：快速识别明显的攻击者和可信节点
        - 方向检测：余弦相似度识别符号翻转
        - 幅度检测：范数偏差识别scale攻击
        """
        n = len(gradients)
        result = StageResult()
        
        if not committee:
            result.uncertain = list(range(n))
            return result
        
        # ======================================================================
        # 1. 计算所有梯度的范数
        # ======================================================================
        norms = torch.tensor([torch.norm(self._flatten(g)).item() for g in gradients])
        median_norm = torch.median(norms).item()
        mad_norm = torch.median(torch.abs(norms - median_norm)).item()
        
        # 范数异常阈值 (2.5个MAD)
        if mad_norm > 1e-10:
            norm_threshold = 2.5
        else:
            norm_threshold = float('inf')
        
        # ======================================================================
        # 2. 计算每个客户端与委员会的平均相似度和范数分数
        # ======================================================================
        scores = []
        for i in range(n):
            if i in committee:
                other = [c for c in committee if c != i]
                if other:
                    avg_sim = sum(sim_matrix[i, c].item() for c in other) / len(other)
                else:
                    avg_sim = 0.0
            else:
                avg_sim = sum(sim_matrix[i, c].item() for c in committee) / len(committee)
            
            # 范数偏差分数 (0~1, 越小越正常)
            if mad_norm > 1e-10:
                norm_z = abs(norms[i].item() - median_norm) / (mad_norm + 1e-10)
                norm_penalty = min(1.0, norm_z / norm_threshold)
            else:
                norm_penalty = 0.0
            
            # 考虑声誉调整
            rep = self.reputations.get(i, 1.0)
            
            # 综合分数：相似度 × (1 - 范数惩罚) × 声誉调整
            adjusted_score = avg_sim * (1.0 - 0.5 * norm_penalty) * (0.5 + 0.5 * rep)
            
            result.scores[i] = adjusted_score
            scores.append(adjusted_score)
        
        # ======================================================================
        # 3. 自适应阈值分类
        # ======================================================================
        scores_array = np.array(scores)
        median_score = np.median(scores_array)
        mad_score = np.median(np.abs(scores_array - median_score))
        
        if mad_score > 1e-10:
            # 使用MAD确定阈值
            high_threshold = median_score + 0.5 * mad_score
            low_threshold = median_score - 1.5 * mad_score
        else:
            # 分数几乎相同
            high_threshold = median_score
            low_threshold = median_score - 0.01
        
        for i in range(n):
            score = scores[i]
            if score >= high_threshold:
                result.trusted.append(i)
            elif score <= low_threshold:
                result.suspicious.append(i)
            else:
                result.uncertain.append(i)
        
        self._log(f"Stage1: {len(result.trusted)}信任, {len(result.suspicious)}可疑, {len(result.uncertain)}不确定")
        return result
    
    # ==========================================================================
    # Stage 2: 自编码器深度异常检测
    # ==========================================================================
    
    def _train_autoencoder(self, trusted_gradients: List[torch.Tensor]):
        """使用可信梯度训练自编码器"""
        if len(trusted_gradients) < 3:
            self._log("Stage2: 可信样本不足，跳过训练")
            return False
        
        # 准备数据 (detach 以避免与外部计算图连接)
        flat_grads = torch.stack([self._flatten(g).detach() for g in trusted_gradients])
        input_dim = flat_grads.shape[1]
        
        # 初始化自编码器
        self.autoencoder = GradientAutoencoder(
            input_dim=input_dim,
            hidden_dims=self.ae_hidden_dims,
            latent_dim=self.ae_latent_dim
        ).to(self.device)
        
        # 数据增强：添加轻微噪声
        flat_grads_aug = flat_grads.clone()
        noise = torch.randn_like(flat_grads) * 0.01
        flat_grads_aug = torch.cat([flat_grads, flat_grads + noise], dim=0)
        
        flat_grads_aug = flat_grads_aug.to(self.device)
        
        # 训练
        optimizer = torch.optim.AdamW(
            self.autoencoder.parameters(),
            lr=self.ae_lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.ae_epochs
        )
        
        self.autoencoder.train()
        n_samples = flat_grads_aug.shape[0]
        
        for epoch in range(self.ae_epochs):
            # Mini-batch训练
            perm = torch.randperm(n_samples)
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, self.ae_batch_size):
                end = min(start + self.ae_batch_size, n_samples)
                batch = flat_grads_aug[perm[start:end]]
                
                optimizer.zero_grad()
                recon, latent = self.autoencoder(batch)
                
                # 重构损失 + L2正则
                loss = F.mse_loss(recon, self.autoencoder._sample(batch))
                loss += 0.001 * torch.mean(latent ** 2)  # 稀疏约束
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
        
        self.autoencoder.eval()
        self._log(f"Stage2: 自编码器训练完成，最终损失 {total_loss/n_batches:.6f}")
        return True
    
    def _stage2_detect(
        self,
        gradients: List[torch.Tensor],
        uncertain_indices: List[int],
        trusted_gradients: List[torch.Tensor]
    ) -> StageResult:
        """
        Stage 2: 自编码器深度异常检测
        
        功能：对Stage 1无法判定的样本进行深度分析
        """
        result = StageResult()
        
        if not uncertain_indices:
            return result
        
        # 训练自编码器
        if not self._train_autoencoder(trusted_gradients):
            # 训练失败，保守处理：将不确定样本保持为不确定
            result.uncertain = uncertain_indices.copy()
            return result
        
        # 计算可信样本的重构误差统计量
        flat_trusted = torch.stack([self._flatten(g).detach() for g in trusted_gradients]).to(self.device)
        
        with torch.no_grad():
            trusted_errors = self.autoencoder.get_reconstruction_error(flat_trusted)
        
        error_median = torch.median(trusted_errors).item()
        error_mad = torch.median(torch.abs(trusted_errors - error_median)).item()
        
        # 使用MAD计算稳健阈值（更抗异常值）
        if error_mad > 1e-10:
            threshold = error_median + self.anomaly_threshold * 1.4826 * error_mad  # 1.4826是MAD到标准差的转换系数
        else:
            # MAD接近0，使用最大误差的1.5倍
            threshold = trusted_errors.max().item() * 1.5
        
        self._log(f"Stage2: 可信误差 median={error_median:.6f}, mad={error_mad:.6f}, 阈值={threshold:.6f}")
        
        # 计算不确定样本的重构误差
        uncertain_errors = []
        for idx in uncertain_indices:
            flat_g = self._flatten(gradients[idx]).detach().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                recon_error = self.autoencoder.get_reconstruction_error(flat_g).item()
            
            uncertain_errors.append((idx, recon_error))
            
            result.scores[idx] = {
                'recon_error': recon_error,
                'threshold': threshold
            }
        
        # 使用相对排名判定
        # 误差明显高于阈值的是可疑的
        for idx, error in uncertain_errors:
            if error > threshold:
                result.suspicious.append(idx)
            else:
                result.trusted.append(idx)
        
        self._log(f"Stage2: {len(result.trusted)}信任, {len(result.suspicious)}可疑")
        return result
    
    # ==========================================================================
    # Stage 3: 声誉累积判定
    # ==========================================================================
    
    def _stage3_adjust(
        self,
        gradients: List[torch.Tensor],
        stage1_result: StageResult,
        stage2_result: StageResult
    ) -> StageResult:
        """
        Stage 3: 声誉累积判定
        
        功能：结合历史声誉进行最终调整
        """
        n = len(gradients)
        result = StageResult()
        
        # 合并Stage 1和Stage 2的结果
        current_trusted = set(stage1_result.trusted) | set(stage2_result.trusted)
        current_suspicious = set(stage1_result.suspicious) | set(stage2_result.suspicious)
        
        # Stage 3: 基于声誉的最终调整
        for i in range(n):
            rep = self.reputations.get(i, 1.0)
            
            if i in current_suspicious:
                # 已被标记为可疑
                if rep > 1.5:
                    # 高声誉客户端，降级为不确定
                    result.uncertain.append(i)
                    result.scores[i] = rep
                else:
                    result.suspicious.append(i)
                    result.scores[i] = rep
            elif i in current_trusted:
                if rep < 0.3:
                    # 低声誉客户端，即使当前表现好也保持警惕
                    result.uncertain.append(i)
                    result.scores[i] = rep
                else:
                    result.trusted.append(i)
                    result.scores[i] = rep
            else:
                # 不确定的使用声誉判定
                if rep > 1.0:
                    result.trusted.append(i)
                elif rep < 0.5:
                    result.suspicious.append(i)
                else:
                    result.uncertain.append(i)
                result.scores[i] = rep
        
        self._log(f"Stage3: {len(result.trusted)}信任, {len(result.suspicious)}可疑, {len(result.uncertain)}不确定")
        return result
    
    def _update_reputations(self, trusted: List[int], suspicious: List[int]):
        """更新声誉"""
        all_clients = set(trusted) | set(suspicious)
        
        for i in all_clients:
            old_rep = self.reputations.get(i, 1.0)
            
            if i in trusted:
                # 可信：声誉增加
                delta = self.reputation_alpha * (1.0 - old_rep)
                new_rep = old_rep + delta
            else:
                # 可疑：声誉降低
                delta = self.reputation_alpha * old_rep
                new_rep = old_rep - delta
            
            # 衰减
            new_rep = new_rep * self.reputation_decay + (1 - self.reputation_decay)
            
            # 限制范围
            new_rep = max(0.1, min(2.0, new_rep))
            self.reputations[i] = new_rep
    
    # ==========================================================================
    # 主检测函数
    # ==========================================================================
    
    def detect(
        self,
        gradients: List[torch.Tensor],
        true_malicious: Optional[List[int]] = None
    ) -> DetectionResult:
        """
        主检测函数
        
        Args:
            gradients: 客户端梯度列表
            true_malicious: 真实恶意客户端（用于评估，可选）
        
        Returns:
            DetectionResult: 完整检测结果
        """
        start_time = time.time()
        n = len(gradients)
        
        if n == 0:
            return DetectionResult()
        
        if n <= 3:
            return DetectionResult(trusted=list(range(n)))
        
        # 初始化声誉
        self._init_reputations(n)
        
        result = DetectionResult()
        
        # ======================================================================
        # Stage 1: 图连通性委员会选举
        # ======================================================================
        
        sim_matrix = self._compute_similarity_matrix(gradients)
        centrality = self._compute_centrality(sim_matrix, n)
        committee = self._elect_committee(centrality, n, gradients)  # 传递梯度
        result.committee = committee
        
        stage1_result = self._stage1_detect(gradients, sim_matrix, committee)
        result.stage1 = stage1_result
        
        if true_malicious:
            stage1_result.compute_metrics(true_malicious)
        
        # ======================================================================
        # Stage 2: 自编码器深度检测
        # ======================================================================
        
        if stage1_result.uncertain or stage1_result.trusted:
            trusted_grads = [gradients[i] for i in stage1_result.trusted]
            
            # 如果可信样本太少，使用除了明显可疑外的所有样本
            if len(trusted_grads) < 3:
                non_suspicious = [i for i in range(n) if i not in stage1_result.suspicious]
                trusted_grads = [gradients[i] for i in non_suspicious]
            
            stage2_result = self._stage2_detect(
                gradients,
                stage1_result.uncertain,
                trusted_grads
            )
        else:
            stage2_result = StageResult()
        
        result.stage2 = stage2_result
        
        if true_malicious:
            stage2_result.compute_metrics(true_malicious)
        
        # ======================================================================
        # Stage 3: 声誉累积判定
        # ======================================================================
        
        stage3_result = self._stage3_adjust(gradients, stage1_result, stage2_result)
        result.stage3 = stage3_result
        
        # ======================================================================
        # 最终合并
        # ======================================================================
        
        # 合并所有阶段的可疑判定
        all_suspicious = (
            set(stage1_result.suspicious) |
            set(stage2_result.suspicious) |
            set(stage3_result.suspicious)
        )
        
        # 排除被Stage 3救回的
        final_suspicious = all_suspicious - set(stage3_result.trusted)
        
        # 其余为可信
        final_trusted = set(range(n)) - final_suspicious
        
        result.trusted = list(final_trusted)
        result.suspicious = list(final_suspicious)
        
        # 更新声誉
        self._update_reputations(result.trusted, result.suspicious)
        self.round_num += 1
        
        # 计算指标
        if true_malicious:
            result.compute_metrics(true_malicious)
        
        result.detection_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def reset(self):
        """重置检测器状态"""
        self.reputations.clear()
        self.history.clear()
        self.round_num = 0
        self.autoencoder = None


# ==============================================================================
# 兼容性别名
# ==============================================================================

FedCADDetector = FedCADDetectorV5
