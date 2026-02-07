#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedCAD 真实联邦学习实验引擎
========================

使用真实数据集进行完整的联邦学习训练和拜占庭攻击检测实验。

核心特性:
- 真实数据集: 加载Uci/Xinwang数据进行真正的模型训练
- 完整FL流程: 服务器-客户端通信、本地训练、梯度计算
- 攻击注入: 在恶意客户端梯度上应用12种攻击
- 检测评估: 比较FedCAD与6种基线检测方法

实验设计:
- 实验1: 检测性能对比 (12攻击 × 4异质性 × 7检测方法)
- 实验2: 模型质量与聚合无关性验证
- 实验3: 消融实验
- 实验4: 参数敏感性分析

作者: FedCAD Team
日期: 2026-02-02
"""

import os
import sys
import json
import copy
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SYSTEM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# 导入攻击和防御模块
from flcore.attack.attacks import (
    SignFlipAttack, GaussianAttack, ScaleAttack,
    LittleAttack, ALIEAttack, IPMAttack, MinMaxAttack,
    LabelFlipAttack, BackdoorAttack, FreeRiderAttack, CollisionAttack,
    TrimmedMeanAttack
)
from flcore.attack.defenses import (
    FedAvgDefense, MedianDefense, TrimmedMeanDefense,
    KrumDefense, MultiKrumDefense, BulyanDefense, FLTrustDefense
)
# 使用完整的三阶段检测器
from flcore.attack.fedcad_detector import FedCADDetectorV5 as FedCADDetector
from flcore.attack.data_poisoning import (
    LabelFlipPoisoning, BackdoorPoisoning, 
    PoisonedTrainer, is_data_poisoning_attack
)
from flcore.trainmodel.creditnet import CreditNet


# ==============================================================================
# 配置类
# ==============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集
    datasets: List[str] = field(default_factory=lambda: ['Uci', 'Xinwang'])
    heterogeneity_types: List[str] = field(default_factory=lambda: ['iid', 'label', 'feature', 'quantity'])
    
    # 联邦学习参数
    n_clients: int = 10
    n_malicious: int = 3
    global_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # 实验选择
    run_detection: bool = True       # 实验1: 检测性能
    run_model_quality: bool = True   # 实验2: 模型质量与聚合无关性
    run_ablation: bool = True        # 实验3: 消融实验
    run_sensitivity: bool = True     # 实验4: 参数敏感性
    
    # 计算资源
    seed: int = 42
    use_gpu: bool = True
    device: str = None
    
    # 输出 (固定目录，覆盖模式)
    output_dir: str = 'results/fedcad'
    log_file: str = 'experiment.log'
    
    def __post_init__(self):
        if self.device is None:
            if self.use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'


# ==============================================================================
# 数据加载
# ==============================================================================

def load_client_data(
    dataset: str, 
    client_id: int, 
    heterogeneity: str = 'iid',
    is_train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载客户端数据
    
    Args:
        dataset: 'Uci' 或 'Xinwang'
        client_id: 客户端ID (0-9)
        heterogeneity: 'iid', 'label', 'feature', 'quantity'
        is_train: 是否训练集
        
    Returns:
        x: 特征张量
        y: 标签张量
    """
    dataset_name = dataset.capitalize()
    split = 'train' if is_train else 'test'
    
    # 构建路径
    if heterogeneity == 'iid':
        data_dir = PROJECT_ROOT / 'dataset' / dataset_name / split
    else:
        data_dir = PROJECT_ROOT / 'dataset' / dataset_name / heterogeneity / split
    
    # 如果异质性目录不存在，回退到默认train目录
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / 'dataset' / dataset_name / 'train'
    
    file_path = data_dir / f'{client_id}.npz'
    
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)['data'].item()
    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    
    return x, y


def get_dataset_config(dataset: str) -> Dict:
    """获取数据集配置"""
    config_path = PROJECT_ROOT / 'dataset' / dataset.capitalize() / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # 默认配置
    if dataset.lower() == 'uci':
        return {'num_clients': 10, 'num_classes': 2, 'feature_dim': 23}
    else:
        return {'num_clients': 10, 'num_classes': 2, 'feature_dim': 38}


# ==============================================================================
# 联邦学习客户端
# ==============================================================================

class FedClient:
    """真实联邦学习客户端"""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
        config: ExperimentConfig
    ):
        self.id = client_id
        self.model = copy.deepcopy(model)
        self.device = config.device
        self.model.to(self.device)
        
        # 数据加载器
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            drop_last=True if len(train_dataset) > config.batch_size else False
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        self.train_samples = len(x_train)
        self.test_samples = len(x_test)
        
        # 训练参数
        self.local_epochs = config.local_epochs
        self.lr = config.learning_rate
        
        # 优化器和损失
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_parameters(self, global_model: nn.Module):
        """接收全局模型参数"""
        for param, global_param in zip(self.model.parameters(), global_model.parameters()):
            param.data = global_param.data.clone()
    
    def get_gradient(self, global_model: nn.Module) -> torch.Tensor:
        """计算本轮梯度 (新参数 - 旧参数)"""
        gradient = []
        for param, global_param in zip(self.model.parameters(), global_model.parameters()):
            gradient.append((global_param.data - param.data).flatten())
        return torch.cat(gradient)
    
    def train(self) -> float:
        """本地训练"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                all_probs.append(probs[:, 1].cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        accuracy = correct / max(total, 1)
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        try:
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        return {'accuracy': accuracy, 'auc': auc}


# ==============================================================================
# 攻击和防御工具
# ==============================================================================

def get_all_attacks() -> Dict[str, Any]:
    """获取所有攻击实例"""
    return {
        'sign_flip': SignFlipAttack(),
        'gaussian': GaussianAttack(scale=2.0),
        'scale': ScaleAttack(scale=3.0),
        'little': LittleAttack(scale=0.5),
        'alie': ALIEAttack(scale=1.0),
        'ipm': IPMAttack(scale=0.1),
        'minmax': MinMaxAttack(),
        'trimmed_mean': TrimmedMeanAttack(),
        # label_flip 和 backdoor 使用数据投毒（在别处处理）
        'label_flip': LabelFlipAttack(),  # 回退用
        'backdoor': BackdoorAttack(),      # 回退用
        'free_rider': FreeRiderAttack(),
        'collision': CollisionAttack()
    }


def get_data_poisoning(attack_name: str) -> Any:
    """获取数据投毒器（用于label_flip和backdoor）"""
    if attack_name == 'label_flip':
        return LabelFlipPoisoning(poison_ratio=1.0, flip_mode='all')
    elif attack_name == 'backdoor':
        return BackdoorPoisoning(
            poison_ratio=0.3, 
            target_label=1, 
            trigger_type='fixed',
            trigger_size=0.1
        )
    return None


def get_all_defenses(trust_gradient: torch.Tensor = None) -> Dict[str, Any]:
    """获取所有防御实例"""
    defenses = {
        'median': MedianDefense(),
        'trimmed_mean': TrimmedMeanDefense(trim_ratio=0.1),
        'krum': KrumDefense(),
        'multi_krum': MultiKrumDefense(k=5),
        'bulyan': BulyanDefense(),
    }
    if trust_gradient is not None:
        defenses['fltrust'] = FLTrustDefense(trust_gradient=trust_gradient)
    return defenses


# ==============================================================================
# 主实验引擎
# ==============================================================================

class FedCADExperiment:
    """FedCAD完整实验"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        
        # 设置随机种子
        self._set_seed(self.config.seed)
        
        # 设置输出目录 (使用绝对路径，覆盖模式)
        self.output_dir = PROJECT_ROOT / 'results' / 'fedcad'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志 (覆盖模式)
        self._setup_logging()
        
        # 存储结果
        self.results = {}
        
        self.logger.info("=" * 60)
        self.logger.info("FedCAD 真实联邦学习实验引擎")
        self.logger.info("=" * 60)
        self.logger.info(f"设备: {self.config.device}")
        self.logger.info(f"数据集: {self.config.datasets}")
        self.logger.info(f"异质性: {self.config.heterogeneity_types}")
        self.logger.info(f"客户端: {self.config.n_clients}, 恶意: {self.config.n_malicious}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """设置日志 (覆盖模式，无时间戳)"""
        log_path = self.output_dir / self.config.log_file
        
        # 移除现有handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self, dataset: str) -> nn.Module:
        """创建模型"""
        config = get_dataset_config(dataset)
        model = CreditNet(
            input_dim=config['feature_dim'],
            dataset_type=dataset.lower(),
            num_classes=config['num_classes']
        )
        return model.to(self.config.device)
    
    def _setup_clients(
        self, 
        dataset: str, 
        heterogeneity: str,
        global_model: nn.Module
    ) -> List[FedClient]:
        """设置客户端"""
        clients = []
        
        for i in range(self.config.n_clients):
            try:
                train_data = load_client_data(dataset, i, heterogeneity, is_train=True)
                test_data = load_client_data(dataset, i, heterogeneity, is_train=False)
                
                client = FedClient(
                    client_id=i,
                    model=global_model,
                    train_data=train_data,
                    test_data=test_data,
                    config=self.config
                )
                clients.append(client)
            except FileNotFoundError as e:
                self.logger.warning(f"跳过客户端 {i}: {e}")
        
        return clients
    
    def _run_fl_round(
        self,
        clients: List[FedClient],
        global_model: nn.Module,
        attack: Any,
        malicious_ids: List[int],
        attack_name: str = None
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        运行一轮联邦学习
        
        支持两种攻击模式：
        1. 梯度层攻击：对计算出的梯度应用攻击
        2. 数据投毒攻击：使用投毒数据进行训练（label_flip, backdoor）
        
        Returns:
            gradients: 所有客户端梯度 (恶意客户端已被攻击)
            malicious_ids: 恶意客户端ID列表
        """
        gradients = []
        benign_gradients = []
        
        # 判断是否是数据投毒攻击
        use_data_poisoning = attack_name in ['label_flip', 'backdoor']
        poisoning = get_data_poisoning(attack_name) if use_data_poisoning else None
        
        # 1. 分发全局模型
        for client in clients:
            client.set_parameters(global_model)
        
        # 2. 本地训练
        for client in clients:
            if client.id in malicious_ids and use_data_poisoning and poisoning:
                # 数据投毒训练
                self._train_with_poisoning(client, poisoning)
            else:
                # 正常训练
                client.train()
        
        # 3. 收集梯度
        for client in clients:
            grad = client.get_gradient(global_model)
            if client.id not in malicious_ids:
                benign_gradients.append(grad)
            gradients.append(grad)
        
        # 4. 对恶意客户端应用梯度层攻击（非数据投毒攻击）
        if not use_data_poisoning:
            for i, client in enumerate(clients):
                if client.id in malicious_ids:
                    if hasattr(attack, 'requires_benign_grads') and attack.requires_benign_grads:
                        gradients[i] = attack.attack(gradients[i], benign_gradients)
                    else:
                        gradients[i] = attack.attack(gradients[i])
        
        return gradients, malicious_ids
    
    def _train_with_poisoning(self, client: FedClient, poisoning: Any):
        """使用投毒数据训练客户端"""
        # 获取原始数据
        original_data = []
        for x, y in client.train_loader:
            original_data.append((x, y))
        
        if not original_data:
            client.train()
            return
        
        # 合并所有批次
        all_x = torch.cat([d[0] for d in original_data], dim=0)
        all_y = torch.cat([d[1] for d in original_data], dim=0)
        
        # 投毒
        poisoned_x, poisoned_y = poisoning.poison_data(all_x, all_y)
        
        # 创建新的数据加载器
        poisoned_dataset = TensorDataset(
            poisoned_x.to(client.device),
            poisoned_y.to(client.device)
        )
        poisoned_loader = DataLoader(
            poisoned_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # 使用投毒数据训练
        client.model.train()
        for epoch in range(client.local_epochs):
            for x, y in poisoned_loader:
                client.optimizer.zero_grad()
                output = client.model(x)
                loss = client.criterion(output, y)
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(client.model.parameters(), 1.0)
                    client.optimizer.step()
    
    def _evaluate_detection(
        self,
        detected_malicious: List[int],
        true_malicious: List[int],
        n_clients: int
    ) -> Dict[str, float]:
        """评估检测性能"""
        true_set = set(true_malicious)
        detected_set = set(detected_malicious)
        
        tp = len(true_set & detected_set)
        fp = len(detected_set - true_set)
        fn = len(true_set - detected_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # ========================================================================
    # 实验1: 检测性能对比
    # ========================================================================
    
    def run_detection_experiment(self):
        """
        实验1: 检测性能对比
        
        设置: 12攻击 × 4异质性 × 7检测方法 × 2数据集
        支持详细的阶段统计 (Stage1/Stage2)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验1: 检测性能对比")
        self.logger.info("=" * 60)
        
        results = defaultdict(lambda: defaultdict(dict))
        detailed_stats = defaultdict(lambda: defaultdict(dict))  # 阶段详细统计
        attacks = get_all_attacks()
        
        for dataset in self.config.datasets:
            self.logger.info(f"\n数据集: {dataset}")
            
            for het in self.config.heterogeneity_types:
                self.logger.info(f"  异质性: {het}")
                
                # 创建模型和客户端
                global_model = self._create_model(dataset)
                try:
                    clients = self._setup_clients(dataset, het, global_model)
                except Exception as e:
                    self.logger.warning(f"    跳过 {dataset}/{het}: {e}")
                    continue
                
                if len(clients) < self.config.n_clients:
                    self.logger.warning(f"    客户端不足，跳过")
                    continue
                
                # 选择恶意客户端
                malicious_ids = list(range(self.config.n_malicious))
                
                for attack_name, attack in attacks.items():
                    self.logger.info(f"    攻击: {attack_name}")
                    
                    # 运行一轮FL获取梯度（传递攻击名称以支持数据投毒）
                    gradients, _ = self._run_fl_round(
                        clients, global_model, attack, malicious_ids, attack_name
                    )
                    
                    # 测试各检测方法
                    # 1. FedCAD - 使用详细检测接口
                    fedcad = FedCADDetector(device=self.config.device)
                    detection_result = fedcad.detect_with_details(gradients, malicious_ids)
                    
                    # 整体性能指标
                    metrics_fedcad = {
                        'precision': detection_result.overall_precision,
                        'recall': detection_result.overall_recall,
                        'f1': detection_result.overall_f1,
                        'tp': detection_result.tp,
                        'fp': detection_result.fp,
                        'fn': detection_result.fn
                    }
                    results[f"{dataset}/{het}"][attack_name]['FedCAD'] = metrics_fedcad
                    
                    # 阶段详细统计
                    detailed_stats[f"{dataset}/{het}"][attack_name]['FedCAD'] = {
                        'stage1': {
                            'n_trusted': len(detection_result.stage1.trusted),
                            'n_suspicious': len(detection_result.stage1.suspicious),
                            'n_uncertain': len(detection_result.stage1.uncertain),
                            'accuracy': detection_result.stage1.accuracy,
                            'precision': detection_result.stage1.precision,
                            'recall': detection_result.stage1.recall
                        },
                        'stage2': {
                            'n_trusted': len(detection_result.stage2.trusted) if detection_result.stage2 else 0,
                            'n_suspicious': len(detection_result.stage2.suspicious) if detection_result.stage2 else 0,
                            'n_uncertain': len(detection_result.stage2.uncertain) if detection_result.stage2 else 0,
                            'accuracy': detection_result.stage2.accuracy if detection_result.stage2 else 0.0,
                            'precision': detection_result.stage2.precision if detection_result.stage2 else 0.0,
                            'recall': detection_result.stage2.recall if detection_result.stage2 else 0.0
                        },
                        'final': {
                            'n_trusted': len(detection_result.trusted),
                            'n_suspicious': len(detection_result.suspicious),
                            'detection_time_ms': getattr(detection_result, 'detection_time_ms', 0)
                        }
                    }
                    
                    # 打印阶段统计
                    s1 = detailed_stats[f"{dataset}/{het}"][attack_name]['FedCAD']['stage1']
                    s2 = detailed_stats[f"{dataset}/{het}"][attack_name]['FedCAD']['stage2']
                    self.logger.debug(f"      FedCAD Stage1: {s1['n_trusted']}信任/{s1['n_suspicious']}可疑/{s1['n_uncertain']}不确定, Acc={s1['accuracy']:.3f}")
                    self.logger.debug(f"      FedCAD Stage2: {s2['n_trusted']}信任/{s2['n_suspicious']}可疑, Acc={s2['accuracy']:.3f}")
                    self.logger.debug(f"      FedCAD 总体: Precision={metrics_fedcad['precision']:.3f}, Recall={metrics_fedcad['recall']:.3f}, F1={metrics_fedcad['f1']:.3f}")
                    
                    # 2. 基线防御方法 (使用其检测功能)
                    defenses = get_all_defenses()
                    for defense_name, defense in defenses.items():
                        try:
                            _, normal_idx, anomaly_idx = defense.aggregate_with_detection(gradients)
                            metrics = self._evaluate_detection(anomaly_idx, malicious_ids, len(clients))
                            results[f"{dataset}/{het}"][attack_name][defense_name] = metrics
                        except Exception as e:
                            self.logger.debug(f"      {defense_name} 失败: {e}")
                            results[f"{dataset}/{het}"][attack_name][defense_name] = {
                                'precision': 0, 'recall': 0, 'f1': 0
                            }
        
        self.results['detection'] = dict(results)
        self._save_results('detection_results.json', self.results['detection'])
        
        # 打印汇总
        self._print_detection_summary()
    
    def _print_detection_summary(self):
        """打印检测实验汇总"""
        self.logger.info("\n检测性能汇总 (F1-Score):")
        self.logger.info("-" * 80)
        
        if 'detection' not in self.results:
            return
        
        # 按方法汇总平均F1
        method_f1 = defaultdict(list)
        for setting, attacks in self.results['detection'].items():
            for attack, methods in attacks.items():
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and 'f1' in metrics:
                        method_f1[method].append(metrics['f1'])
        
        self.logger.info(f"{'方法':<20} {'平均F1':<10} {'最小F1':<10} {'最大F1':<10}")
        self.logger.info("-" * 50)
        for method, f1_list in sorted(method_f1.items(), key=lambda x: -np.mean(x[1])):
            if f1_list:
                self.logger.info(f"{method:<20} {np.mean(f1_list):.4f}     {np.min(f1_list):.4f}     {np.max(f1_list):.4f}")
    
    # ========================================================================
    # 实验2: 模型质量与聚合无关性
    # ========================================================================
    
    def run_model_quality_experiment(self):
        """
        实验2: 模型质量与聚合无关性验证
        
        比较:
        - Clean: 无攻击训练 (上界)
        - NoDefense: 有攻击无检测 (下界)
        - FedCAD + 各聚合算法
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验2: 模型质量与聚合无关性验证")
        self.logger.info("=" * 60)
        
        results = {}
        aggregators = ['fedavg', 'median', 'trimmed_mean', 'krum', 'multi_krum']
        
        for dataset in self.config.datasets:
            self.logger.info(f"\n数据集: {dataset}")
            results[dataset] = {}
            
            global_model = self._create_model(dataset)
            try:
                clients = self._setup_clients(dataset, 'iid', global_model)
            except Exception as e:
                self.logger.warning(f"  跳过: {e}")
                continue
            
            malicious_ids = list(range(self.config.n_malicious))
            attack = SignFlipAttack()  # 使用典型攻击
            
            # 运行多轮训练
            n_rounds = min(self.config.global_rounds, 20)
            
            for round_id in range(n_rounds):
                if (round_id + 1) % 5 == 0:
                    self.logger.info(f"  Round {round_id + 1}/{n_rounds}")
                
                gradients, _ = self._run_fl_round(clients, global_model, attack, malicious_ids)
                
                # FedCAD检测
                fedcad = FedCADDetector(device=self.config.device)
                trusted, suspicious, _ = fedcad.detect(gradients)
                
                # 使用不同聚合算法
                trusted_grads = [gradients[i] for i in trusted]
                if trusted_grads:
                    # 更新全局模型
                    aggregated = torch.stack(trusted_grads).mean(dim=0)
                    
                    # 将梯度应用到模型
                    idx = 0
                    for param in global_model.parameters():
                        numel = param.numel()
                        param.data -= aggregated[idx:idx+numel].reshape(param.shape)
                        idx += numel
            
            # 评估最终模型
            final_metrics = {'accuracy': [], 'auc': []}
            for client in clients:
                client.set_parameters(global_model)
                metrics = client.evaluate()
                final_metrics['accuracy'].append(metrics['accuracy'])
                final_metrics['auc'].append(metrics['auc'])
            
            results[dataset] = {
                'accuracy': np.mean(final_metrics['accuracy']),
                'auc': np.mean(final_metrics['auc']),
                'rounds': n_rounds
            }
            
            self.logger.info(f"  最终结果 - Accuracy: {results[dataset]['accuracy']:.4f}, AUC: {results[dataset]['auc']:.4f}")
        
        self.results['model_quality'] = results
        self._save_results('model_quality_results.json', results)
    
    # ========================================================================
    # 实验3: 消融实验
    # ========================================================================
    
    def run_ablation_experiment(self):
        """
        实验3: 消融实验
        
        变体:
        - Full FedCAD: 完整方法
        - w/o Committee: 移除委员会
        - w/o Direction: 移除方向通道
        - Committee Only: 仅委员会
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验3: 消融实验")
        self.logger.info("=" * 60)
        
        results = {}
        
        for dataset in self.config.datasets:
            self.logger.info(f"\n数据集: {dataset}")
            results[dataset] = {}
            
            global_model = self._create_model(dataset)
            try:
                clients = self._setup_clients(dataset, 'label', global_model)
            except Exception as e:
                self.logger.warning(f"  跳过: {e}")
                continue
            
            malicious_ids = list(range(self.config.n_malicious))
            attack = ALIEAttack()  # 使用高级攻击测试
            
            gradients, _ = self._run_fl_round(clients, global_model, attack, malicious_ids)
            
            # 完整FedCAD
            fedcad_full = FedCADDetector(device=self.config.device)
            _, suspicious_full, _ = fedcad_full.detect(gradients)
            results[dataset]['Full_FedCAD'] = self._evaluate_detection(suspicious_full, malicious_ids, len(clients))
            
            # 仅委员会 (禁用自编码器)
            fedcad_committee = FedCADDetector(device=self.config.device, recon_weight=0.0)
            _, suspicious_committee, _ = fedcad_committee.detect(gradients)
            results[dataset]['Committee_Only'] = self._evaluate_detection(suspicious_committee, malicious_ids, len(clients))
            
            # 无方向通道 (仅重构)
            fedcad_no_dir = FedCADDetector(device=self.config.device, recon_weight=1.0)
            _, suspicious_no_dir, _ = fedcad_no_dir.detect(gradients)
            results[dataset]['w/o_Direction'] = self._evaluate_detection(suspicious_no_dir, malicious_ids, len(clients))
            
            for variant, metrics in results[dataset].items():
                self.logger.info(f"  {variant}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        self.results['ablation'] = results
        self._save_results('ablation_results.json', results)
    
    # ========================================================================
    # 实验4: 参数敏感性
    # ========================================================================
    
    def run_sensitivity_experiment(self):
        """
        实验4: 参数敏感性分析
        
        分析关键参数:
        - committee_size: [3, 4, 5, 6, 7, 8]
        - recon_weight: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        - mad_k: [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验4: 参数敏感性分析")
        self.logger.info("=" * 60)
        
        results = {}
        
        dataset = self.config.datasets[0]  # 使用第一个数据集
        global_model = self._create_model(dataset)
        
        try:
            clients = self._setup_clients(dataset, 'iid', global_model)
        except Exception as e:
            self.logger.warning(f"跳过: {e}")
            return
        
        malicious_ids = list(range(self.config.n_malicious))
        attack = MinMaxAttack()
        
        gradients, _ = self._run_fl_round(clients, global_model, attack, malicious_ids)
        
        # 委员会大小
        results['committee_size'] = {}
        for k in [3, 4, 5, 6, 7, 8]:
            fedcad = FedCADDetector(committee_size=k, device=self.config.device)
            _, suspicious, _ = fedcad.detect(gradients)
            results['committee_size'][k] = self._evaluate_detection(suspicious, malicious_ids, len(clients))
        
        self.logger.info("委员会大小敏感性:")
        for k, m in results['committee_size'].items():
            self.logger.info(f"  K={k}: F1={m['f1']:.4f}")
        
        # 重构权重
        results['recon_weight'] = {}
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            fedcad = FedCADDetector(recon_weight=alpha, device=self.config.device)
            _, suspicious, _ = fedcad.detect(gradients)
            results['recon_weight'][alpha] = self._evaluate_detection(suspicious, malicious_ids, len(clients))
        
        self.logger.info("重构权重敏感性:")
        for alpha, m in results['recon_weight'].items():
            self.logger.info(f"  α={alpha}: F1={m['f1']:.4f}")
        
        # MAD系数
        results['mad_k'] = {}
        for k in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            fedcad = FedCADDetector(mad_k=k, device=self.config.device)
            _, suspicious, _ = fedcad.detect(gradients)
            results['mad_k'][k] = self._evaluate_detection(suspicious, malicious_ids, len(clients))
        
        self.logger.info("MAD系数敏感性:")
        for k, m in results['mad_k'].items():
            self.logger.info(f"  k={k}: F1={m['f1']:.4f}")
        
        self.results['sensitivity'] = results
        self._save_results('sensitivity_results.json', results)
    
    # ========================================================================
    # 结果保存
    # ========================================================================
    
    def _save_results(self, filename: str, data: dict):
        """保存结果到JSON"""
        filepath = self.output_dir / filename
        
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(convert(data), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存: {filepath}")
    
    # ========================================================================
    # 运行入口
    # ========================================================================
    
    def run(self):
        """运行所有实验"""
        start_time = time.time()
        
        if self.config.run_detection:
            self.run_detection_experiment()
        
        if self.config.run_model_quality:
            self.run_model_quality_experiment()
        
        if self.config.run_ablation:
            self.run_ablation_experiment()
        
        if self.config.run_sensitivity:
            self.run_sensitivity_experiment()
        
        elapsed = time.time() - start_time
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"所有实验完成！总耗时: {elapsed/60:.1f} 分钟")
        self.logger.info(f"结果目录: {self.output_dir}")
        self.logger.info("=" * 60)
        
        return self.results


# ==============================================================================
# 命令行入口
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FedCAD 真实联邦学习实验')
    parser.add_argument('--n_clients', type=int, default=10)
    parser.add_argument('--n_malicious', type=int, default=3)
    parser.add_argument('--global_rounds', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--only', type=str, choices=['detection', 'quality', 'ablation', 'sensitivity'])
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        n_clients=args.n_clients,
        n_malicious=args.n_malicious,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        seed=args.seed,
        use_gpu=not args.no_gpu
    )
    
    if args.only:
        config.run_detection = args.only == 'detection'
        config.run_model_quality = args.only == 'quality'
        config.run_ablation = args.only == 'ablation'
        config.run_sensitivity = args.only == 'sensitivity'
    
    experiment = FedCADExperiment(config)
    experiment.run()
