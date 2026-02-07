#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedCAD 完整对比实验
==================

在UCI数据集上进行完整的检测性能对比实验：
- 12种攻击 × 4种异质性 × 多种检测方法
- 统计各阶段性能
- 与基线方法对比

作者: FedCAD Team
日期: 2026-02-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import time

from fedcad_experiment_engine import (
    FedCADExperiment, ExperimentConfig, 
    get_all_attacks, get_all_defenses
)
from flcore.attack.fedcad_detector import FedCADDetectorV5


def evaluate_detection(detected_suspicious: list, true_malicious: list, n_clients: int) -> dict:
    """评估检测性能"""
    true_set = set(true_malicious)
    detected_set = set(detected_suspicious)
    
    tp = len(true_set & detected_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)
    tn = n_clients - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_clients if n_clients > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def run_baseline_detection(defense, gradients: list, malicious_ids: list, n_clients: int) -> dict:
    """运行基线防御方法的检测"""
    try:
        # 大多数防御方法通过aggregate_with_detection返回检测结果
        _, normal_idx, anomaly_idx = defense.aggregate_with_detection(gradients)
        return evaluate_detection(anomaly_idx, malicious_ids, n_clients)
    except Exception as e:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'error': str(e)}


def run_experiment():
    """运行完整对比实验"""
    print("=" * 80)
    print("FedCAD 完整对比实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置
    config = ExperimentConfig(
        datasets=['Uci'],
        heterogeneity_types=['iid', 'label', 'feature', 'quantity'],
        n_clients=10,
        n_malicious=3,
        global_rounds=1,
        local_epochs=3
    )
    
    experiment = FedCADExperiment(config)
    attacks = get_all_attacks()
    malicious_ids = [0, 1, 2]
    
    # 结果存储
    results = defaultdict(lambda: defaultdict(dict))
    stage_stats = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n配置: {config.n_clients}客户端, {config.n_malicious}恶意")
    print(f"攻击: {len(attacks)}种")
    print(f"异质性: {config.heterogeneity_types}")
    
    for het in config.heterogeneity_types:
        print(f"\n{'='*60}")
        print(f"异质性: {het}")
        print(f"{'='*60}")
        
        # 创建模型和客户端
        try:
            global_model = experiment._create_model('Uci')
            clients = experiment._setup_clients('Uci', het, global_model)
        except Exception as e:
            print(f"  跳过: {e}")
            continue
        
        n_clients = len(clients)
        
        for attack_name, attack in attacks.items():
            print(f"\n  攻击: {attack_name}")
            
            # 重置客户端
            for client in clients:
                client.set_parameters(global_model)
            
            # 运行FL获取梯度
            try:
                gradients, _ = experiment._run_fl_round(
                    clients, global_model, attack, malicious_ids, attack_name
                )
            except Exception as e:
                print(f"    FL失败: {e}")
                continue
            
            # ================================================================
            # 1. FedCAD V5检测 (优化参数加速)
            # ================================================================
            detector = FedCADDetectorV5(
                device='cpu', 
                verbose=False,
                ae_epochs=30,  # 减少训练轮次加速
                ae_hidden_dims=[256, 128],  # 更小的网络
                ae_latent_dim=64
            )
            detection_result = detector.detect(gradients, malicious_ids)
            
            fedcad_metrics = {
                'precision': detection_result.precision,
                'recall': detection_result.recall,
                'f1': detection_result.f1,
                'tp': detection_result.tp,
                'fp': detection_result.fp,
                'fn': detection_result.fn,
                'time_ms': detection_result.detection_time_ms
            }
            results[het][attack_name]['FedCAD'] = fedcad_metrics
            
            # 阶段统计
            stage_stats[het][attack_name] = {
                'stage1': {
                    'trusted': len(detection_result.stage1.trusted) if detection_result.stage1 else 0,
                    'suspicious': len(detection_result.stage1.suspicious) if detection_result.stage1 else 0,
                    'uncertain': len(detection_result.stage1.uncertain) if detection_result.stage1 else 0,
                    'precision': detection_result.stage1.precision if detection_result.stage1 else 0,
                    'recall': detection_result.stage1.recall if detection_result.stage1 else 0,
                    'f1': detection_result.stage1.f1 if detection_result.stage1 else 0
                },
                'stage2': {
                    'trusted': len(detection_result.stage2.trusted) if detection_result.stage2 else 0,
                    'suspicious': len(detection_result.stage2.suspicious) if detection_result.stage2 else 0,
                    'precision': detection_result.stage2.precision if detection_result.stage2 else 0,
                    'recall': detection_result.stage2.recall if detection_result.stage2 else 0,
                    'f1': detection_result.stage2.f1 if detection_result.stage2 else 0
                },
                'stage3': {
                    'trusted': len(detection_result.stage3.trusted) if detection_result.stage3 else 0,
                    'suspicious': len(detection_result.stage3.suspicious) if detection_result.stage3 else 0
                }
            }
            
            # ================================================================
            # 2. 基线方法检测
            # ================================================================
            defenses = get_all_defenses()
            
            for defense_name, defense in defenses.items():
                baseline_metrics = run_baseline_detection(
                    defense, gradients, malicious_ids, n_clients
                )
                results[het][attack_name][defense_name] = baseline_metrics
            
            # 打印简要结果
            s1 = stage_stats[het][attack_name]['stage1']
            print(f"    FedCAD: F1={fedcad_metrics['f1']:.3f} (S1:{s1['trusted']}/{s1['suspicious']}/{s1['uncertain']})")
            
            best_baseline = max(defenses.keys(), 
                              key=lambda x: results[het][attack_name].get(x, {}).get('f1', 0))
            best_f1 = results[het][attack_name].get(best_baseline, {}).get('f1', 0)
            print(f"    最佳基线: {best_baseline} F1={best_f1:.3f}")
    
    # ========================================================================
    # 汇总统计
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("实验汇总")
    print("=" * 80)
    
    # 按方法汇总
    method_f1 = defaultdict(list)
    for het, attacks_dict in results.items():
        for attack_name, methods_dict in attacks_dict.items():
            for method, metrics in methods_dict.items():
                if isinstance(metrics, dict) and 'f1' in metrics:
                    method_f1[method].append(metrics['f1'])
    
    print(f"\n{'方法':<20} {'平均F1':<10} {'标准差':<10} {'最小':<10} {'最大':<10}")
    print("-" * 60)
    for method in sorted(method_f1.keys(), key=lambda x: -np.mean(method_f1[x])):
        f1_list = method_f1[method]
        if f1_list:
            print(f"{method:<20} {np.mean(f1_list):.4f}     {np.std(f1_list):.4f}     {np.min(f1_list):.4f}     {np.max(f1_list):.4f}")
    
    # 按攻击汇总
    print(f"\n按攻击类型汇总 (FedCAD vs 最佳基线):")
    print("-" * 60)
    for attack_name in attacks.keys():
        fedcad_f1s = []
        best_baseline_f1s = []
        
        for het in config.heterogeneity_types:
            if het in results and attack_name in results[het]:
                fedcad_f1 = results[het][attack_name].get('FedCAD', {}).get('f1', 0)
                fedcad_f1s.append(fedcad_f1)
                
                baseline_f1s = [v.get('f1', 0) for k, v in results[het][attack_name].items() 
                               if k != 'FedCAD' and isinstance(v, dict)]
                if baseline_f1s:
                    best_baseline_f1s.append(max(baseline_f1s))
        
        if fedcad_f1s:
            avg_fedcad = np.mean(fedcad_f1s)
            avg_baseline = np.mean(best_baseline_f1s) if best_baseline_f1s else 0
            diff = avg_fedcad - avg_baseline
            print(f"{attack_name:<15}: FedCAD={avg_fedcad:.3f}, 最佳基线={avg_baseline:.3f}, 差异={diff:+.3f}")
    
    # 按异质性汇总
    print(f"\n按异质性汇总:")
    print("-" * 60)
    for het in config.heterogeneity_types:
        if het in results:
            fedcad_f1s = [results[het][a].get('FedCAD', {}).get('f1', 0) 
                         for a in results[het]]
            avg_f1 = np.mean(fedcad_f1s) if fedcad_f1s else 0
            print(f"{het:<15}: FedCAD平均F1={avg_f1:.4f}")
    
    # 阶段贡献分析
    print(f"\n各阶段贡献分析:")
    print("-" * 60)
    stage1_detections = []
    stage2_detections = []
    stage3_adjustments = []
    
    for het, attacks_dict in stage_stats.items():
        for attack_name, stages in attacks_dict.items():
            stage1_detections.append(stages['stage1']['suspicious'])
            stage2_detections.append(stages['stage2']['suspicious'])
            stage3_adjustments.append(stages['stage3']['suspicious'])
    
    print(f"Stage 1 平均检出可疑: {np.mean(stage1_detections):.2f}")
    print(f"Stage 2 平均检出可疑: {np.mean(stage2_detections):.2f}")
    print(f"Stage 3 最终可疑: {np.mean(stage3_adjustments):.2f}")
    
    # 保存结果
    output = {
        'config': {
            'n_clients': config.n_clients,
            'n_malicious': config.n_malicious,
            'datasets': config.datasets,
            'heterogeneity': config.heterogeneity_types
        },
        'results': dict(results),
        'stage_stats': dict(stage_stats),
        'summary': {
            'method_avg_f1': {m: float(np.mean(f)) for m, f in method_f1.items()}
        }
    }
    
    output_path = 'fedcad_comparison_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, stage_stats


if __name__ == '__main__':
    run_experiment()
