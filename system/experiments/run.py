#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedCAD 真实联邦学习实验启动器
===========================

使用真实数据集进行完整的联邦学习训练和拜占庭攻击检测实验。

使用方法:
---------
# 运行完整实验
python run.py

# 快速测试 (减少轮数)
python run.py --quick

# 仅运行检测实验
python run.py --only detection

# 仅运行消融实验
python run.py --only ablation

# 自定义参数
python run.py --n_clients 10 --n_malicious 3 --global_rounds 30

# 禁用GPU
python run.py --no_gpu
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from fedcad_experiment_engine import ExperimentConfig, FedCADExperiment


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FedCAD 真实联邦学习实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                      # 运行完整实验 (真实FL训练)
  python run.py --quick              # 快速测试 (5轮训练)
  python run.py --only detection     # 仅检测性能实验
  python run.py --only quality       # 仅模型质量实验
  python run.py --only ablation      # 仅消融实验
  python run.py --only sensitivity   # 仅参数敏感性
        """
    )
    
    # 联邦学习参数
    parser.add_argument('--n_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--n_malicious', type=int, default=3, help='恶意客户端数量')
    parser.add_argument('--global_rounds', type=int, default=50, help='全局训练轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 计算资源
    parser.add_argument('--no_gpu', action='store_true', help='禁用GPU')
    
    # 实验选择
    parser.add_argument('--only', type=str, default=None,
                       choices=['detection', 'quality', 'ablation', 'sensitivity'],
                       help='仅运行指定实验')
    
    # 快速模式
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (5轮)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        n_clients=args.n_clients,
        n_malicious=args.n_malicious,
        global_rounds=5 if args.quick else args.global_rounds,
        local_epochs=2 if args.quick else args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        use_gpu=not args.no_gpu,
    )
    
    # 选择性运行
    if args.only:
        config.run_detection = args.only == 'detection'
        config.run_model_quality = args.only == 'quality'
        config.run_ablation = args.only == 'ablation'
        config.run_sensitivity = args.only == 'sensitivity'
    
    if args.quick:
        print("[快速测试] 5轮全局训练, 2轮本地训练")
    
    print("\n" + "=" * 60)
    print("FedCAD 真实联邦学习实验")
    print("=" * 60)
    print(f"客户端: {config.n_clients}, 恶意: {config.n_malicious}")
    print(f"训练: {config.global_rounds}轮 × {config.local_epochs}本地epoch")
    print(f"设备: {config.device}")
    print("=" * 60 + "\n")
    
    # 运行实验
    experiment = FedCADExperiment(config)
    experiment.run()


if __name__ == '__main__':
    main()
