# FedCAD: Federated Credit Assessment with Adversarial Defense

基于联邦学习的信用评估系统，支持拜占庭攻击检测与防御。

## 特性

- **多种联邦学习算法**: FedAvg, FedProx, SCAFFOLD, MOON, PerAvg, FedRep, FedGWO, FedPSO, FedTLBO
- **三阶段拜占庭检测**: 图连通性委员会选举 → 自编码器异常检测 → 声誉累积判定
- **多种攻击防御**: 支持12种攻击类型和7种检测方法
- **数据异质性支持**: IID, Label Skew, Feature Skew, Quantity Skew

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── dataset/                    # 数据集
│   ├── Uci/                   # UCI信用卡违约数据集
│   └── Xinwang/               # 新网银行信用数据集
├── system/
│   ├── flcore/                # 联邦学习核心
│   │   ├── attack/            # 攻击与防御模块
│   │   │   ├── attacks.py     # 攻击实现
│   │   │   ├── defenses.py    # 防御实现
│   │   │   └── fedcad_detector.py  # FedCAD检测器
│   │   ├── clients/           # 客户端实现
│   │   ├── servers/           # 服务端实现
│   │   └── trainmodel/        # 神经网络模型
│   ├── experiments/           # 实验脚本
│   └── utils/                 # 工具函数
├── results/                   # 实验结果
└── ASCJ/                      # 论文相关文件
```

## 快速开始

### 运行完整实验

```bash
cd system/experiments
python run.py
```

### 快速测试

```bash
python run.py --quick
```

### 自定义实验

```bash
# 指定客户端数量和恶意客户端数量
python run.py --n_clients 10 --n_malicious 3 --global_rounds 50

# 仅运行检测性能实验
python run.py --only detection

# 仅运行消融实验
python run.py --only ablation
```

### 使用主程序

```bash
cd system/flcore
python main.py --dataset Uci --algorithm FedTLBO --num_clients 10 --global_rounds 50
```

## 支持的算法

| 算法 | 说明 |
|------|------|
| Centralized | 中心化训练 (性能上界) |
| FedAvg | 联邦平均 |
| FedProx | 联邦近端优化 |
| SCAFFOLD | 方差缩减联邦学习 |
| MOON | 模型对比联邦学习 |
| FedRep | 联邦表征学习 |
| FedGWO | 灰狼优化聚合 |
| FedPSO | 粒子群优化聚合 |
| FedTLBO | TLBO优化聚合 (本文方法) |

## 数据集

| 数据集 | 特征数 | 样本数 | 说明 |
|--------|--------|--------|------|
| UCI | 23 | 30,000 | UCI信用卡违约数据集 |
| Xinwang | 38 | 17,886 | 新网银行信用数据集 |

## 许可证

MIT License
