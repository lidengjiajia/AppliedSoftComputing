# HierFed: Hierarchical Byzantine Detection for Secure Federated Credit Scoring

基于层级检测与声誉治理的拜占庭鲁棒联邦信用评估框架。

> **论文**: *HierFed: Hierarchical Byzantine Detection with Reputation-Based Governance for Secure Federated Credit Scoring*
>
> 投稿期刊: Expert Systems with Applications

## 核心贡献

1. **HierFed层级检测框架**: 三阶段互补检测架构，在保持模型精度的同时实现高检测覆盖率
2. **图委员会选举机制**: 基于加权中心度分析的梯度相似性结构筛选，容忍异质性导致的梯度变化
3. **对比自编码器检测**: 利用高阶分布特征检测优化类攻击（ALIE, MinMax等）
4. **跨时序声誉追踪**: 非对称更新动态——慢速信任累积与快速衰减，识别战略性攻击者

## 三阶段检测架构

```
Stage 1: 图委员会选举 → 结构异常过滤 + 异质性容忍
Stage 2: 对比自编码器 → 优化类攻击检测 (重构误差 + 潜在距离)
Stage 3: 声誉累积判定 → 战略性攻击者识别 + 审计日志
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选)

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── dataset/                    # 数据集
│   ├── Uci/                   # UCI信用卡违约数据集
│   └── Xinwang/               # 商业银行信用数据集
├── system/
│   ├── flcore/                # 联邦学习核心
│   │   ├── attack/            # 攻击与防御模块
│   │   │   ├── attacks.py     # 12种攻击实现
│   │   │   ├── defenses.py    # 防御策略
│   │   │   └── fedcad_detector.py  # HierFed检测器
│   │   ├── clients/           # 客户端实现
│   │   ├── servers/           # 服务端实现
│   │   └── trainmodel/        # 信用评分网络
│   ├── experiments/           # 实验脚本
│   └── utils/                 # 工具函数
├── results/                   # 实验结果
└── ASCJ/                      # 论文LaTeX源文件
```

## 快速开始

```bash
cd system/experiments
python run.py              # 完整实验
python run.py --quick      # 快速测试
```

## 支持的攻击类型

| 类别 | 攻击 | 特点 |
|------|------|------|
| 扰动攻击 | Sign-flip, Gaussian, Scale | 产生结构异常 |
| 优化攻击 | ALIE, IPM, MinMax | 匹配诚实梯度统计量 |
| 语义攻击 | Label-flip, Backdoor, Free-rider | 无明显梯度模式 |

## 对比方法

| 方法 | 扰动攻击 | 优化攻击 | 战略攻击 |
|------|----------|----------|----------|
| Median | ✓ | ✗ | ✗ |
| Krum | ✓ | ✗ | ✗ |
| FLTrust | ✓ | ○ | ✗ |
| **HierFed** | **✓** | **✓** | **✓** |

## 数据集

| 数据集 | 特征 | 样本 | 来源 |
|--------|------|------|------|
| UCI | 23 | 30,000 | UCI公开数据集 |
| Xinwang | 38 | 17,886 | 商业银行数据 |

## 许可证

MIT License
