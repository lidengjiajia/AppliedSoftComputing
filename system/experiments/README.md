# FedCAD 完整实验方案

## 📋 实验概述

本文档详细描述了 FedCAD (Committee-Autoencoder Detection) 框架的完整实验方案。

### 核心定位
- **研究问题**: 联邦学习中的拜占庭攻击检测
- **方法类型**: 防御/检测方法 (不是攻击方法)
- **核心挑战**: 在数据异质性 (Non-IID) 环境下区分攻击与正常梯度差异
- **创新点**: 聚合无关的两阶段检测框架

---

## 🔧 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    FedCAD 系统流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  客户端梯度 [g₁, g₂, ..., gₙ]                                    │
│           ↓                                                     │
│  ┌─────────────────────────────────┐                            │
│  │  检测层 (Detection Layer)        │  ← FedCAD / 基线检测方法   │
│  │  Stage 1: 委员会筛选             │                            │
│  │  Stage 2: 双通道自编码器          │                            │
│  │  输出: 信任集 / 可疑集            │                            │
│  └─────────────────────────────────┘                            │
│           ↓ (仅信任梯度)                                         │
│  ┌─────────────────────────────────┐                            │
│  │  聚合层 (Aggregation Layer)      │  ← FedAvg / FedProx / ... │
│  │  输出: 全局模型更新               │                            │
│  └─────────────────────────────────┘                            │
│           ↓                                                     │
│  全局模型                                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键区分**：
- **检测方法**: FedCAD, Median, Krum, FLTrust等 → 识别恶意客户端
- **聚合方法**: FedAvg, FedProx, SCAFFOLD等 → 合并梯度

---

## 📊 实验要素

### 攻击类型 (12种)

| 类别 | 攻击名称 | 论文出处 | 特点 |
|------|----------|----------|------|
| **基础攻击** | sign_flip | - | 符号翻转 |
| | gaussian | - | 高斯噪声 |
| | scale | - | 缩放攻击 |
| | zero | - | 零梯度 |
| **优化攻击** | little | NeurIPS'19 | 边界隐蔽 |
| | alie | NeurIPS'19 | 自适应 |
| | ipm | ICML'18 | 内积操纵 |
| | minmax | IEEE S&P'20 | 最大化偏离 |
| **语义攻击** | label_flip | - | 标签翻转 |
| | backdoor | AISTATS'20 | 后门植入 |
| | free_rider | - | 搭便车 |
| | collision | - | 串通攻击 |

### 检测方法 (6种基线 + FedCAD)

| 方法 | 论文出处 | 类型 |
|------|----------|------|
| Median | ICML'18 | 统计鲁棒 |
| TrimMean | ICML'18 | 统计鲁棒 |
| Krum | NeurIPS'17 | 距离选择 |
| MultiKrum | NeurIPS'17 | 距离选择 |
| Bulyan | ICML'18 | 混合方法 |
| FLTrust | NDSS'21 | 信任根 |
| **FedCAD** | **本文** | **委员会+双通道** |

### 异质性类型 (4种)

| 类型 | 描述 | 挑战 |
|------|------|------|
| IID | 独立同分布 | 基准场景 |
| Label Skew | 标签分布不均 | 梯度方向差异大 |
| Feature Skew | 特征分布不均 | 梯度幅度差异大 |
| Quantity Skew | 数据量不均 | 梯度尺度差异大 |

### 聚合算法 (5种)

| 算法 | 用途 |
|------|------|
| FedAvg | 基础聚合 |
| FedProx | 近端正则化 |
| SCAFFOLD | 方差修正 |
| MOON | 对比学习 |
| FedNova | 归一化平均 |

### 数据集 (2种)

| 数据集 | 特征数 | 样本数 | 场景 |
|--------|--------|--------|------|
| UCI Credit | 23 | 30,000 | 经典金融 |
| Xinwang Credit | 38 | 17,886 | 真实工业 |

---

## 📈 评估指标

### 检测指标 (主要)

| 指标 | 公式 | 说明 |
|------|------|------|
| **Precision** | TP / (TP + FP) | 精确率，避免误报 |
| **Recall** | TP / (TP + FN) | 召回率，避免漏检 |
| **F1-score** | 2×P×R / (P+R) | 综合指标 (主指标) |

其中：
- TP: 正确检测的恶意客户端
- FP: 误报 (正常客户端被判为恶意)
- FN: 漏检 (恶意客户端未被检测)

### 模型指标 (辅助)

| 指标 | 说明 |
|------|------|
| Accuracy | 模型准确率 |
| AUC | ROC曲线下面积 |
| Gap to Clean | 与无攻击基线的差距 |

### 基线定义

| 基线 | 含义 | 作用 |
|------|------|------|
| **Clean** | 无攻击 + 正常聚合 | 性能上界 |
| **No Defense** | 有攻击 + 无检测 | 性能下界 |

---

## 🧪 实验设计

### 实验1: 检测性能对比 (主实验)

**目标**: 证明FedCAD检测恶意客户端的能力优于基线

**设置**:
- 攻击: 12种
- 异质性: 4种
- 检测方法: 7种 (6基线 + FedCAD)
- 聚合: 固定FedAvg (控制变量)

**指标**: Precision, Recall, F1-score

**配置数**: 12 × 4 × 7 = **336组**

**结果呈现**:
- Table: 攻击 × 异质性 × 方法 (取F1)
- Heatmap: 可视化优势场景

---

### 实验2: 模型质量验证

**目标**: 证明检测后模型性能接近无攻击基线

**设置**:
- 基线: Clean (上界), No Defense (下界)
- 代表性配置: 4种 (iid+gaussian, label+minmax, feature+alie, quantity+ipm)

**指标**: Accuracy, AUC, Gap to Clean

**结果呈现**:
- 条形图: 各方法与Clean基线的差距

---

### 实验3: 聚合无关性验证

**目标**: 证明FedCAD可与任意聚合算法配合

**设置**:
- 检测: 固定FedCAD
- 聚合: FedAvg, FedProx, SCAFFOLD, MOON, FedNova

**预期结果**: 不同聚合算法下，FedCAD检测性能稳定

**意义**: 验证"检测与聚合解耦"的设计理念

---

### 实验4: 消融实验

**目标**: 验证各组件贡献

**变体**:

| 变体 | 设置 | 验证 |
|------|------|------|
| Full FedCAD | 完整方法 | 基准 |
| w/o Committee | 禁用委员会 | 委员会必要性 |
| w/o Direction | 仅重构通道 | 方向通道贡献 |
| Committee Only | 仅委员会 | 自编码器贡献 |

**配置**: 4变体 × 4代表性场景 = **16组**

---

### 实验5: 参数敏感性分析

**目标**: 分析关键参数影响

**参数**:

| 参数 | 取值范围 | 默认值 |
|------|----------|--------|
| 委员会大小 K | [3, 4, 5, 6, 7, 8] | 5 |
| 重构权重 α | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] | 0.6 |
| MAD系数 k | [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] | 3.5 |

**配置**: 3参数 × 6取值 = **18组**

---

## 📁 文件结构

```
system/experiments/
├── fedcad_complete_experiments.py   # 主实验代码
├── run_all_experiments.py           # 启动脚本
├── experiment_config.json           # 配置文件
└── README.md                        # 本文档

system/flcore/attack/
├── attacks.py                       # 12种攻击实现
├── defenses.py                      # 6种基线防御
└── fedcad_detector.py               # FedCAD核心检测器
```

---

## 🚀 运行方法

### 运行所有实验

```bash
cd system/experiments
python run_all_experiments.py
```

### 运行单个实验

```bash
# 实验1: 检测性能对比
python run_all_experiments.py --experiment 1

# 实验4: 消融实验
python run_all_experiments.py --experiment 4
```

### 快速测试

```bash
python run_all_experiments.py --quick
```

### 自定义参数

```bash
python run_all_experiments.py --n_clients 20 --n_malicious 6 --seed 123
```

---

## 📤 输出文件

| 文件类型 | 文件名 | 内容 |
|----------|--------|------|
| JSON | `all_results_{timestamp}.json` | 完整实验数据 |
| LaTeX | `tables_{timestamp}.tex` | 论文表格 |
| Excel | `experiment1_detection_{timestamp}.xlsx` | 检测结果汇总 |
| Excel | `experiment4_ablation_{timestamp}.xlsx` | 消融实验汇总 |

---

## 📋 实验规模汇总

| 实验 | 配置数 | 说明 |
|------|--------|------|
| 实验1 | 336 | 12攻击 × 4异质性 × 7方法 |
| 实验2 | ~40 | 4配置 × (7方法 + 2基线) |
| 实验3 | ~10 | 2配置 × 5聚合算法 |
| 实验4 | 16 | 4变体 × 4配置 |
| 实验5 | 18 | 3参数 × 6取值 |
| **总计** | **~420组** | |

---

## ✅ 检查清单

- [x] 12种攻击全覆盖 (基础+优化+语义)
- [x] 6种检测基线 (统计+距离+信任根)
- [x] 4种数据异质性 (IID + 3种Non-IID)
- [x] 5种聚合算法 (验证聚合无关性)
- [x] 2种数据集 (UCI + Xinwang)
- [x] 消融实验 (4种变体)
- [x] 参数敏感性 (3个关键参数)
- [x] 完整评估指标 (P/R/F1 + Acc/AUC)
- [x] 清晰基线定义 (Clean/No Defense)
- [x] 自动化结果导出 (JSON/LaTeX/Excel)
