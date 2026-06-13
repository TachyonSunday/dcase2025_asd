# DCASE 2025 Task 2 — 机器音频无监督异常检测

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **BEATs + GenRep 分块时序池化 + Domain-wise Density Scoring**
> 
> 面向 DCASE 2025 Task 2 的无监督机器音频异常检测系统，支持 Domain-Generalized 设定。

---

## 📋 目录

- [项目概述](#项目概述)
- [算法架构](#算法架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [评估结果](#评估结果)
- [工具脚本](#工具脚本)
- [参考文献](#参考文献)

---

## 项目概述

本项目针对 **DCASE 2025 Task 2: Unsupervised Anomaly Detection for Machine Audio** 挑战赛，实现了一套完整的机器音频异常检测管线。

### 核心思路

1. **特征提取**：利用在 AudioSet 上预训练的 **BEATs** (Audio Pre-Training with Acoustic Tokenizers) 模型，提取多层 Transformer 注意力特征
2. **特征聚合**：按照 GenRep baseline 的方式执行分块时序池化（Temporal Chunk Pooling），将变长序列压缩为固定维度 Embedding
3. **异常打分**：基于 Domain-wise Local Density Normalization 和智能双模 Z-Score 算法，计算每个测试样本的异常分数

### 任务设定

- **无监督**：仅使用正常样本构建内存库（Memory Bank）
- **域泛化**：训练集和测试集来自不同声学环境（source/target domain）
- **7 类机器**：ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve

---

## 算法架构

```
┌─────────────────────────────────────────────────────────────┐
│                    输入：16kHz 单声道 WAV                      │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  BEATs Encoder (12层 Transformer, frozen)                   │
│  - 内部 fbank: 128 mel bins                                 │
│  - Conv2d patch embedding (stride=16)                       │
│  - 提取 Layer 0~10 的注意力权重                              │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  GenRep 分块时序池化                                         │
│  - freq_groups = 8 (频率分块)                                │
│  - num_chunks = 8 (时间分块)                                 │
│  - Hybrid Mean + Max Pooling                                │
│  - 输出: 12288-D Embedding per layer                        │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Domain-wise Density Scoring                                │
│  - source_memory_bank + target_memory_bank                  │
│  - KNN 局部密度估计                                          │
│  - 智能双模 Z-Score (batch / fallback)                      │
│  - score = min(Z_source, Z_target)                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
              异常分数 (越高越异常)
```

### 关键设计

| 组件 | 设计选择 | 说明 |
|:---|:---|:---|
| **预训练模型** | BEATs (AudioSet fine-tuned) | 12层 Transformer, 768-D hidden |
| **最优层** | Layer 2 | Official Score Ω = 0.5950 (dev set) |
| **池化方式** | GenRep 分块时序池化 | freq_groups=8, Hybrid Mean+Max |
| **打分算法** | Domain-wise KNN + Z-Score | 双模切换防 NaN |
| **推理公式** | `score = min(Z_source, Z_target)` | 跨域鲁棒性 |

---

## 项目结构

```
Dcase_competition/
│
├── configs/                      # 全局配置
│   └── config.yaml               #   音频/模型/评估参数
│
├── models/                       # 特征提取核心
│   ├── __init__.py
│   ├── feature_extractor.py      #   BEATs 多层特征提取器
│   └── beats/                    #   BEATs 预训练模型源码
│       ├── BEATs.py
│       ├── backbone.py
│       ├── modules.py
│       ├── Tokenizers.py
│       └── quantizer.py
│
├── utils/                        # 异常打分
│   ├── __init__.py
│   └── scoring.py                #   DomainWiseDensityScorer
│
├── tools/                        # 辅助工具（可视化 & 诊断）
│   ├── generate_sci_plots.py     #   SCI 级可视化套件
│   ├── generate_3d_gif.py        #   3D UMAP 旋转 GIF
│   ├── plot_ablations.py         #   消融实验可视化
│   ├── plot_anm_score.py         #   异常分数分布绘图
│   ├── analyze_distribution.py   #   分数分布诊断
│   ├── check_dataset.py          #   数据集完整性校验
│   └── test_plots.py             #   通用绘图基础设施
│
├── docs/                         # 项目文档
│
├── dataset.py                    # AudioAnomalyDataset
├── extract_features.py           # 特征提取主脚本
├── evaluate.py                   # DCASE 官方评估脚本
├── generate_submission.py        # 盲测提交 CSV 生成
└── requirements.txt              # Python 依赖
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <repo-url>
cd Dcase_competition

# 安装依赖 (推荐 Python 3.9+, CUDA 11.8+)
pip install -r requirements.txt
```

### 2. 数据准备

下载 DCASE 2025 Task 2 官方数据集，按以下结构放置：

```
checkpoints/          # 放入 BEATs 预训练权重
└── beats_ft1.pt      #   (从 GenRep 仓库获取)

dev_data/raw/         # DCASE 官方开发集
├── ToyCar/
│   ├── train/
│   └── test/
├── ToyTrain/
├── bearing/
├── fan/
├── gearbox/
├── slider/
└── valve/

add_data/             # 额外训练集（盲测建库用）
test/                 # 官方盲测评估集
```

### 3. 特征提取

```bash
# 提取 dev_data 特征（训练 + 测试）
python extract_features.py --config configs/config.yaml --mode both

# 提取盲测数据特征
# 修改 config.yaml 中 dataset 部分指向 add_data/ 和 test/
python extract_features.py --config configs/config.yaml --mode both
```

### 4. 评估

```bash
# 运行 DCASE 官方评估（自动搜索最优层）
python evaluate.py --config configs/config.yaml
```

### 5. 生成提交文件

```bash
# 生成盲测提交 CSV
python generate_submission.py --best_layer 2

# 或指定其他层
python generate_submission.py --best_layer 4
```

---

## 评估结果

### 开发集性能 (Layer 2, dev_data)

| Machine Type | AUC (Source) | AUC (Target) | pAUC | Official Score Ω |
|:---|:---:|:---:|:---:|:---:|
| ToyCar | 0.6280 | 0.6820 | 0.5121 | 0.5986 |
| ToyTrain | 0.7412 | 0.5544 | 0.5237 | 0.5926 |
| bearing | 0.6276 | 0.7260 | 0.6211 | **0.6549** |
| fan | 0.5948 | 0.5488 | 0.5211 | 0.5532 |
| gearbox | 0.6516 | 0.6304 | 0.5716 | 0.6160 |
| slider | 0.5336 | 0.6340 | 0.5184 | 0.5576 |
| valve | 0.6808 | 0.5552 | 0.5568 | 0.5922 |
| **Average** | **0.6368** | **0.6187** | **0.5464** | **0.5950** |

> Official Score Ω = Harmonic Mean of (AUC_source, AUC_target, pAUC)

### 层级消融 (Layer Search)

| Layer | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ω | 0.569 | 0.581 | **0.595** | 0.591 | 0.592 | 0.589 | 0.586 | 0.583 | 0.567 | 0.550 | 0.537 |

**结论**: 浅层特征（Layer 2）优于深层特征，说明 BEATs 的低层注意力模式已捕获足够的声学异常线索。

---

## 工具脚本

| 脚本 | 功能 |
|:---|:---|
| `tools/generate_sci_plots.py` | 生成 UMAP、KDE、ROC、热力图等 SCI 级可视化 |
| `tools/generate_3d_gif.py` | 生成 3D UMAP 旋转 GIF 动图 |
| `tools/plot_ablations.py` | 消融实验对比可视化 |
| `tools/plot_anm_score.py` | 异常分数分布直方图 |
| `tools/analyze_distribution.py` | ToyCar 分数分布诊断分析 |
| `tools/check_dataset.py` | 数据集完整性校验（文件数、标签统计） |

```bash
# 生成可视化图表
python tools/generate_sci_plots.py
python tools/generate_3d_gif.py
python tools/plot_ablations.py
```

---

## 参考文献

1. **BEATs**: S. Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers," ICML 2023.
2. **GenRep**: Y. Liu et al., "Generative Representation Learning for Domain-Generalized Anomalous Sound Detection," DCASE 2025.
3. **DCASE 2025 Task 2**: [dcase.community/challenge2025](https://dcase.community/challenge2025/task-unsupervised-anomalous-sound-detection)

---

## License

MIT License
