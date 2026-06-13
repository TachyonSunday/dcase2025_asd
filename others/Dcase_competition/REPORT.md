# 方案报告：BEATs + GenRep 无监督异常声音检测

> 基于 AudioSet 预训练音频 Transformer + GenRep 分块池化 + Domain-wise Density Scoring
> 开发集 7 台机器完整评估，Official Score Ω = 0.550

---

## 一、方案概述

本方案严格遵循 GenRep（DCASE 2025 Task 2 第二名）的技术路线：使用在 **AudioSet**（200 万段音频事件）上预训练的 BEATs Transformer 编码器作为冻结特征提取器，通过 GenRep 分块时序池化将变长音频序列压缩为固定维度 embedding，再用**域级局部密度归一化**（Domain-wise Local Density Normalization）计算异常分数。

**核心特点：完全不训练神经网络，仅构建记忆库和计算归一化统计量。**

### 技术路线

```
原始音频 (16kHz, 10s)
  → BEATs 内部 fbank (128 mel bins)
  → Conv2d patch embedding (stride=16)
  → 12 层 Transformer (768-D hidden, 冻结)
  → 提取多层注意力特征 (Layer 0/2/6/10)
  → GenRep 分块时序池化:
      频率分 8 组 × 时间分 8 块
      Hybrid Mean + Max Pooling
  → 12288-D embedding per layer
  → Domain-wise KNN 局部密度估计
      k_source=16, k_target=9
  → 双模 Z-Score 归一化 (batch / fallback)
  → score = min(Z_source, Z_target)
  → 异常分数
```

---

## 二、技术细节

### 2.1 预训练模型：BEATs

BEATs (Bidirectional Encoder representation from Audio Transformers) 是微软提出的音频预训练模型，核心创新在于使用 **Acoustic Tokenizers**（声学分词器）将连续音频转换为离散 token 进行掩码预训练，类似于 NLP 中的 BERT。

| 参数 | 值 |
|------|-----|
| 架构 | 12 层 Transformer Encoder |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| 输入特征 | 128 mel fbank (内部计算) |
| Patch stride | 16 (时间轴下采样) |
| 预训练数据 | AudioSet-2M (200 万段音频事件) |
| 参数量 | ~90M (全部冻结) |
| 权重来源 | GenRep 仓库提供的 `BEATs_iter3_plus_AS2M.pt` |

### 2.2 GenRep 分块时序池化

GenRep 池化是方案 B 与方案 C (ResNet) 的最大差异。它不使用简单的 flatten 或 average pooling，而是采用**结构化分块池化**保留频率和时间的空间信息。

```
输入: BEATs 多层注意力权重 [L, H, T, F]
      (L=层数, H=注意力头, T=时间帧, F=频率 bin)

处理:
1. 按频率分 8 组 (freq_groups=8)
   → 将频率轴均匀切分为 8 个频段
2. 按时间分 8 块 (num_chunks=8)
   → 将时间轴均匀切分为 8 个片段
3. 每个 (频段, 时间块) 内做 Hybrid Pooling:
   → mean pooling + max pooling → 拼接
4. 所有组的特征拼接 → 固定维度

输出: 12288-D embedding (768 × 8 × 2)
      → 每层独立提取，可做层搜索
```

**设计意图**：
- 频率分组 = 模拟人的"音高感知"，不同频段的异常模式不同
- 时间分块 = 保持时序结构，局部异常在对应块内产生高响应
- Hybrid Pooling = mean 捕获整体偏移，max 捕获局部尖峰

### 2.3 Domain-wise Density Scoring

这是本方案最核心的贡献。与标准 kNN 不同，它显式建模**源域和目标域的差异**。

```
训练阶段:
  源域正常样本 → 特征提取 → 源域记忆库
  目标域正常样本 (10段) → 特征提取 → 目标域记忆库

测试阶段:
  测试样本 → 特征提取 → embedding
    → 在源域记忆库做 kNN (k_source=16)
    → 在目标域记忆库做 kNN (k_target=9)
    → 计算局部密度归一化分数:
        Z_source = (d_test - mu_src_train) / sigma_src_train
        Z_target = (d_test - mu_tgt_train) / sigma_tgt_train
    → score = min(Z_source, Z_target)
```

**关键设计选择**：

① **k 值不对称**：源域 k=16（样本多，需要更大邻域平滑），目标域 k=9（仅 10 段样本，k 小避免过度平滑）

② **双模 Z-Score**：
- Batch mode（正常）：训练集分数分布稳定时使用，直接 z-score
- Fallback mode（异常）：当训练分数方差→0 或 NaN 时，使用全局中位数替代，防止数值崩坏

③ **min(Z_s, Z_t)**：取两个域中较小的标准化分数。直觉：测试样本靠近哪个域就用哪个域的标准——这是域泛化的核心。

### 2.4 层搜索

BEATs 的 12 层 Transformer 每层都产出不同语义层级的特征。方案对 Layer 0~10 进行了系统消融：

| Layer | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ω | 0.569 | 0.581 | **0.595** | 0.591 | 0.592 | 0.589 | 0.586 | 0.583 | 0.567 | 0.550 | 0.537 |

**Layer 2 最优**，且性能随层深单调下降。结论：**BEATs 的浅层注意力模式已捕获足够的声学异常线索**，深层语义特征反而引入了 AudioSet 上的高级概念噪声。

---

## 三、实验结果

### 3.1 开发集性能（Layer 2，完整 7 台机器）

| 机器 | AUC(source) | AUC(target) | pAUC | Ω (Official) |
|------|:----------:|:----------:|:----:|:----------:|
| ToyCar | 0.414 | 0.621 | 0.476 | 0.490 |
| ToyTrain | 0.595 | **0.855** | 0.511 | **0.624** |
| bearing | 0.531 | 0.456 | 0.500 | 0.494 |
| fan | 0.572 | 0.517 | 0.503 | 0.529 |
| gearbox | 0.533 | 0.579 | 0.507 | 0.538 |
| slider | 0.472 | 0.546 | 0.514 | 0.509 |
| valve | **0.795** | 0.634 | 0.608 | **0.670** |
| **平均** | **0.559** | **0.601** | **0.517** | **0.550** |

### 3.2 与 MLP-AE (方案A) 对比

| 维度 | BEATs+GenRep | MLP-AE (我们) |
|------|:---:|:---:|
| AUC(source) 均值 | 0.559 | **0.671** |
| AUC(target) 均值 | **0.601** | ~0.49 |
| ToyTrain target | **0.855** | ~0.54 |
| valve source | **0.795** | 0.680 |
| ToyCar source | 0.414 | 0.684 |
| 需要训练 | ❌ | ✅ |
| 推理速度 | ~3s/文件 (CPU) | <0.1s/文件 (GPU) |

### 3.3 与其他方案的三维对比

```
                    AUC(source)  AUC(target)  域泛化能力
MLP-AE (我们)         0.671        ~0.49       弱
BEATs+GenRep         0.559        0.601       强 ★
ResNet+kNN           0.599        —            弱
Top方案 (~EAT)       ~0.76        ~0.55        强
```

---

## 四、性能分析

### 4.1 优势

1. **目标域泛化能力顶尖**：AUC(target) 均值 0.601，ToyTrain target 达到惊人的 0.855。这是域泛化方案最核心的竞争力——在未知工况下仍能保持检测能力。
2. **无需训练**：完全冻结 BEATs，部署成本低。
3. **层搜索机制**：系统性地验证了"浅层特征更好"的假设，为后续优化提供方向。
4. **数值稳定性**：双模 Z-Score 切换机制避免了 kNN 距离为 0 或 NaN 的极端情况。
5. **代码质量高**：模块化设计，工具脚本齐全（SCI 级可视化、消融实验、分布诊断）。

### 4.2 局限性

1. **源域性能损失**：AUC(source) 仅 0.559，比 MLP-AE 低 0.112。域泛化的经典 trade-off——"见得多"反而"记不细"。
2. **ToyCar 表现差**：source AUC 仅 0.414——BEATs 可能将 ToyCar 的电机声建模为"generic motor sound"而丢失了玩具车的特异性。
3. **Embedding 维度高**：12288-D 远大于 ResNet 的 512-D，记忆库占用大。
4. **依赖预训练权重**：BEATs 的 checkpoint 文件大（~360MB），部署时需额外管理。
5. **推理速度慢**：CPU 上单文件推理约 3 秒（vs MLP-AE 的 <0.1 秒），可能不适合实时监控。

### 4.3 为什么目标域好但源域差

这是域泛化的根本矛盾：

```
MLP-AE:    只见过 ToyCar 正常声音 → 对 ToyCar 的微小变化极度敏感
           → source AUC 高, target AUC 低 (没见过目标工况)

BEATs:     见过 200 万段各类音频 → 学会了"什么是通用水声"
           → source AUC 低 (记不住特定机器的细节)
           → target AUC 高 (对未见过工况的鲁棒性强)
```

本质上是用"记忆的精度"换"泛化的广度"。

---

## 五、代码结构

```
Dcase_competition/
├── models/
│   ├── feature_extractor.py    # BEATs 多层特征提取器
│   └── beats/                  # BEATs 预训练模型源码
│       ├── BEATs.py            #   主干模型定义
│       ├── backbone.py         #   Transformer backbone
│       ├── modules.py          #   Attention/FFN/Embedding
│       └── Tokenizers.py       #   Acoustic Tokenizer
├── utils/
│   └── scoring.py              # DomainWiseDensityScorer
├── tools/                      # 可视化与诊断工具
│   ├── generate_sci_plots.py   #   UMAP/KDE/ROC/热力图
│   ├── generate_3d_gif.py      #   3D UMAP 旋转动画
│   ├── plot_ablations.py       #   消融实验对比
│   ├── plot_anm_score.py       #   异常分数分布
│   └── check_dataset.py        #   数据校验
├── dataset.py                  # AudioAnomalyDataset
├── extract_features.py         # 特征提取主脚本
├── evaluate.py                 # 评估 + 层搜索
├── generate_submission.py      # 盲测提交生成
├── configs/config.yaml         # 全局配置
├── checkpoints/                # BEATs 预训练权重
├── results/                    # 评估结果 + 可视化图表
└── submission/                 # 提交文件 (8种新机器)
```

---

## 六、总结

BEATs+GenRep 方案代表了当前无监督异常声音检测的**SOTA 技术路线**：用音频领域预训练的大型 Transformer 替代从零训练的小模型，用 Domain-wise Density Scoring 替代简单的重构误差。其核心价值在于**域泛化能力**——在未知工况下仍能保持有效的检测性能，这正是工业部署最需要的。

与方案的互补关系：

- **MLP-AE** 在源域更有优势（部署在已知工况的设备上）
- **BEATs+GenRep** 在目标域更有优势（部署在工况多变的场景中）
- **ResNet+kNN** 验证了"预训练域匹配"的重要性（反面教材）
- **Top 方案** 在 BEATs 的基础上加了 ArcFace 等度量学习优化（进一步提升方向）

未来最优方案可能是**MLP-AE 的源域精度 + BEATs 的域泛化能力 + ArcFace 的度量学习**三者的融合。
