# 无监督异常声音检测技术研究 — PPT 大纲

> 以 DCASE 2025 Challenge Task 2 为例 | 20 页 | 20 分钟

---

## 第一部分：背景与问题定义（4 页，3 分钟）

### Slide 1 — 封面
- 标题：**无监督异常声音检测技术研究**
- 副标题：以 DCASE 2025 Challenge Task 2 为例
- 姓名 / 学号 / 日期

> 📊 配图建议：从 `paper/01_官方综述...md` 的图 1（baseline 结果图）截取一张代表性图形做封面背景

---

### Slide 2 — 问题定义
- 工厂场景：电机、齿轮箱、风扇、阀门等设备运行
- 目标：安装麦克风，从声音中自动识别设备是否故障
- 传统方式靠人工巡检 → 效率低、不及时
- 输入：一段 10 秒音频 → 输出：正常 / 异常 + 异常分数

> 📊 配图建议：自行绘制一个简洁的 "麦克风 → 设备 → 检测结果" 示意图

---

### Slide 3 — 三大技术难点
| 难点 | 说明 |
|------|------|
| ① 无监督 | 训练时只有正常声音，不知道故障长什么样 |
| ② 域偏移 | 训练工况（转速/负载）与测试不同 → 正常声音也可能被误报 |
| ③ First-Shot | 评估时是完全不同的新机器类型，参数不能单独调整 |

> 📊 配图建议：画一个「源域 vs 目标域」对比示意图 → 左半 = 已知机器（训练），右半 = 全新机器（测试），标注域偏移

---

### Slide 4 — 数据集与评价指标
| 项目 | 详情 |
|------|------|
| 来源 | DCASE 2025 Challenge Task 2 开发集 |
| 机器类型 | 7 种：ToyCar / ToyTrain / fan / gearbox / bearing / slider / valve |
| 训练集 | 每台约 1000 段正常声音（10 秒/段） |
| 测试集 | 每台 200 段（100 正常 + 100 异常），含 source/target 两个域 |
| 评分指标 | Ω = harmonic mean( AUC_source, AUC_target, pAUC ) |

> 📊 配图建议：从 `paper/01_官方综述...md` 的表 1（Baseline results for development dataset）截图

---

## 第二部分：相关技术调研（6 页，6 分钟）

### 路线全景

三种主流技术路线：

```
路线 A ── 自编码器 + 重构误差        → 官方基线，我们复现
路线 B ── 冻结预训练模型 + kNN        → 第二名，完全不训练
路线 C ── 预训练模型微调 + 度量学习    → 第一名，当前最优
```

---

### Slide 5 — 路线 A：自编码器（官方基线）— 原理

**核心思想**：训练一个网络把正常声音"压缩再还原"。测试时看还原误差——误差小 = 正常（模型见过），误差大 = 异常（模型没见过）。

```
训练阶段:  正常声音 → [编码器] → 8维瓶颈 → [解码器] → 还原声音
                         └──→ 最小化 |输入 - 还原|² (MSE损失)

测试阶段:  任意声音 → [编码器] → [解码器] → |输入 - 还原|² = 异常分数
                                                      ↑ 越大越异常
```

**网络结构**：
```
输入(640维) → Linear128+BN+ReLU ×4 → Linear8 (瓶颈)
            → Linear128+BN+ReLU ×4 → Linear640 (输出)

参数：~145K，轻量可部署边缘设备
```

> 📊 配图建议：画一个漏斗形的编码器-解码器结构图。从 `paper/01_官方综述...md` 第三页截取 baseline 架构描述的文字框。

---

### Slide 6 — 路线 A：两种评分模式

| 模式 | 异常分数计算 | 特点 |
|------|------------|------|
| **MSE 模式** | 逐帧 MSE → 文件级均值 | 简单直接 |
| **马氏距离模式** | 重构误差向量(640维)的马氏距离 | 对域偏移更鲁棒 |

马氏距离公式（来自论文公式 6-8）：
```
异常分数 = min( D_source(ψ, r(ψ)), D_target(ψ, r(ψ)) )
D(ψ) = (r(ψ) - ψ)^T · Σ^{-1} · (r(ψ) - ψ)
```

- Σ 是训练集所有重构误差向量的协方差矩阵（640×640）
- source 和 target 域分别建一个 Σ → 测试时取两个域中较小的马氏距离

> 📊 配图建议：从 `paper/01_官方综述...md` 截取公式 (5)-(8) 附近的文字段落

---

### Slide 7 — 路线 A：官方基线成绩

从 `paper/01_官方综述...md` 的表 1：

| 机器 | AUC(source) | AUC(target) | pAUC |
|------|:----------:|:----------:|:----:|
| ToyCar | 71.05% | 53.32% | 49.79% |
| fan | 70.96% | 38.75% | 49.46% |
| gearbox | 64.80% | 50.49% | 52.49% |
| valve | 63.53% | 67.18% | 57.35% |
| ... | ... | ... | ... |

特点：source 域 AUC 尚可（60-78%），target 域显著下降（38-67%），说明**域偏移是主要瓶颈**。

> 📊 配图建议：**直接截取论文 Table 1 全部数据**（`paper/01_官方综述...md` 中搜索 "ToyCar" 附近），这是最重要的基准数据

---

### Slide 8 — 路线 B：第二名 GenRep 方案

**论文**：`paper/03_第二名_Saengthong_GenRep冻结编码器_中文翻译.md`
**代码**：https://github.com/Phuriches/GenRepASD（已 clone 到本地）

**核心思想**：完全不训练！用 5 个冻结的预训练音频编码器提取特征 → kNN 最近邻搜索 → 域归一化 → 异常分数。

```
训练数据:  源域正常音频 → 冻结编码器 → embedding → 源域记忆库
          目标域正常音频 → 冻结编码器 → embedding → 目标域记忆库

测试时:   待测音频 → 冻结编码器 → embedding → 在两个记忆库各做 kNN
               → 取 min(源域距离, 目标域距离) → 异常分数
```

> 📊 配图建议：从 `paper/03_第二名...md` 中截取 **图 1（Figure 1）**，对比了微调路线和冻结路线的流程差异——这是全场最有价值的图之一

---

### Slide 9 — 路线 B：GenRep 关键技术细节

**5 个预训练编码器**：
| 编码器 | 参数量 | 来源 |
|--------|:-----:|------|
| BEATs | ~90M | 微软 |
| M2D CLAP | ~300M | NTT |
| EAT large | ~300M | 自监督 |
| SSLAM | ~300M | 自监督 |
| CED tiny | **5.49M** | 最轻量 |

**域归一化是关键**：论文共对比了 4 种归一化策略——测试时 Z-score、域级 Z-score、局部密度归一化、域级局部密度归一化。

**最小模型 CED tiny（5.49M）官方得分 62.15**，超过基线 5.89 分。

> 📊 配图建议：从 `paper/03_第二名...md` 中截取 **图 2（Figure 2）**，展示了不同编码器和归一化策略的对比效果；截取 **表 3** 的 7 台机器具体数据

**代码开源**，仓库结构简析：`run_genrep_dcase2023.py` → 加载编码器 → 建记忆库 → kNN → 归一化 → 输出分数。

---

### Slide 10 — 路线 C：第一名 Wang_MYPS 方案

**论文**：`paper/02_第一名_Wang_MYPS_预训练模型增强ASD_中文翻译.md`

**核心思想**：EAT 预训练编码器 + ArcFace 属性/域分类微调 + KNN 后端。

```
音频 → EAT 编码器(88M) → 平均池化 → ArcFace 分类头 → fine-tune
                                           ↓
                                  两种微调: FFT(全量) 或 LoRA(低秩)
                                           ↓
                                  正常数据 embedding → KNN(k=1)
                                           ↓
                                  余弦距离 = 异常分数
```

**创新点**：
- ArcFace 损失函数（人脸识别迁移到音频异常检测）
- LoRA 微调降低训练成本
- 补充数据（干净机器声/噪声）直接加入训练或做数据增强

**成绩**：开发集 harmonic mean = 60.9%，官方最终得分 61.628（🥇）

> 📊 配图建议：从 `paper/02_第一名...md` 中截取 **图 1（Figure 1：Architecture of proposed ASD system）**——展示从 mel-spectrogram → EAT encoder → classifier → KNN 的完整流程；截取 **表 1（Table 1）**——7 台机器 AUC(source/target)/pAUC 与基线对比

---

## 第三部分：系统设计与实现（4 页，4 分钟）

### Slide 11 — 系统总体架构

```
上传音频 → [特征提取流水线] → [MLP 自编码器] → [异常评分] → [前端展示]
              │                      │                │
         去噪 / STFT /          640→128→8→         逐帧 MSE →
         Mel / 帧堆叠           128→640             文件级分数
```

> 📊 配图建议：画一张垂直流程图，标注每个模块的技术选型

---

### Slide 12 — 特征提取流水线（核心）

```
原始 wav (16kHz, 10秒, 160000 采样点)
  → 高通滤波(80Hz, 2阶 Butterworth)
  → 谱减法降噪(noisereduce)
  → STFT(n_fft=1024, hop_length=512) → 每帧 513 个频率分量
  → 128 个 Mel 滤波器 → 人耳感知尺度
  → dB 对数压缩 → 动态范围压缩
  → 5 帧连续堆叠 → 640 维输入向量 (128 mel × 5 帧)
```

参数与官方基线完全一致。

> 📊 配图建议：用 `logs/mel_explained.png`（四宫格解释图）中的左上（波形）和左下（梅尔频谱）子图，或截取该图的 2-3 个子图

---

### Slide 13 — 模型结构与训练

**网络结构**：
```
输入 640 → Linear(128)+BN+ReLU × 4 → 瓶颈(8)+BN+ReLU
         → Linear(128)+BN+ReLU × 4 → 输出 640

参数量：145K | BN momentum=0.01, eps=1e-3 | 无 Dropout
```

**训练配置**：
| 参数 | 值 |
|------|-----|
| 优化器 | Adam(lr=0.001) |
| Batch Size | 2048（GPU 预加载） |
| Epochs | 100 |
| 损失函数 | MSE |
| 学习率调度 | ReduceLROnPlateau(factor=0.5, patience=10) |

> 📊 配图建议：画一个漏斗形的网络结构图，标注每层维度

---

### Slide 14 — 异常评分机制

```
测试音频所有帧 → 每帧 MSE → frame_scores[0..N]
                                    ↓
               均值 = file_score         top-10% 均值 = topk_score
                    ↓                          ↓
          用于整体评估                  用于局部异常捕获
```

**域分离评估**（匹配官方标准）：
- AUC(source)：仅用 source 域正常样本 + 全部异常样本
- AUC(target)：仅用 target 域正常样本 + 全部异常样本

> 📊 配图建议：画一个 "逐帧分数 → 综合分数" 的聚合流程图

---

## 第四部分：实验与结果（4 页，5 分钟）

### Slide 15 — 实验设置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop (8GB) |
| 框架 | PyTorch 2.5.1 + CUDA 12.1 |
| 数据集 | DCASE 2025 T2 全部 7 种机器 |
| 训练 | 每种机器独立训练 100 epochs |
| 评估 | 文件级 MSE，source/target 域分离计算 AUC |

**GPU 预加载优化**：
- 训练数据全部加载至显存（~650MB / 8GB）
- 消除磁盘 I/O 瓶颈
- 训练速度：30 秒/epoch → 1 秒/epoch（30× 提升）

> 📊 配图建议：画一个前后对比图（优化前 GPU 利用率 42% → 优化后接近 100%）

---

### Slide 16 — 复现结果（核心页）

**与官方基线 7 台机器对比**：

| 机器 | 我们 AUC(src) | 官方 AUC(src) | 差距 |
|------|:----------:|:----------:|:----:|
| ToyCar | 0.684 | 0.711 | -0.027 |
| ToyTrain | 0.649 | 0.618 | **+0.031** |
| bearing | 0.634 | 0.665 | -0.031 |
| fan | 0.729 | 0.710 | **+0.019** |
| gearbox | 0.630 | 0.648 | -0.018 |
| slider | 0.689 | 0.701 | -0.012 |
| valve | 0.680 | 0.635 | **+0.045** |
| **平均** | **0.6706** | **0.6696** | **持平** |

- 7 台平均 AUC(source) 与官方基线仅差 0.001
- 4 台超过官方、3 台略低
- 差距均在官方报告的标准差范围（±0.5%）内

> 📊 配图建议：**使用 `logs/auc_comparison.png`**（7 机器四柱对比图）——这是 PPT 中最重要的数据图

---

### Slide 17 — 实验迭代历程

**三轮迭代，每轮解决一个关键问题**：

| 版本 | AUC(src) | 关键改进 |
|:---:|:--------:|---------|
| v1 | 0.554 | 初版：跨文件打散批处理，AUC 低 |
| v2 | 0.483 | 尝试 640 维马氏距离，协方差矩阵求逆数值不稳定 |
| v3-v5 | 0.680 | **P0 修复**：逐文件评分 + 域分离 AUC → 追平基线 |

**5 个关键差异（vs 官方代码）**：
1. 测试批处理方式（逐文件 vs 跨文件打散）——最关键
2. 马氏距离计算对象（640维误差向量 vs 8维隐向量）
3. AUC 按域分离计算
4. 特征归一化（无 vs Z-score）
5. 学习率（0.001 vs 0.03）

> 📊 配图建议：画一张从 v1 到 v5 的 AUC 上升折线图

---

### Slide 18 — Web 前端演示

**功能**：
- 上传音频（wav/mp3/flac）→ 实时推理
- 四屏展示：波形图 + 梅尔瀑布图 + 异常仪表盘 + 逐帧分数
- 支持 MLP / ConvAE / DANN 三种模型切换
- 7 种机器模型一键选择

**部署**：
- Streamlit Cloud 免费托管，浏览器即用
- 无需安装 Python 或任何依赖

> 📊 配图建议：**截取 Streamlit 前端实际运行截图**，包含侧边栏和四屏结果展示；或录制一段 30 秒 GIF

---

## 第五部分：总结与展望（2 页，2 分钟）

### Slide 19 — 工作总结

| 类别 | 完成内容 |
|------|---------|
| 特征工程 | 完整流水线（去噪/频谱/Dataset/归一化） |
| 基线复现 | MLP 自编码器，7 台全部追平官方基线 |
| 文献调研 | 8 篇论文逐字翻译（涵盖 DANN/ConvAE 理论 + 比赛 Top3 方案） |
| 工程优化 | GPU 预加载 30× 加速、断点续训、实验隔离存储 |
| 前端系统 | Streamlit 交互式界面，已部署 Streamlit Cloud |
| 源码对比 | 官方 baseline 仓库 + GenRep 仓库已 clone，核心代码分析完毕 |

---

### Slide 20 — 不足与展望

**当前不足**：
- 复现停留在基线水平（AUC 0.67），未引入预训练模型
- ConvAE 和 DANN 框架已实现但未在 DCASE 数据上完整的实验验证
- 缺失马氏距离评分模式的实现

**未来方向**：
- 集成预训练编码器（BEATs / EAT）替代 MLP → 预期 AUC 提升 5-8%
- DANN 域对抗训练配合预训练模型 → 解决 target 域 AUC 低的问题
- 提交 DCASE 2026 Challenge 进行官方评估

**技术洞察**：
- 发现了 Mel 频谱可视为"灰度图像"，现有 Top 方案本质上使用计算机视觉技术处理
- 这为未来工作打开了路径：直接借用 CV 预训练模型（ResNet/ViT）处理频谱

---

### Slide 21 — 参考文献

1. Nishida et al., "Description and Discussion on DCASE 2025 Challenge Task 2", DCASE Workshop 2025
2. Ganin & Lempitsky, "Domain-Adversarial Training of Neural Networks", JMLR 2016
3. Masci et al., "Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction", ICANN 2011
4. Wang, "Pre-trained Model Enhanced Anomalous Sound Detection System for DCASE2025 Task2", DCASE 2025 Tech. Report (🥇)
5. Saengthong & Shinozaki, "GenRep for First-Shot UASD of DCASE 2025 Challenge", DCASE 2025 Tech. Report (🥈)
6. Yang, "A Two Stage Fusion Anomaly Detection Approach for Task2", DCASE 2025 Tech. Report (🥉)
7. Kim et al., "AISTAT Lab System for DCASE 2025 Challenge Task 2", DCASE 2025 Tech. Report
8. Zheng et al., "SITU-AITHU System for DCASE 2025 ASD Challenge", DCASE 2025 Tech. Report

---

### Slide 22 — Q&A

> 准备回答的问题：
> - 为什么 MLP 能追平基线但 ConvAE/DANN 没跑分？
> - 和官方代码的具体差异在哪？如何发现的？
> - 为什么不直接用比赛第一名的开源代码？
> - DANN 在这个任务中的作用是什么？理论基础是什么？

---

## 附录：插图来源速查表

| Slide | 插图内容 | 来源 |
|:-----:|---------|------|
| 2 | 工厂检测示意图 | 自行绘制 |
| 3 | 源域 vs 目标域 | 自行绘制 |
| 4 | 数据集结构表 | `paper/01_官方综述...md` |
| 4 | Baseline 结果表 (Table 1) | `paper/01_官方综述...md` 的表 1 |
| 5 | 漏斗形网络结构 | 自行绘制 |
| 7 | 官方基线 Table 1 全部数据 | `paper/01_官方综述...md` 搜索 "ToyCar" |
| 8 | GenRep 图 1 (路线对比) | `paper/03_第二名...md` 的 Figure 1 |
| 9 | GenRep 图 2 + 表 3 | `paper/03_第二名...md` 的 Figure 2, Table 3 |
| 10 | Wang_MYPS 图 1 (架构) + 表 1 (结果) | `paper/02_第一名...md` 的 Figure 1, Table 1 |
| 12 | 梅尔频谱四宫格 | `logs/mel_explained.png` |
| 15 | GPU 优化前后对比 | 自行绘制 |
| 16 | 7 机器 AUC 对比柱状图 | `logs/auc_comparison.png` |
| 17 | 迭代历程折线图 | 自行绘制（v1→v5 AUC 变化） |
| 18 | 前端截图 | 实际运行 streamlit 后截图 |
