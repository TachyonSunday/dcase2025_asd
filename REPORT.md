# DCASE 2025 Task 2 工程项目报告

> 无监督异常声音检测 (First-Shot Unsupervised Anomalous Sound Detection)
> 基于 PyTorch + Streamlit 全栈实现

---

## 一、项目概览

| 维度 | 详情 |
|------|------|
| 任务 | 从机器运行声音中检测异常 (7 种机器类型) |
| 方法 | 自编码器重构误差 + 马氏距离异常评分 |
| 模型 | ConvAE (卷积自编码器) / MLP-AE (官方 Baseline) / DANN (域对抗网络) |
| 前端 | Streamlit 交互式 Web 界面 |
| 环境 | WSL2 Ubuntu, PyTorch 2.5.1+cu121, RTX 4060 GPU |
| 数据集 | DCASE 2025 T2 Development (7 机器 × 1000 train + 200 test) |

---

## 二、项目结构

```
dcase2025_asd/
├── config.yaml                     # 全局超参数 (音频/频谱/模型/训练)
├── requirements.txt                # Python 依赖
├── setup.py                        # 包安装脚本
├── README.md                       # 使用文档
├── REPORT.md                       # 本文档
├── LICENSE                         # MIT
│
├── data/
│   ├── raw/                        # 原始音频 (DCASE 数据集)
│   │   ├── ToyCar/ToyCar/{train,test}/    (1000/200 wav)
│   │   ├── ToyTrain/...                   (1000/200 wav)
│   │   ├── bearing/...                    (1000/200 wav)
│   │   ├── fan/...                        (1000/200 wav)
│   │   ├── gearbox/...                    (1000/200 wav)
│   │   ├── slider/...                     (1000/200 wav)
│   │   └── valve/...                      (1000/200 wav)
│   └── processed/                  # 预处理 .pt 张量 (按实验隔离)
│       ├── baseline_v1/            # 实验1: fmin=0, 无归一化, 马氏(z)
│       └── baseline_v2/            # 实验2: fmin=0, 无归一化, 马氏(diff)
│
├── src/
│   ├── features/                   # 特征工程
│   │   ├── denoiser.py             # AudioDenoiser: 高通滤波 + 谱减法
│   │   ├── spectrogram.py          # LogMelExtractor: STFT→Mel→log-dB
│   │   ├── dataset.py              # MachineSoundDataset: 滑动窗口帧加载
│   │   └── pipeline.py             # FeaturePipeline: 端到端编排 + 归一化
│   ├── models/                     # 神经网络
│   │   ├── conv_ae.py              # ConvAE v3: 动态层数卷积瓶颈自编码器
│   │   ├── losses.py               # MSE重构损失 + 马氏距离 + 异常分数
│   │   ├── grad_reverse.py         # GradientReversalLayer: 自定义autograd
│   │   └── dann.py                 # DANNAutoEncoder: ConvAE+GRL+DomainClassifier
│   └── utils/                      # 训练与评估
│       ├── trainer.py              # ConvAE Trainer: Early Stopping + Checkpoint
│       ├── trainer_dann.py         # DANN Trainer: 双优化器 + λ退火
│       ├── evaluator.py            # Evaluator: AUC + 阈值选取 + 推理
│       └── prepare_data.py         # DCASE 数据适配器
│
├── scripts/                        # 训练脚本
│   ├── train_toycar.py             # ConvAE ToyCar 训练 (旧)
│   ├── train_baseline_mlp.py       # MLP 基线训练 (旧)
│   └── train_all_baseline.py       # 全7机器官方复现训练 (当前)
│
├── ui/                             # Streamlit 前端
│   ├── app.py                      # 主入口 + session 管理
│   ├── inference.py                # InferenceEngine: 模型加载+推理桥接
│   ├── components.py               # 5 种 Plotly 可视化组件
│   └── layout.py                   # 侧边栏 + 三列主区域布局
│
├── results/                        # 实验结果 (按实验隔离)
│   ├── baseline_v1/                # 实验1: 5/7 完成, AUC~0.55
│   │   ├── ToyCar/{eval.json, scores.png, checkpoint.pt}
│   │   ├── ToyTrain/...  bearing/...  fan/...  gearbox/...
│   │   └── slider/ (未完成)  valve/ (未完成)
│   └── baseline_v2/                # 实验2: ToyCar, AUC=0.48
│       └── ToyCar/{eval.json, scores.png, summary.json}
│
├── checkpoints/                    # 旧模型权重 (已弃用)
└── logs/                           # 旧训练图表 (已弃用)
```

---

## 三、功能模块清单

### 3.1 特征工程

| 模块 | 类 | 功能 |
|------|-----|------|
| `denoiser.py` | `AudioDenoiser` | 2 阶 Butterworth 高通滤波 (80Hz) + noisereduce 谱减法 |
| `spectrogram.py` | `LogMelExtractor` | STFT(1024) → 128 Mel 滤波器 → 对数 dB 压缩 |
| `dataset.py` | `MachineSoundDataset` | 滑动窗口 (window_size=64, hop=32) 帧切分 |
| `pipeline.py` | `FeaturePipeline` | 加载→重采样→去噪→Mel→帧切分→保存 .pt, 支持 Z-score 归一化 |

### 3.2 神经网络模型

| 模块 | 架构 | 参数量 |
|------|------|--------|
| `conv_ae.py` | ConvAE v3: N 层 Conv2d 下采样 + 卷积瓶颈 + 对称 ConvTranspose2d 上采样 | ~186K |
| `dann.py` | ConvAE + GradientReversalLayer + DomainClassifier (MLP) | ~194K |
| `train_all_baseline.py` | MLP-AE: 640→128×5→8 + 对称解码, BN(0.01, 1e-3) | ~145K |

### 3.3 训练基础设施

| 模块 | 功能 |
|------|------|
| `trainer.py` | Adam 优化, ReduceLROnPlateau, Early Stopping (15 epoch), Checkpoint 保存/恢复 |
| `trainer_dann.py` | 双优化器联合训练, LambdaScheduler 指数退火 λ(p)=λ_final×(2/(1+e^(-γp))-1) |
| `evaluator.py` | AUC 计算, best_f1/percentile 阈值选取, Mahalanobis 模式, 推理接口 |

### 3.4 Streamlit 前端

| 组件 | 功能 |
|------|------|
| 侧边栏 | 音频上传 (wav/mp3/flac), 模型选择 (ConvAE/DANN), 阈值设置 |
| 波形图 | Plotly 交互式波形, 异常区域高亮 |
| 梅尔瀑布图 | Plotly Heatmap, 可缩放/hover 查看 dB 值 |
| 异常仪表盘 | Gauge Chart 展示异常分数, 红/黄/绿三色区域 |
| 逐帧分数 | 时序折线图, 阈值虚线标注 |
| 重建对比 | 原始频谱 vs 重建误差热力图 |

---

## 四、与官方 Baseline 的对比

### 4.1 官方仓库

我们 clone 了官方仓库 `nttcslab/dcase2023_task2_baseline_ae`，在同一份 ToyCar 数据上跑了训练，拿到了**真实的官方 AUC 基准**。

| 配置项 | 值 |
|--------|-----|
| 仓库 | https://github.com/nttcslab/dcase2023_task2_baseline_ae |
| 训练命令 | `python train.py --dataset DCASE2025T2ToyCar -d --dev --epochs 100` |
| 数据集 | 与我们相同的 ToyCar dev data (通过软链接共享) |
| 环境 | 我们现有的 conda dcase2025 (PyTorch 2.5.1+cu121) |

### 4.2 架构对比

| 维度 | 官方 | 我们的 MLP-AE (baseline_v1) | 我们的 ConvAE |
|------|------|--------------------------|--------------|
| 模型类型 | MLP 自编码器 | MLP 自编码器 | 卷积自编码器 |
| 输入维度 | 640 (128 mel × 5 帧) | 640 | 128×64 (2D patch) |
| 网络层 | 5×Linear(128)+BN+ReLU | 5×Linear(128)+BN+ReLU | 3×Conv2d+ConvTranspose2d |
| 瓶颈维度 | 8 | 8 | 16 通道卷积 |
| BN 参数 | momentum=0.01, eps=1e-3 | 默认 momentum=0.1 (v1) → 0.01 (v2) | 默认 0.1 |
| 参数量 | ~145K | ~145K | ~186K |
| 学习率 | 0.001 | 0.001 (v1) / 0.03 (v2) | 0.001 |

### 4.3 实测 AUC 对比 (ToyCar, 100 epochs, MSE 模式)

| 来源 | AUC(source) | AUC(target) | 备注 |
|------|:----------:|:----------:|------|
| **官方仓库实测** | **0.7072** | 0.5306 | 100 epochs, MSE mode |
| 我们的 baseline_v1 | 0.554* | — | 跨文件打散, 未按 source/target 分离 |
| 我们的 baseline_v2 | 0.483* | — | 640-dim 马氏距离数值不稳定 |

> \* 我们的 AUC 是全部 200 文件 (source+target) 混算, 官方 AUC(source) 仅用 source 域正常+全部异常。两者不完全可直接对比, 但差距方向明确。

### 4.4 关键实现差异

| # | 差异点 | 官方做法 | 我们的做法 | 影响 |
|---|--------|---------|-----------|------|
| 1 | **测试批处理** | 每文件一个 batch (`batch_size=n_vectors_ea_file`), 取该文件所有帧 MSE 均值 | 跨文件随机打散成 256/batch (v1) / 已修复 (v2) | ⭐⭐⭐ |
| 2 | **马氏距离对象** | 重构误差向量 diff (640-dim), 协方差 640×640 | 隐向量 z (8-dim) (v1) / diff 但矩阵求逆数值不稳定 (v2) | ⭐⭐⭐ |
| 3 | **AUC 计算域分离** | 按 source/target 域分开算 AUC, 仅 source 正常+全部异常 | 全部文件混合计算 | ⭐⭐ |
| 4 | **特征处理** | 原始 dB 值, fmin=0, 无归一化 | 初期 Z-score 归一化 + fmin=50, 后修正 | ⭐⭐ |

### 4.5 官方 ToyCar 测试集结构

```
ToyCar test (200 文件):
├── source normal:   50 文件   (section_00_source_test_normal_*.wav)
├── source anomaly:  50 文件   (section_00_source_test_anomaly_*.wav)
└── target normal:  100 文件   (section_00_target_test_normal_*.wav)
```

**关键发现**: 官方代码按域分离评估——AUC(source) 仅用 source 域正常样本+全部异常样本, AUC(target) 同理。我们的 v1 将 200 文件混合评分, 所以 AUC 数不能直接对比, 但 0.554 vs 0.707 的差距明确指示了实现差异。

---

## 五、当前状态与下一步

### 5.1 已完成

- [x] 完整工程骨架 (OOP, Type Hints, 中文注释)
- [x] 特征流水线 (去噪 + Log-Mel + 归一化)
- [x] ConvAE 模型 (v3 动态层数卷积瓶颈)
- [x] DANN 域对抗网络 (GRL + DomainClassifier + λ 退火)
- [x] Streamlit 交互式前端 (5 种可视化)
- [x] DCASE 2025 T2 全 7 机器数据集下载 & 适配
- [x] 官方 Baseline MLP-AE 复现 (640→128×5→8)
- [x] 2 轮实验 (baseline_v1/v2), 独立结果存储
- [x] 官方仓库 clone + 实测 AUC(ToyCar source)=0.707

### 5.2 核心瓶颈

**我们 0.554 vs 官方 0.707 的 0.15 AUC 差距。** 根因已定位：

1. **测试批处理方式** — 官方逐文件评分 (batch=一个文件的所有帧), 我们是跨文件打散
2. **马氏距离实现** — 官方对 640-dim 误差向量做马氏距离, 我们数值不稳定
3. **域分离评估** — 官方按 source/target 域分开算 AUC

### 5.3 推荐下一步

| 优先级 | 方案 | 预估效果 |
|--------|------|---------|
| P0 | 修复测试批处理：逐文件评分 + 按域分离 AUC 计算 | AUC 0.55 → 0.60+ |
| P1 | 对 640 维协方差做 PCA 降维后求马氏距离 | 数值稳定, AUC 接近官方 |
| P2 | 全部 7 机器用官方仓库跑分, 拿到完整基准 | 可对比验证 |

---

## 六、运行命令

```bash
# 激活环境
conda activate dcase2025

# 启动 Web 前端
streamlit run ui/app.py

# 训练 (单机器快速测试)
python scripts/train_all_baseline.py --exp my_exp --machine ToyCar --epochs 50

# 训练 (全部 7 机器)
python scripts/train_all_baseline.py --exp my_exp --epochs 100
```

## 七、技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.5.1, CUDA 12.1 |
| 音频处理 | librosa 0.11, soundfile, noisereduce |
| 前端 | Streamlit 1.58, Plotly 6.8 |
| 数值计算 | NumPy, SciPy, scikit-learn |
| 可视化 | Matplotlib, Plotly |
| 环境管理 | Conda 26.1 |
