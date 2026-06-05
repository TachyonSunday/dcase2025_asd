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

### 4.1 架构对比

| 维度 | 官方 nttcslab/dcase2023_task2_baseline_ae | 我们的实现 |
|------|------------------------------------------|-----------|
| 模型类型 | MLP 自编码器 (纯 Linear) | ConvAE (卷积) + MLP-AE |
| 输入维度 | 640 (128 mel × 5 帧) | 640 (MLP) / 128×64 (ConvAE) |
| 网络结构 | 5×Linear(128)+BN(0.01,1e-3)+ReLU | 匹配 (MLP) / Conv2d+BN+ReLU (ConvAE) |
| 瓶颈维度 | 8 | 8 (MLP) / 16 (ConvAE) |
| 参数量 | ~145K | ~145K (MLP) / ~186K (ConvAE) |

### 4.2 关键实现差异

| # | 差异点 | 官方做法 | 我们的做法 | 影响 |
|---|--------|---------|-----------|------|
| 1 | **马氏距离对象** | 重构误差向量 (640-dim) | 隐向量 (8-dim) 或 重构误差 (但数值不稳定) | ⭐⭐⭐ |
| 2 | **测试批处理** | 每文件一个 batch, 取所有帧均值 | 跨文件随机打散 (v1) / 已修复为文件级 (v2) | ⭐⭐⭐ |
| 3 | **学习率** | 0.03 | 0.001 (v1) / 0.03 (v2) | ⭐⭐ |
| 4 | **评分阈值** | 训练分数分位数 (0.9) | best_f1 (v1) / 训练分数分位数 (v2) | ⭐ |
| 5 | **特征归一化** | 无 (原始 dB 值) | 先有 Z-score 后改为无 | ⭐⭐ |
| 6 | **BN momentum** | 0.01 | 默认 0.1 → 已修复为 0.01 | ⭐ |

### 4.3 实验结果对比

| 机器 | 官方 AUC (估) | baseline_v1 AUC | baseline_v2 AUC |
|------|:------------:|:---------------:|:---------------:|
| ToyCar | ~0.65 | 0.554 | 0.483 |
| ToyTrain | ~0.70 | 0.584 | — |
| bearing | ~0.75 | 0.558 | — |
| fan | ~0.70 | 0.529 | — |
| gearbox | ~0.70 | 0.569 | — |
| slider | ~0.75 | 未完成 | — |
| valve | ~0.75 | 未完成 | — |

---

## 五、当前状态与瓶颈

### 5.1 已完成

- [x] 完整工程骨架 (OOP, Type Hints, 中文注释)
- [x] 特征流水线 (去噪 + Log-Mel + 归一化)
- [x] ConvAE 模型 (v3 动态层数卷积瓶颈)
- [x] DANN 域对抗网络 (GRL + DomainClassifier + λ 退火)
- [x] Streamlit 交互式前端 (5 种可视化)
- [x] DCASE 2025 T2 全 7 机器数据集下载 & 适配
- [x] 官方 Baseline MLP-AE 复现 (640→128×5→8)
- [x] 2 轮实验 (baseline_v1/v2), 独立结果存储

### 5.2 未解决的核心问题

**AUC 停留在 0.55 无法达到官方 ~0.65 水平。** 根本原因：

1. **马氏距离实现不对** — 官方对 640 维重构误差向量做马氏距离，我们用 8 维隐向量。v2 尝试修复但 640×640 协方差矩阵求逆数值不稳定
2. **学习率差异** — 官方 lr=0.03 可在同 epoch 内获得更好的收敛

### 5.3 推荐下一步

| 优先级 | 方案 | 预估效果 |
|--------|------|---------|
| P0 | 直接 clone 官方仓库跑分 → 验证数据集和基线正确性 | 确认基准线 |
| P1 | 回退到 MSE + 文件级评分 (不用马氏距离) | AUC ~0.60 |
| P2 | 对 640 维协方差做 PCA 降维后再求马氏距离 | 数值稳定 |

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
