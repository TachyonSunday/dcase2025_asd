# DCASE 2025 Task 2 — First-Shot Unsupervised Anomalous Sound Detection

基于 PyTorch 的无监督异常声音检测系统，支持 **ConvAE (卷积自编码器)** 和 **DANN (域对抗神经网络)** 两种模型，并附带基于 Streamlit 的交互式 Web 前端。

## 功能特性

- **完整音频特征流水线** — 原始音频 → 高通滤波 + 谱减法降噪 → Log-Mel 频谱 → 帧切分
- **ConvAE 基线模型** — 基于重构误差的异常检测
- **DANN 领域泛化** — 引入梯度反转层 (GRL) 的域对抗训练，提升跨域泛化能力
- **交互式 Web 前端** — 上传音频 → 实时梅尔瀑布图 → 异常分数判定，开箱即用

## 目录结构

```
dcase2025_asd/
├── config.yaml              # 全局超参数配置
├── data/
│   ├── raw/                 # 原始音频文件 (.wav)
│   ├── processed/           # 预处理后的 .pt 张量
│   ├── train/               # 训练集
│   └── test/                # 测试集
├── src/
│   ├── features/            # 特征工程：降噪、频谱提取、数据集
│   ├── models/              # 网络结构：ConvAE、DANN、损失函数
│   └── utils/               # 训练器、评估器
├── ui/
│   ├── app.py               # Streamlit 主入口
│   ├── components.py        # 可视化组件
│   ├── inference.py         # 推理桥接
│   └── layout.py            # 页面布局
├── checkpoints/             # 模型权重
├── logs/                    # 训练日志
├── requirements.txt
├── setup.py
└── LICENSE                  # MIT
```

## 环境要求

- **Python** >= 3.10
- **CUDA** >= 12.1 (推荐，CPU 也可运行但较慢)
- **OS** — Linux (WSL2) / macOS / Windows

## 快速开始

### 1. 创建虚拟环境

```bash
conda create -n dcase2025 python=3.10 -y
conda activate dcase2025
```

### 2. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. 准备数据

将你的音频文件 (.wav / .mp3 / .flac) 放入对应目录：

- `data/train/` — 正常训练样本 (仅正常声音，用于无监督训练)
- `data/test/`  — 测试样本 (含正常与异常，用于评估)

### 4. 特征提取

```python
from src.features.pipeline import FeaturePipeline

pipeline = FeaturePipeline(config_path="config.yaml")
pipeline.process_directory("data/train/", "data/processed/train/")
```

### 5. 训练 ConvAE 模型

```python
from src.models.conv_ae import ConvAE
from src.utils.trainer import Trainer

model = ConvAE.from_config("config.yaml")
trainer = Trainer(model, config_path="config.yaml")
trainer.train("data/processed/train/")
```

### 6. 训练 DANN 模型 (领域泛化)

```python
from src.models.dann import DANNAutoEncoder
from src.utils.trainer_dann import DANNTrainer

model = DANNAutoEncoder.from_config("config.yaml")
trainer = DANNTrainer(model, config_path="config.yaml")
trainer.train("data/processed/train/")
```

### 7. 启动 Web 前端

```bash
streamlit run ui/app.py
```

浏览器打开 `http://localhost:8501`，上传音频即可查看检测结果。

## 配置文件说明

所有超参数统一在 `config.yaml` 中管理：

| 分类 | 主要参数 |
|------|----------|
| `audio` | 采样率、时长、高通滤波截止频率、降噪开关 |
| `mel` | n_fft、hop_length、n_mels、频率范围、动态范围 |
| `frame` | 窗口大小、滑动步长、最小帧数 |
| `conv_ae` | 瓶颈层维度、通道数、卷积核大小 |
| `dann` | 域数量、域分类器维度、对抗损失权重 |
| `train` | batch_size、epochs、学习率、Early Stopping |
| `anomaly` | 阈值百分位、手动阈值覆盖 |

## 许可

[MIT](LICENSE)
