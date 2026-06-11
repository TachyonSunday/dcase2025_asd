# 无监督异常声音检测系统

> 基于 DCASE 2025 Challenge Task 2 (First-Shot Unsupervised Anomalous Sound Detection)
> 支持 MLP-AE / ConvAE / DANN 三种模型 + Streamlit 交互式前端

## 项目概览

| 维度 | 详情 |
|------|------|
| 任务 | 从机器运行声音中检测异常（无监督，仅用正常样本训练） |
| 模型 | MLP-AE (官方基线) / ConvAE (卷积自编码器) / DANN (域对抗网络) |
| 数据集 | DCASE 2025 T2 开发集 (7 种机器: ToyCar/ToyTrain/fan/gearbox/bearing/slider/valve) |
| 前端 | Streamlit Web 界面, 上传音频 → 实时频谱可视化 → 异常判定 |
| 环境 | WSL2 Ubuntu, PyTorch 2.5.1+cu121, RTX 4060 GPU |

## 实验成果

**7 台机器 MLP-AE 平均 AUC(src) = 0.671，追平官方基线 (0.670)。**

| 机器 | AUC(source) | AUC(target) |
|------|:----------:|:----------:|
| ToyCar | 0.684 | 0.381 |
| ToyTrain | 0.649 | 0.543 |
| bearing | 0.634 | 0.492 |
| fan | 0.729 | 0.330 |
| gearbox | 0.630 | 0.475 |
| slider | 0.689 | 0.493 |
| valve | 0.680 | 0.675 |

## 目录结构

```
dcase2025_asd/
├── config.yaml              # 全局超参数
├── src/
│   ├── features/            # 去噪 + Log-Mel 频谱 + Dataset 流水线
│   ├── models/              # ConvAE / DANN / GRL / Losses
│   └── utils/               # Trainer / DANNTrainer / Evaluator
├── scripts/
│   └── train_all_baseline.py  # 主训练脚本 (GPU预加载, 7机器, 断点续训)
├── ui/                      # Streamlit 前端
│   ├── app.py               # 主入口
│   ├── inference.py         # 推理引擎 (MLP/ConvAE/DANN)
│   ├── components.py        # 5种可视化 (Plotly+Matplotlib)
│   └── layout.py            # 页面布局
├── data/
│   ├── raw/                 # DCASE 数据集 (7 + 8 附加)
│   ├── processed/           # 预处理 .pt 张量
│   └── demo/                # 测试音频样本
├── results/                 # 实验结果
│   └── baseline_v5/         # v5: 7/7 完成, 追平官方基线
├── paper/                   # 论文调研 (8篇, 中英对照)
│   ├── DANN_Ganin_2016 + 翻译
│   ├── ConvAE_Masci_2011 + 翻译
│   ├── 官方综述 + 翻译
│   ├── 第一名/第二名/第三名 技术报告 + 翻译
│   └── Kim / Zheng 高质量方案 + 翻译
├── checkpoints/
├── logs/
└── REPORT.md                # 完整项目报告
```

## 快速开始

### 1. 环境

```bash
conda create -n dcase2025 python=3.10 -y && conda activate dcase2025
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pymupdf fasteners seaborn  # 论文翻译 + 官方依赖
```

### 2. 启动前端

```bash
streamlit run ui/app.py
```

浏览器打开 `http://localhost:8501`，选择 MLP-AE 模型，上传 `data/demo/` 下的测试音频即可体验。

### 3. 训练

```bash
# 全 7 机器训练 (GPU 预加载, ~15 分钟)
python scripts/train_all_baseline.py --exp my_exp --epochs 100

# 单机器快速测试
python scripts/train_all_baseline.py --exp my_exp --machine ToyCar --epochs 50
```

### 4. 下载数据集

```bash
# DCASE 2025 T2 开发集 (7 机器, ~2.4 GB)
# 从 https://zenodo.org/records/15097779 下载所有 dev_*.zip
# 解压到 data/raw/<machine_type>/

# 附加训练集 (8 新机器, ~1.9 GB)
# 从 https://zenodo.org/records/15392814 下载 eval_data_*_train.zip
```

## 三种模型

| 模型 | 架构 | 评分 | 状态 |
|------|------|------|:---:|
| MLP-AE | 640→128×5→8, BN(0.01) | 逐文件 MSE | ✅ 追平官方基线 |
| ConvAE | 3层Conv2d+卷积瓶颈 | 逐文件 MSE | 🔧 框架完成, 待跑分 |
| DANN | ConvAE + GRL + 域分类器 | MSE + 域对抗 | 🔧 框架完成, 待跑分 |

## 论文调研

8 篇论文逐字翻译为中文，`paper/` 目录下：

- 基础理论：DANN (Ganin 2016), ConvAE (Masci 2011)
- 赛事全景：DCASE 2025 Task 2 官方综述
- Top 方案：🥇Wang_MYPS  🥈Saengthong  🥉Yang_NBU
- 高质量方案：Kim (ArcFace+Center), Zheng (BEATs+EAT)

## 许可

[MIT](LICENSE)
