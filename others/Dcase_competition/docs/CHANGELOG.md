# 更新日志 (CHANGELOG)

## [Unreleased] - 2026-06-11

### 🎯 重大更新：DCASE 2025 官方评估协议对齐

#### 新增功能

**1. Domain-wise Local Density Normalization 打分器** (`utils/scoring.py`)
- ✅ 实现 GenRep 技术报告中的域感知局部密度归一化算法
- ✅ 分域内存库构建 (source_memory_bank + target_memory_bank)
- ✅ 局部密度预计算 (K_s=16, K_t=9)
- ✅ 密度归一化异常分数: `score = min(score_s, score_t)`
- ✅ 新增 `DomainWiseDensityScorer` 类替代传统 `KNNScorer`
- ✅ 保留 `KNNScorer` 用于对比实验

**2. DCASE 2025 官方评估指标** (`evaluate.py`)
- ✅ 按机器类型分别评估 (7 个机器独立计算)
- ✅ AUC_source: 仅针对 source 域测试样本
- ✅ AUC_target: 仅针对 target 域测试样本
- ✅ pAUC: 针对所有测试样本 (max_fpr=0.1)
- ✅ Official Score Ω: 三者的调和平均数
- ✅ 汇总所有机器的平均 Official Score
- ✅ 输出 CSV 和 YAML 格式结果

**3. Domain-Generalized 数据集支持** (`dataset.py`)
- ✅ 从文件名解析 source/target 域标签
- ✅ 双标签系统: `anomaly_label` + `domain_label`
- ✅ 返回原始波形而非 Log-Mel 频谱图
- ✅ 移除 Log-Mel 计算逻辑，直接配合 CED-Tiny

**4. CED-Tiny 特征提取器** (`models/feature_extractor.py`)
- ✅ 使用 Hugging Face `mispeech/ced-tiny` 预训练模型
- ✅ 输入: 16kHz 单声道原始波形
- ✅ 输出: 384 维 Embedding (最后两层 Transformer 拼接)
- ✅ 提取最后两层 Transformer Embedding + Mean Pooling
- ✅ 完全冻结模型参数

#### 重构

**1. 特征提取流程**
- 从: ResNet18 (512维, ImageNet预训练, Log-Mel输入)
- 到: CED-Tiny (384维, AudioSet预训练, 原始波形输入)
- 优势: 音频专用预训练，更适合 DCASE 任务

**2. 异常打分算法**
- 从: 传统 KNN (K=5, 无域感知)
- 到: Domain-wise Local Density Normalization (K_s=16, K_t=9, 域感知)
- 优势: 考虑域偏移和局部密度，更鲁棒

**3. 评估指标**
- 从: 单一 AUC + pAUC (混合所有样本)
- 到: AUC_source + AUC_target + pAUC + Official Score Ω (按机器、按域)
- 优势: 完全对齐 DCASE 2025 官方评估协议

**4. 数据集接口**
- 从: 返回 Log-Mel 频谱图 + 单一标签
- 到: 返回原始波形 + 双标签 (anomaly + domain)
- 优势: 支持域感知训练和评估

#### 配置文件更新

**`configs/config.yaml`**
```yaml
# 模型参数
model:
  name: "ced-tiny"
  pretrained: true
  embedding_dim: 384
  pool_mode: "concat"
  input_type: "waveform"

# 打分器参数
knn:
  k_source: 16
  k_target: 9
  k_score: 5
  metric: "euclidean"
  algorithm: "auto"
  n_jobs: -1
```

#### 依赖更新

**`requirements.txt`**
```txt
transformers>=4.30.0        # Hugging Face 模型加载
huggingface_hub>=0.20.0     # 模型下载和缓存
```

#### 文档

- ✅ `DCASE2025_EVALUATION.md`: 官方评估协议详细说明
- ✅ `DOMAIN_GENERALIZATION.md`: 域泛化数据集重构说明
- ✅ `REFACTOR_SUMMARY.md`: CED-Tiny 模型重构总结
- ✅ `CHANGELOG.md`: 本更新日志

### 🔧 技术细节

#### 域标签解析

```python
def extract_domain_label(filename: str) -> int:
    """
    从文件名提取域标签
    - section_00_source_train_normal_001.wav → 0 (source)
    - section_00_target_train_normal_001.wav → 1 (target)
    """
    if "source" in filename.lower():
        return 0
    elif "target" in filename.lower():
        return 1
    else:
        return 0  # 默认 source
```

#### 机器类型解析

```python
def extract_machine_type(filename: str) -> str:
    """
    从文件名提取机器类型
    - ToyCar_train_normal_001.wav → "ToyCar"
    - bearing_test_anomaly_042.wav → "bearing"
    """
    MACHINE_TYPES = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    for machine in MACHINE_TYPES:
        if machine.lower() in filename.lower():
            return machine
    return "Unknown"
```

#### 调和平均数计算

```python
def harmonic_mean(values: list) -> float:
    """
    H = n / (1/x₁ + 1/x₂ + ... + 1/xₙ)
    
    用于计算 Official Score Ω
    """
    if not values or any(v <= 0 for v in values):
        return 0.0
    n = len(values)
    return n / sum(1.0 / v for v in values)
```

### 📊 性能对比

| 特性 | 旧版本 | 新版本 | 改进 |
|------|--------|--------|------|
| 特征提取器 | ResNet18 (512维) | CED-Tiny (384维) | 音频专用预训练 |
| 输入类型 | Log-Mel 频谱图 | 原始波形 | 更简单，信息保留更完整 |
| 打分算法 | KNN (K=5) | Domain-wise Density (K_s=16, K_t=9) | 域感知，密度归一化 |
| 评估指标 | AUC + pAUC | AUC_s + AUC_t + pAUC + Ω | 完全对齐官方 |
| 域支持 | ❌ | ✅ | 支持 source/target 域 |
| 按机器评估 | ❌ | ✅ | 7 个机器独立评估 |
| Official Score | ❌ | ✅ | 调和平均数 |

### 🚀 使用方法

#### 完整流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 提取特征
python extract_features.py --config configs/config.yaml --mode both

# 3. 运行官方评估
python evaluate.py --config configs/config.yaml

# 4. 查看结果
# - 控制台输出: 7 个机器的详细指标 + 平均值
# - CSV: results/dcase2025_results_YYYYMMDD_HHMMSS.csv
# - YAML: results/dcase2025_summary_YYYYMMDD_HHMMSS.yaml
```

#### 自定义参数

```bash
python evaluate.py \
    --config configs/config.yaml \
    --k_source 20 \
    --k_target 12 \
    --k_score 7 \
    --metric cosine \
    --output_dir ./my_results
```

### 🐛 已知问题

1. **CED-Tiny 首次加载慢**: 首次运行会从 Hugging Face 下载模型 (~100MB)
   - 解决: 使用 `HF_HUB_OFFLINE=1` 环境变量离线模式

2. **Windows 缓存警告**: Windows 下 Hugging Face 缓存使用符号链接会警告
   - 解决: 设置 `HF_HUB_DISABLE_SYMLINKS_WARNING=1`

3. **域标签默认值**: 无法识别域的文件名默认为 source
   - 建议: 确保文件名符合 DCASE 命名规范

### 🔮 未来计划

- [ ] 支持多模型集成 (Ensemble)
- [ ] 实现 DANN (Domain-Adversarial Neural Network)
- [ ] 添加可视化分析工具 (t-SNE, UMAP)
- [ ] 支持提交 DCASE 官方评估系统
- [ ] 添加实时推理接口

### 📚 参考文献

1. **CED-Tiny**
   - Hugging Face: https://huggingface.co/mispeech/ced-tiny
   - Paper: "CED: A Compact and Efficient Audio Classifier"

2. **GenRep**
   - Paper: "Generative Representation Learning for Domain-Generalized Anomalous Sound Detection"
   - DCASE 2025 Workshop

3. **DCASE 2025 Task 2**
   - Official Website: https://dcase.community/challenge2025/task-unsupervised-anomalous-sound-detection
   - Baseline: https://github.com/dcase-community/dcase2025_task2_baseline

4. **Domain-Adversarial Training**
   - Paper: "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2015)

### 👥 贡献者

- 项目维护者: [Your Name]
- 主要贡献: CED-Tiny 集成、域泛化支持、官方评估协议实现

### 📄 许可证

MIT License

---

## [1.0.0] - 2026-06-01

### 初始版本

- ✅ 基础项目结构搭建
- ✅ ResNet18 特征提取器
- ✅ KNN 异常打分
- ✅ AUC + pAUC 评估
- ✅ 数据下载和校验脚本
- ✅ 配置文件系统

---

**注意**: 此更新日志记录了项目的重要变更。详细的技术实现请参考对应的文档文件。
