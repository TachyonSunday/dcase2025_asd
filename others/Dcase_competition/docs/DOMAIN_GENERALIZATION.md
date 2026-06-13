# Domain-Generalized 数据集重构总结

## 概述
将 `AudioAnomalyDataset` 重构为支持 Domain-Generalized 设定，能够解析 source/target 域标签，并移除 Log-Mel 频谱图计算逻辑，直接返回 16kHz 原始波形。

## 主要变更

### 1. dataset.py - 核心重构

#### 新增功能
- ✅ **域标签解析**: 从文件名中解析 source/target 域
  - 使用正则表达式匹配 `_source_` 或 `_target_` 模式
  - 支持 fallback 字符串匹配
  - 默认未识别域标记为 source (0)

- ✅ **双标签系统**:
  - `anomaly_label`: 0=正常, 1=异常
  - `domain_label`: 0=source, 1=target

- ✅ **移除 Log-Mel 计算**:
  - 删除 `_compute_log_mel()` 方法
  - 删除相关的配置参数 (n_mels, n_fft, hop_length, win_length, fmin, fmax)
  - 直接返回 16kHz 单声道原始波形

#### 返回格式变更

**旧格式** (Log-Mel 频谱图):
```python
{
    "mel": tensor,        # shape: (1, n_mels, T)
    "label": int,         # 0=正常, 1=异常
    "file_path": str,
    "filename": str,
}
```

**新格式** (原始波形 + 域标签):
```python
{
    "waveform": tensor,        # shape: (num_samples,) - 16kHz 单声道
    "anomaly_label": int,      # 0=正常, 1=异常
    "domain_label": int,       # 0=source, 1=target
    "file_path": str,
    "filename": str,
}
```

#### 新增辅助方法
```python
get_anomaly_labels()      # 返回所有异常标签
get_domain_labels()       # 返回所有域标签
get_source_count()        # 返回 source 域样本数
get_target_count()        # 返回 target 域样本数
get_domain_name(label)    # 将域标签转为可读名称 ("source"/"target")
```

### 2. extract_features.py - 适配新格式

#### 变更内容
- 使用 `"waveform"` 替代 `"mel"`
- 使用 `"anomaly_label"` 替代 `"label"`
- 新增收集 `"domain_label"`
- 保存元数据包含 `anomaly_labels` 和 `domain_labels`

#### 返回值变更
```python
# 旧
return embeddings, labels, file_paths

# 新
return embeddings, anomaly_labels, domain_labels, file_paths
```

### 3. evaluate.py - 适配新字段名

#### 变更内容
- `load_features()` 返回 `anomaly_labels` 和 `domain_labels`
- 所有使用 `"labels"` 的地方改为 `"anomaly_labels"`
- 评估摘要包含域标签信息

### 4. models/feature_extractor.py - 适配新方法

#### extract_features() 方法变更
```python
# 旧返回值
return (embeddings, labels, file_paths)

# 新返回值
return (embeddings, anomaly_labels, domain_labels, file_paths)
```

## 文件名解析逻辑

### DCASE 2025 文件名约定

```
section_00_source_train_normal_0000_n_B.wav  → source, normal
section_00_target_train_normal_0000_n_B.wav  → target, normal
section_00_source_test_anomaly_0000_n_B.wav  → source, anomaly
section_00_target_test_anomaly_0000_n_B.wav  → target, anomaly
```

### 解析方法

```python
import re

# 正则表达式匹配 _source_ 或 _target_
domain_pattern = re.compile(r"[_\-.](source|target)[_\-.]")

filename = "section_00_source_train_normal_0000_n_B.wav"
match = domain_pattern.search(filename.lower())
if match:
    domain_str = match.group(1)  # "source" 或 "target"
    domain_label = 0 if domain_str == "source" else 1
```

## 使用示例

### 加载数据集
```python
from dataset import AudioAnomalyDataset
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 训练集 (仅正常样本，包含 source 和 target)
train_ds = AudioAnomalyDataset(
    audio_dir='dev_data/raw/fan/train',
    config=config,
    mode='train'
)

# 获取样本
sample = train_ds[0]
waveform = sample['waveform']        # shape: (160000,)
anomaly_label = sample['anomaly_label']  # 0 (正常)
domain_label = sample['domain_label']    # 0 (source) 或 1 (target)

# 统计信息
print(f"Source 域: {train_ds.get_source_count()}")
print(f"Target 域: {train_ds.get_target_count()}")
```

### 特征提取
```python
from models.feature_extractor import get_feature_extractor
from dataset import create_dataloader

# 创建数据加载器
dataloader = create_dataloader(
    audio_dir='dev_data/raw/fan/train',
    config=config,
    mode='train'
)

# 创建特征提取器
model = get_feature_extractor(config)

# 提取特征
embeddings, anomaly_labels, domain_labels, file_paths = \
    model.extract_features(dataloader, device='cuda')

print(f"Embeddings: {embeddings.shape}")  # (N, 384)
print(f"Anomaly labels: {len(anomaly_labels)}")
print(f"Domain labels: {len(domain_labels)}")
```

## 数据集统计输出

运行数据集时会显示详细的统计信息：

```
[Dataset] mode=train | 目录: dev_data/raw/fan/train | 样本数: 1000 | 正常: 1000 | 异常: 0 | source: 500 | target: 500
```

## 兼容性说明

### 已更新的脚本
- ✅ `dataset.py`
- ✅ `models/feature_extractor.py`
- ✅ `extract_features.py`
- ✅ `evaluate.py`
- ✅ `configs/config.yaml`
- ✅ `requirements.txt`
- ✅ `models/__init__.py`

### 需要注意的旧代码
如果有其他脚本引用了旧的字段名，需要更新：
- `"mel"` → `"waveform"`
- `"label"` → `"anomaly_label"`
- 新增 `"domain_label"` 字段

## 下一步：Domain-Adaptive KNN

有了 domain_labels 后，可以实现更高级的异常检测策略：

### 方案 1: 分域 KNN
```python
# 分别为 source 和 target 域训练 KNN
source_indices = [i for i, d in enumerate(train_domain_labels) if d == 0]
target_indices = [i for i, d in enumerate(train_domain_labels) if d == 1]

source_embeddings = train_embeddings[source_indices]
target_embeddings = train_embeddings[target_indices]

# 训练两个 KNN 模型
scorer_source = KNNScorer(k=5).fit(source_embeddings)
scorer_target = KNNScorer(k=5).fit(target_embeddings)

# 测试时根据域标签选择对应的 KNN
for test_sample in test_samples:
    if test_sample.domain_label == 0:
        score = scorer_source.score(test_sample.embedding)
    else:
        score = scorer_target.score(test_sample.embedding)
```

### 方案 2: 域对抗训练
使用 domain_labels 训练域对抗网络 (DANN)，学习域不变的特征表示。

### 方案 3: 域感知评估
按 source/target 域分别计算 AUC/pAUC，分析模型在不同域上的性能差异。

## 参考文献

- DCASE 2025 Task 2: Domain-Generalized Anomaly Detection
- "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2015)
- CED-Tiny: https://huggingface.co/mispeech/ced-tiny
