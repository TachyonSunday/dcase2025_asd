# CED-Tiny 模型重构总结

## 概述
将特征提取器从 ResNet18 重构为 CED-Tiny (Compact Encoder-Decoder) 音频预训练模型，这是 DCASE 2025 顶尖方案中广泛使用的模型。

## 主要变更

### 1. models/feature_extractor.py
**从**: ResNet18 图像分类模型  
**到**: CED-Tiny 音频预训练模型

**关键特性**:
- ✅ 输入: 16kHz 单声道原始波形 (1D Tensor)
- ✅ 输出: 384维 Embedding (最后两层 Transformer 拼接)
- ✅ 架构: 12层 Transformer Encoder
- ✅ 所有参数彻底冻结，不参与梯度更新
- ✅ 支持两种融合模式: "concat" (384维) 或 "mean" (192维)

**技术细节**:
- 模型来源: `mispeech/ced-tiny` (Hugging Face)
- 特征提取: 内部使用 feature_extractor 将波形转为 Mel 频谱图
- 层选择: 提取最后两层 (layer 10 & 11) 的 Transformer Embedding
- 池化: 对时间维度取均值 (Mean Pooling)
- 融合: 默认拼接两层得到 384 维，或平均得到 192 维

### 2. dataset.py
**从**: 返回 Log-Mel 频谱图  
**到**: 返回原始波形

**变更内容**:
- 移除 Log-Mel 频谱图计算逻辑
- 返回字段从 `"mel"` 改为 `"waveform"`
- 波形 shape: `(num_samples,)` 而非 `(1, n_mels, T)`
- 保持原有的音频加载、重采样、截断/填充逻辑

### 3. configs/config.yaml
**模型配置更新**:
```yaml
model:
  name: "ced-tiny"           # 从 "ResNet18" 改为 "ced-tiny"
  pretrained: true           # 使用预训练权重
  embedding_dim: 384         # 从 512 改为 384
  pool_mode: "concat"        # 新增: 最后两层融合方式
  input_type: "waveform"     # 新增: 输入类型标识
```

### 4. requirements.txt
**新增依赖**:
```txt
transformers>=4.30.0         # Hugging Face 模型加载
huggingface_hub>=0.20.0      # 模型下载和缓存管理
```

### 5. models/__init__.py
**导出类名更新**:
- 从 `ResNetFeatureExtractor` 改为 `CEDFeatureExtractor`

## 使用方式

### 加载模型
```python
from models.feature_extractor import get_feature_extractor
import yaml

# 加载配置
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建特征提取器
model = get_feature_extractor(config, pool_mode='concat')
```

### 提取特征
```python
import torch

# 准备输入: 16kHz 单声道波形
waveform = torch.randn(2, 160000)  # batch_size=2, 10秒

# 提取特征
with torch.no_grad():
    embeddings = model(waveform)  # shape: (2, 384)
```

### 数据集加载
```python
from dataset import AudioAnomalyDataset

dataset = AudioAnomalyDataset(
    audio_dir='dev_data/raw/fan/train',
    config=config,
    mode='train'
)

sample = dataset[0]
waveform = sample['waveform']  # shape: (160000,)
label = sample['label']        # 0 或 1
```

## 性能对比

| 特性 | ResNet18 | CED-Tiny |
|------|----------|----------|
| 输入类型 | Log-Mel 频谱图 | 原始波形 |
| 输出维度 | 512 | 384 (concat) / 192 (mean) |
| 预训练数据 | ImageNet (图像) | AudioSet (音频) |
| 模型大小 | ~11M | ~5M |
| 适用场景 | 通用图像特征 | 音频专用特征 |
| 音频理解 | 需手动转换 | 原生支持 |

## 注意事项

1. **采样率**: 必须使用 16kHz 采样率 (CED-Tiny 要求)
2. **输入格式**: 原始波形，无需预计算 Mel 频谱图
3. **参数冻结**: 所有参数已冻结，仅作为特征提取器使用
4. **设备支持**: 支持 CPU 和 CUDA，自动检测
5. **首次运行**: 会自动下载 CED-Tiny 模型 (~100MB)

## 下一步

1. 运行特征提取:
   ```bash
   python extract_features.py --config configs/config.yaml --mode both
   ```

2. 运行 KNN 评估:
   ```bash
   python evaluate.py --config configs/config.yaml
   ```

3. 调优建议:
   - 尝试不同的 `pool_mode` (concat vs mean)
   - 调整 KNN 的 K 值和距离度量
   - 考虑使用 CED-Tiny 的多尺度特征

## 参考文献

- CED 论文: "CED: A Compact and Efficient Audio Classifier"
- 模型仓库: https://huggingface.co/mispeech/ced-tiny
- DCASE 2025 Task 2: https://dcase.community/challenge2025/task-unsupervised-detection-anomalous-sounds
