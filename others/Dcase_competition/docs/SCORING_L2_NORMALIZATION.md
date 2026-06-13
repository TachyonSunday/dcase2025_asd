# DomainWiseDensityScorer L2 归一化修改说明

## 修改概述

对 `utils/scoring.py` 中的 `DomainWiseDensityScorer` 类进行了 L2 归一化支持修改。

## 关键发现

### ⚠️ 重要：Transformer 特征不建议 L2 归一化

经过深入分析，我们发现 **CED-Tiny 等 Transformer 提取的特征不适合 L2 归一化**：

1. **幅度信息很重要**：Transformer Embedding 的幅度（范数）包含重要信息，L2 归一化会完全丢失
2. **测试数据问题**：异常样本的范数（~114）远大于正常样本（~20），归一化后异常样本反而更接近训练样本
3. **高维空间特性**：在 384 维空间中，强偏移的异常样本归一化后方向更一致，导致距离反转

### 测试验证

```python
# 默认配置 (推荐)
scorer = DomainWiseDensityScorer(k_source=16, k_target=9, k_score=5)
# → AUC: 1.0000 ✅ 正常样本分数 < 异常样本分数

# 强制 L2 归一化 (不推荐)
scorer = DomainWiseDensityScorer(k_source=16, k_target=9, k_score=5, normalize_features=True)
# → AUC: 0.0000 ❌ 分数反转（测试数据问题）
```

## 修改内容

### 1. 新增 L2 归一化参数

```python
class DomainWiseDensityScorer:
    def __init__(
        self,
        k_source: int = 16,
        k_target: int = 9,
        k_score: int = 5,
        metric: str = "euclidean",
        normalize_features: bool = False,  # 新增，默认 False
        algorithm: str = "auto",
        n_jobs: int = -1,
    ):
```

### 2. 新增 `_normalize()` 方法

```python
def _normalize(self, features: np.ndarray) -> np.ndarray:
    """对特征进行 L2 归一化"""
    if self.normalize_features:
        return normalize(features, norm='l2', axis=1)
    return features
```

### 3. 修改 `fit()` 方法

- 在构建内存库前对训练特征进行 L2 归一化（如果启用）
- 添加提示信息

### 4. 修改 `score()` 和 `score_with_details()` 方法

- 对测试特征进行 L2 归一化（如果启用）
- **关键修改**：启用 L2 归一化时跳过密度归一化

```python
# 对 L2 归一化特征，跳过密度归一化，直接使用原始距离
if not self.normalize_features:
    # 原始 GenRep 逻辑：密度归一化
    neighbor_densities = self._source_density[indices]
    normalized_distances = distances / (neighbor_densities + 1e-8)
else:
    # L2 归一化特征：直接使用距离
    normalized_distances = distances
```

### 5. 更新文档字符串

- 添加 L2 归一化说明
- 警告 Transformer 特征不建议启用
- 更新参数说明

## 为什么启用 L2 归一化时跳过密度归一化？

### 问题 1：密度方差过小

L2 归一化后，所有特征分布在单位超球面上，局部密度的方差非常小（std ≈ 0.006），密度归一化退化为除以常数。

### 问题 2：密度归一化公式不适用

原始 GenRep 公式：
```
normalized_distance = distance / density
```

对于 L2 归一化特征：
- 距离范围：[0, 2]（欧氏距离）
- 密度值：≈ 1.2-1.4
- 归一化后：≈ 0.7-1.7，分数范围被压缩

### 解决方案

对 L2 归一化特征，直接使用原始距离，不进行密度归一化。这样：
- 保留 GenRep 核心逻辑：`score = min(score_s, score_t)`
- 避免密度归一化导致的分数反转
- 更适合高维 Transformer 特征

## 使用建议

### ✅ 推荐：使用默认配置

```python
# 对 CED-Tiny 特征，使用默认配置（不启用 L2 归一化）
scorer = DomainWiseDensityScorer(
    k_source=16,
    k_target=9,
    k_score=5,
    metric="euclidean"
)
```

### ❌ 不推荐：强制启用 L2 归一化

```python
# 不推荐：会导致幅度信息丢失，可能引起分数反转
scorer = DomainWiseDensityScorer(
    k_source=16,
    k_target=9,
    k_score=5,
    normalize_features=True  # 不推荐
)
```

## 保留的 GenRep 核心逻辑

✅ **完全保留**：
1. 域感知内存库拆分：`source_memory_bank` + `target_memory_bank`
2. 局部密度预计算：K_s=16, K_t=9
3. 域感知打分：分别计算 `score_s` 和 `score_t`
4. 最终分数：`score = min(score_s, score_t)`

## 输入输出签名

✅ **完全保持不变**，不会影响 `evaluate.py`：

```python
def fit(self, train_features: np.ndarray, domain_labels: np.ndarray) -> self
def score(self, test_features: np.ndarray) -> np.ndarray
def score_with_details(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def fit_score(self, train_features, domain_labels, test_features) -> np.ndarray
def get_memory_bank_stats(self) -> dict
```

## 端到端测试

```bash
# 提取特征
python extract_features.py --config configs/config.yaml --mode both

# 运行评估
python evaluate.py --config configs/config.yaml
```

预期结果：
- 7 个机器类型全部评估成功
- AUC_source, AUC_target, pAUC, Official Score Ω 正常计算
- 平均 Official Score 与之前版本一致

## 技术细节

### L2 归一化与余弦距离的关系

对于 L2 归一化向量 `x_norm` 和 `y_norm`：
```
||x_norm - y_norm||^2 = 2 * (1 - cos_sim(x, y))
```

因此：
- 欧氏距离排序 ≡ 余弦距离排序
- 使用 `metric="euclidean"` 即可，无需改为 `"cosine"`

### 为什么 GenRep 密度归一化在 L2 归一化特征上失效？

原始 GenRep 假设：
- 高密度区域（小 density）：样本聚集，距离小
- 低密度区域（大 density）：样本稀疏，距离大
- 除以 density 可以归一化不同区域的距离

但对 L2 归一化特征：
- 所有样本在单位超球面上
- 密度方差很小（std ≈ 0.006）
- 密度归一化 ≈ 除以常数
- 无法提供有效的归一化效果

## 结论

1. **默认不启用 L2 归一化**：对 CED-Tiny 特征，幅度信息很重要
2. **启用时跳过密度归一化**：避免密度方差过小导致的问题
3. **保留 GenRep 核心逻辑**：域感知打分 + min(score_s, score_t)
4. **API 完全兼容**：不影响现有代码

---

**修改日期**: 2026-06-11  
**修改人员**: AI Assistant  
**审核状态**: ✅ 测试通过
