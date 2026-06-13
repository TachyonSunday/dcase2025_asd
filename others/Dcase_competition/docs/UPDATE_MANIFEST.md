# 更新文件清单 (2026-06-11)

## 本次更新概述

本次更新将项目从 ResNet18 完全迁移到 CED-Tiny，并实现 DCASE 2025 Task 2 官方评估协议。

## 更新的文件列表

### 📄 核心代码文件 (5 个)

#### 1. **models/feature_extractor.py** ⭐ 完全重构
- **变更**: ResNet18 → CED-Tiny
- **关键改动**:
  - 新增 `CEDFeatureExtractor` 类
  - 使用 Hugging Face `mispeech/ced-tiny` 预训练模型
  - 输入: 16kHz 原始波形 (替代 Log-Mel 频谱图)
  - 输出: 384 维 Embedding (最后两层 Transformer 拼接)
  - 完全冻结模型参数
  - 移除 ResNet18 相关代码

#### 2. **dataset.py** ⭐ 重大更新
- **变更**: 支持 Domain-Generalized
- **关键改动**:
  - 新增 `domain_label` 字段 (0=source, 1=target)
  - 从文件名解析域标签 (`_source_` / `_target_`)
  - 返回原始波形而非 Log-Mel 频谱图
  - 双标签系统: `anomaly_label` + `domain_label`
  - 新增辅助方法: `get_domain_labels()`, `get_source_count()`, `get_target_count()`
  - 移除 `_compute_log_mel()` 方法

#### 3. **utils/scoring.py** ⭐ 完全重构
- **变更**: KNN → Domain-wise Local Density Normalization
- **关键改动**:
  - 新增 `DomainWiseDensityScorer` 类 (GenRep 算法)
  - 分域内存库: `source_memory_bank` + `target_memory_bank`
  - 局部密度预计算: K_s=16 (源域), K_t=9 (目标域)
  - 密度归一化异常分数: `score = min(score_s, score_t)`
  - 新增 `score_with_details()` 返回三域分数
  - 新增 `get_memory_bank_stats()` 获取统计信息
  - 保留 `KNNScorer` 用于对比实验

#### 4. **extract_features.py** 🔄 适配更新
- **变更**: 适配新的特征提取器和标签系统
- **关键改动**:
  - 更新 `extract_features_for_dataset()` 收集 `domain_labels`
  - 更新 `save_features()` 保存 `anomaly_labels` + `domain_labels`
  - 更新 docstring: ResNet → CED-Tiny
  - 返回值从 3 元组改为 4 元组

#### 5. **evaluate.py** ⭐ 完全重构
- **变更**: 实现 DCASE 2025 官方评估协议
- **关键改动**:
  - 新增 `evaluate_machine_type()` 按机器类型评估
  - 新增 `run_evaluation()` 汇总所有机器
  - 新增 `extract_machine_type()` 从文件名提取机器类型
  - 新增 `harmonic_mean()` 计算调和平均数
  - 计算 AUC_source, AUC_target, pAUC, Official Score Ω
  - 输出 CSV + YAML 格式结果
  - 完全对齐 DCASE 2025 官方评估协议

### ⚙️ 配置文件 (2 个)

#### 6. **configs/config.yaml** 🔄 更新
- **变更**: 适配 CED-Tiny 和域感知打分
- **关键改动**:
  ```yaml
  # 模型参数
  model:
    name: "ced-tiny"           # 从 "ResNet18" 改为 "ced-tiny"
    embedding_dim: 384         # 从 512 改为 384
    pool_mode: "concat"        # 新增
    input_type: "waveform"     # 新增
  
  # 打分器参数
  knn:
    k_source: 16               # 新增: 源域近邻数
    k_target: 9                # 新增: 目标域近邻数
    k_score: 5                 # 新增: 推理近邻数
    # 移除旧的 k: 5 参数
  ```

#### 7. **requirements.txt** 🔄 更新
- **变更**: 添加 Hugging Face 依赖
- **关键改动**:
  ```txt
  # 新增
  transformers>=4.30.0         # CED-Tiny 模型加载
  huggingface_hub>=0.20.0      # 模型下载和缓存
  ```

### 🔧 模块初始化文件 (2 个)

#### 8. **models/__init__.py** 🔄 更新
- **变更**: 导出新的特征提取器类
- **关键改动**:
  ```python
  # 旧
  from .feature_extractor import ResNetFeatureExtractor, get_feature_extractor
  
  # 新
  from .feature_extractor import CEDFeatureExtractor, get_feature_extractor
  ```

#### 9. **utils/__init__.py** 🔄 更新
- **变更**: 导出新的打分器类
- **关键改动**:
  ```python
  # 旧
  from .scoring import KNNScorer, compute_auc, compute_pauc
  
  # 新
  from .scoring import DomainWiseDensityScorer, KNNScorer, compute_auc, compute_pauc
  ```

### 📚 文档文件 (6 个)

#### 10. **.claude/CLAUDE.md** ⭐ 完全更新
- **变更**: 项目说明文档
- **关键改动**:
  - 更新核心算法框架描述 (ResNet18 → CED-Tiny)
  - 更新技术栈 (添加 transformers, huggingface_hub)
  - 更新关键设计决策 (特征提取、异常打分、评估指标)
  - 更新当前状态 (标记重构完成的项目)
  - 添加模型对比表格
  - 添加参考文献

#### 11. **PROJECT_STATUS.txt** ⭐ 完全更新
- **变更**: 项目进度报告
- **关键改动**:
  - 更新日期: 2026-06-01 → 2026-06-11
  - 更新状态: 数据准备完成 → 核心代码重构完成
  - 更新项目概述 (新算法框架)
  - 更新已完成工作 (重构详情)
  - 更新下一步计划 (官方评估流程)
  - 添加模型对比表格
  - 更新文件清单
  - 添加参考文献

#### 12. **DCASE2025_EVALUATION.md** ✨ 新增
- **内容**: DCASE 2025 官方评估协议详细说明
- **包含**:
  - 官方评估指标定义 (AUC_source, AUC_target, pAUC, Ω)
  - 实现细节和核心函数说明
  - 输出格式 (控制台、CSV、YAML)
  - 使用方法
  - 评估流程图
  - 与官方对齐检查表
  - 调试建议

#### 13. **DOMAIN_GENERALIZATION.md** ✨ 新增
- **内容**: 域泛化数据集重构说明
- **包含**:
  - 域标签解析逻辑
  - 双标签系统说明
  - 返回格式变更对比
  - 使用示例
  - 域自适应 KNN 方案

#### 14. **REFACTOR_SUMMARY.md** ✨ 新增
- **内容**: CED-Tiny 模型重构总结
- **包含**:
  - 主要变更列表
  - 使用方式示例
  - 性能对比表格
  - 注意事项
  - 下一步建议

#### 15. **CHANGELOG.md** ✨ 新增
- **内容**: 完整更新日志
- **包含**:
  - 所有重大更新详情
  - 技术细节
  - 性能对比
  - 使用方法
  - 已知问题
  - 未来计划
  - 参考文献

## 更新统计

### 文件数量
- **核心代码**: 5 个文件
- **配置文件**: 2 个文件
- **模块初始化**: 2 个文件
- **文档**: 6 个文件
- **总计**: 15 个文件

### 代码变更规模
- **完全重构**: 3 个文件 (feature_extractor.py, scoring.py, evaluate.py)
- **重大更新**: 2 个文件 (dataset.py, CLAUDE.md)
- **适配更新**: 4 个文件 (extract_features.py, config.yaml, requirements.txt, __init__.py × 2)
- **新增文档**: 4 个文件

### 关键变更总结

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| 特征提取器 | ResNet18 (512维) | CED-Tiny (384维) |
| 预训练数据 | ImageNet (图像) | AudioSet (音频) |
| 输入类型 | Log-Mel 频谱图 | 原始波形 |
| 打分算法 | KNN (K=5) | Domain-wise Density (K_s=16, K_t=9) |
| 域支持 | ❌ | ✅ (source/target) |
| 标签系统 | 单一 (anomaly) | 双标签 (anomaly + domain) |
| 评估指标 | AUC + pAUC | AUC_s + AUC_t + pAUC + Ω |
| 官方对齐 | ❌ | ✅ (DCASE 2025) |

## 验证检查

### ✅ 已完成的检查

1. **代码语法检查**
   - [x] evaluate.py 语法检查通过
   - [x] 所有 Python 文件无语法错误

2. **引用一致性检查**
   - [x] 无 ResNet18 残留引用
   - [x] 无 Log-Mel 残留引用 (代码中)
   - [x] 所有 docstring 已更新

3. **导入检查**
   - [x] models/__init__.py 导出正确
   - [x] utils/__init__.py 导出正确

4. **文档检查**
   - [x] CLAUDE.md 完全更新
   - [x] PROJECT_STATUS.txt 完全更新
   - [x] 所有新增文档完整

### 🔍 建议的后续检查

1. **功能测试**
   - [ ] 运行 `python extract_features.py --mode both`
   - [ ] 运行 `python evaluate.py`
   - [ ] 验证输出格式正确

2. **性能测试**
   - [ ] 对比 ResNet18 和 CED-Tiny 的特征提取速度
   - [ ] 对比 KNN 和 Domain-wise Density 的打分性能
   - [ ] 验证 Official Score 计算正确性

3. **兼容性测试**
   - [ ] 验证 Hugging Face 模型下载
   - [ ] 验证 Windows 环境兼容性
   - [ ] 验证 CUDA 加速正常

## 回滚方案

如果需要回滚到 ResNet18 版本，可以通过 Git 恢复到本次更新之前的状态：

```bash
# 查看当前 commit
git log --oneline

# 回滚到指定 commit
git reset --hard <commit_hash>
```

或者保留当前版本，创建新的分支进行实验：

```bash
# 创建新分支
git checkout -b resnet18-fallback

# 在新分支上回滚
git reset --hard <old_commit_hash>
```

## 下一步

1. **立即执行**
   ```bash
   # 安装新依赖
   pip install -r requirements.txt
   
   # 提取特征
   python extract_features.py --config configs/config.yaml --mode both
   
   # 运行官方评估
   python evaluate.py --config configs/config.yaml
   ```

2. **查看结果**
   - 控制台: 7 个机器的详细指标 + 平均值
   - CSV: `results/dcase2025_results_YYYYMMDD_HHMMSS.csv`
   - YAML: `results/dcase2025_summary_YYYYMMDD_HHMMSS.yaml`

3. **性能调优** (可选)
   - 调整 K_s, K_t, K_score 参数
   - 尝试不同距离度量
   - 实现 DANN 域对抗训练

---

**更新时间**: 2026-06-11  
**更新人员**: AI Assistant  
**审核状态**: 待用户验证
