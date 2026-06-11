# GenRep用于DCASE 2025挑战赛的首试无监督异常声音检测

## 技术报告

**Phurich Saengthong, Takahiro Shinozaki**

**东京科学大学（Institute of Science Tokyo）**

www.ts.ip.titech.ac.jp

---

## 摘要

近年来，大规模预训练音频模型的最新进展表明，冻结嵌入（frozen embeddings）能够为通用音频任务提供鲁棒且可迁移的表示。GenRep 使用冻结嵌入结合 k 近邻（k-nearest neighbors）和域级 Z 分数归一化（domain-wise Z-score normalization）来实现域偏移下的异常检测。我们在 GenRep 的基础上，探索了多个改进方向，包括归一化策略、模型缩放和特征集成。首先，我们研究了替代归一化方法，如全局 Z 分数归一化、局部密度归一化（local density normalization）和域级局部密度归一化。其次，我们在 DCASE2025 Task 2 数据集上评估了参数量从 5M 到 300M 的预训练音频编码器，以考察模型规模的影响。第三，我们研究了使用来自多个冻结编码器的特征进行集成融合的效果。我们的结果表明，即使是最小的预训练编码器（5.49M）也能超越基线自编码器（autoencoder）的性能，而更大的模型和集成方法则能进一步带来提升，且无需更新模型参数。代码已开源[1]。

**索引关键词**——异常检测、声学状态监测、域偏移、首试问题、DCASE 挑战赛

---

## 1. 引言

DCASE2025 Task 2 挑战赛继续聚焦于域泛化设置下异常声音检测（Anomalous Sound Detection, ASD）的首试问题（first-shot problem），参赛者必须开发能够在完全未见过的机器类型上泛化的系统，而不能对目标域数据进行调优。为了反映这一实际约束，评估数据集包含了开发集中未出现的机器类型。此外，挑战赛引入了两种可选资源以支持性能提升：（1）补充数据，例如干净的机器声音或仅噪声录音；（2）来自往届 DCASE Task 2 挑战赛的外部数据集，用于模拟在真实世界场景中使用历史数据进行模型预训练的情况 [1, 2, 3, 4]。

在典型的异常检测设置中，模型在来自已有域的正常数据上进行训练，该域中正常状态的定义已明确建立。然而，在部署过程中，环境可能会发生变化。例如，背景噪声、运行条件或传感器配置的变化会导致域偏移（domain shift）。仅在原始域上训练的模型在此类偏移下可能无法泛化，从而导致误报或漏检异常。为了应对域偏移，还需要定义一个能代表新条件的目标域数据集 [1, 5]。

> [1] https://github.com/Phuriches/GenRepASD

近期针对域泛化 ASD 任务的最先进方法通过使用离群值暴露（Outlier Exposure）框架 [6] 微调或更新模型参数来解决这一问题。这通常涉及从头训练音频编码器或微调大规模预训练音频编码器 [7, 8, 9, 10, 11, 12]，如图 1a 所示。

异常检测系统成功的一个关键因素是其依赖的音频嵌入（audio embeddings）的质量。能够在不同域之间鲁棒地表示正常声音特征的嵌入，对于在变化条件下区分异常至关重要。因此，使用一个能够提取有意义嵌入的通用音频编码器，而无需微调，同时还能明确定义目标域，将是非常有益的。这将消除在部署期间重新训练或更新编码器的需求，从而避免引入停机时间和操作复杂性。

如图 1b 所示，这种系统可以通过将训练数据存储在特定域的记忆库（memory banks）中来实现，这些记忆库定义了源域和目标域的正常行为。在推理过程中，测试数据会与两个记忆库进行比较，并选择最小距离分数作为最终的异常分数。GenRep [13] 证明了从大规模预训练音频编码器中提取的嵌入可以有效地用于域泛化 ASD，而无需微调。然而，GenRep 依赖于测试时统计量来标准化或归一化异常分数，这些统计量在推理期间可能并不总是可用。为了解决这一问题，我们探索了不依赖测试时统计量的归一化策略，旨在改善域偏移下的泛化能力。具体来说，我们研究了域级 Z 分数归一化、局部密度归一化 [12] 和域级局部密度归一化，这些方法应用于通过最近邻搜索计算的异常分数，使用的是来自各种大规模预训练音频编码器的冻结表示。我们的贡献包括：

- 我们研究了归一化策略——域级 Z 分数、局部密度和域级局部密度归一化——及其对域对齐和异常检测性能的影响。
- 我们在 DCASE2025 Task 2 数据集上使用一系列预训练音频编码器（5M–300M 参数）评估了 GenRep，观察到模型越大性能持续提升。
- 我们所有的系统都超越了基线系统，最佳系统达到了 64.53 的官方分数。使用最小编码器（ced tiny）配合域级局部密度归一化，仍以 62.15 的分数保持了竞争力。

---

**图 1：异常检测流程对比。**（a）最先进的系统通常使用来自微调过的音频编码器的嵌入。（b）使用通用音频编码器而不进行微调，从而能更轻松地适应目标域。

```
a) 使用微调音频编码器的嵌入

训练数据/微调
音频编码器
     ↙               ↘
源域音频    →    源记忆库
目标域音频  →    目标记忆库
     ↘               ↙
微调后的音频编码器 → 最近邻搜索 → 异常分数
                    ↑
               测试输入音频

b) 使用通用音频编码器而无需微调，便于轻松适应目标域

训练数据          参考音频
     ↓               ↓
通用音频编码器   通用音频编码器
     ↙               ↘
源域音频    →    源记忆库
目标域音频  →    目标记忆库
     ↘               ↙
            最近邻搜索 → 异常分数
                ↑
            输入音频
```

---

## 2. 方法

GenRep [13] 证明了使用从大规模预训练音频编码器中提取的通用冻结特征可以显著改善域泛化异常声音检测（ASD）。值得注意的是，这种方法优于依赖目标域数据微调的方法 [14]。促成这一成功的一个关键因素是归一化技术的使用。具体来说，GenRep 基于测试数据的分布应用 Z 分数标准化。

然而，这种对测试分布的依赖在实际部署中构成了挑战，因为实际中不可能提前获得测试数据。此外，DCASE2025 Task 2 挑战赛 [1] 明确禁止使用测试数据进行任何形式的训练，这进一步突显了此类归一化策略的局限性。

为解决这一问题，我们在 GenRep 的基础上，探索了不依赖测试分布的替代归一化方法。具体来说，我们专注于仅基于训练数据进行操作的归一化方法，使其更适合域泛化 ASD。

### 2.1. 使用通用表示的域泛化 kNN

我们使用 GenRep [13] 但不采用 MemMixup。我们使用大规模预训练音频编码器，将来自训练源域和目标域的特征嵌入存储在记忆库中。在测试时，样本 y 的异常分数通过将其特征 f_y 与两个记忆库进行比较来计算。

对于每个域，我们计算到 K_n 个最近邻的平均距离：

$$d(y) = \frac{1}{K_n} \sum_{f \in \mathcal{N}_{K_n}(f_y)} \| f - f_y \|_2, \tag{1}$$

其中 $\mathcal{N}_{K_n}(f_y)$ 表示对应记忆库中的最近邻，产生分数 $d_s(y)$ 和 $d_t(y)$。

由于在推理时 y 的域是未知的，我们假设它属于其表现得最正常的域。为了比较分数，我们使用由测试异常分数分布计算得到的测试时均值 $\mu_s^{\text{test}}$、$\mu_t^{\text{test}}$ 和标准差 $\sigma_s^{\text{test}}$、$\sigma_t^{\text{test}}$ 进行 Z 分数归一化。归一化分数定义为 $Z\text{-score}(d_s) = \frac{d_s(y) - \mu_s^{\text{test}}}{\sigma_s^{\text{test}}}$ 和 $Z\text{-score}(d_t) = \frac{d_t(y) - \mu_t^{\text{test}}}{\sigma_t^{\text{test}}}$。

最终异常分数为：

$$\text{score}(y) = \min \left( \frac{d_s(y) - \mu_s^{\text{test}}}{\sigma_s^{\text{test}}}, \frac{d_t(y) - \mu_t^{\text{test}}}{\sigma_t^{\text{test}}} \right). \tag{2}$$

### 2.2. 域级 Z 分数归一化

为了消除前述公式中引入的测试时分布需求，我们转而从训练数据中估计归一化统计量。对于来自域 $d \in \{s, t\}$ 的每个训练样本 $f_i$，我们计算其域内 kNN 距离为 $d(f_i) = \frac{1}{K_n} \sum_{f_j \in \mathcal{N}_{K_n}(f_i)} \| f_i - f_j \|_2$，其中 $f_j$ 是来自同一域的 $K_n$ 个最近邻。这产生了域特定的训练统计量：$\mu_s^{\text{train}}$、$\sigma_s^{\text{train}}$ 和 $\mu_t^{\text{train}}$、$\sigma_t^{\text{train}}$。

在测试时，我们像之前一样计算 $d_s(y)$ 和 $d_t(y)$，并使用基于训练的均值对其进行归一化。虽然我们保留了域特定的均值，但我们通过实验发现，使用目标域的标准差 $\sigma_t^{\text{train}}$ 进行两种归一化，可以通过将分数对齐到共同的尺度来提升性能。最终异常分数变为：

$$\text{score}(y) = \min \left( \frac{d_s(y) - \mu_s^{\text{train}}}{\sigma_t^{\text{train}}}, \frac{d_t(y) - \mu_t^{\text{train}}}{\sigma_t^{\text{train}}} \right). \tag{3}$$

### 2.3. 局部密度归一化

我们进一步研究了将局部密度归一化方法 [12] 应用于 GenRep 框架，该方法根据每个参考样本周围的密度来调整异常分数。对于测试样本特征 $f_y$ 和一组参考特征 $F_{\text{ref}}$（例如来自记忆库），局部归一化的异常分数定义为：

$$\text{score}(y) = \min_{f \in F_{\text{ref}}} \frac{d(f_y, f)}{\sum_{k=1}^{K} d(f, f_k)}, \tag{4}$$

其中 $d(f_y, f)$ 是测试样本与参考特征之间的距离，$f_k$ 表示 $f$ 在 $F_{\text{ref}}$ 中的 $K$ 个最近邻。该公式通过参考特征周围的局部密度来重新缩放异常分数。

### 2.4. 域级局部密度归一化

我们还探索了局部密度归一化 [12] 的一种可能扩展，即以域级方式应用它。具体来说，对于每个测试样本特征 $f_y$，我们分别针对源记忆库 $F_s$ 和目标记忆库 $F_t$ 计算其局部归一化异常分数。每个分数都根据相应域内参考特征周围的局部密度进行调整。为了适应不平衡的参考集大小，例如当目标域包含显著更少的样本时，我们允许不同域的近邻数量 $K$ 不同，分别记为源域的 $K_s$ 和目标域的 $K_t$。最终异常分数取两个归一化分数的最小值：

$$\text{score}(y) = \min \left( \min_{f \in F_s} \frac{d(f_y, f)}{\sum_{k=1}^{K_s} d(f, f_k)}, \min_{f \in F_t} \frac{d(f_y, f)}{\sum_{k=1}^{K_t} d(f, f_k)} \right), \tag{5}$$

其中 $f_k$ 表示 $f$ 在其相应的域特定记忆库中的 $K_s$ 或 $K_t$ 个最近邻。

### 2.5. 异常检测细节

我们在 GenRep 框架内研究了五种最先进的大规模预训练音频编码器，分别记为：BEATs ft1 [2] 用于 BEATs [15]，m2d clap [3] 用于 M2D CLAP [16]，EAT large [4] 用于 EAT [17]，SSLAM [5] 用于 SSLAM [18]，以及 ced base 和 ced tiny [6] 用于 CED [19]。对于每个编码器，我们从训练数据中提取特征并将其存储在相应的源记忆库和目标记忆库中。在此过程中未使用任何补充数据。

> [2] https://github.com/microsoft/unilm/tree/master/beats
> [3] https://github.com/nttcslab/m2d
> [4] https://github.com/cwx-worst-one/EAT
> [5] https://github.com/ta012/SSLAM/
> [6] https://github.com/RicherMans/CED

**表 1：系统的 GenRep 配置，包括归一化方法、使用的特征层和模型复杂度。**

| 系统 | 归一化 | 特征层 | MACs / 参数量 |
|------|--------|--------|---------------|
| System 1 | 域级 LD | 最后两层 | 271.71 G / 569.28 M |
| System 2 | LD | 最后两层 | 271.71 G / 569.28 M |
| System 3 | 域级 Z-score | 第 7 层和第 10 层* | 271.71 G / 569.28 M |
| System 4 | 域级 LD | 第 7 层和第 10 层 | 1.34 G / 5.49 M |

> *LD = 局部密度（Local density）。*对于 EAT large，System 3 使用最后两层。

我们应用了三种不同的分数归一化方法，形成了三个集成系统：System 1 使用域级局部密度归一化，System 2 使用不带域分离的局部密度归一化，System 3 采用域级 Z 分数归一化。每个集成系统共享相同的模型复杂度，即 271.71 G MACs 和 569.28 M 参数。此外，我们还提交了一个轻量级变体 System 4，它仅使用 ced tiny 和域级局部密度归一化，形成紧凑的配置，仅需 1.34 G MACs 和 5.49 M 参数。

对于归一化参数，我们为域级 Z 分数归一化设置 $K = 1$，源域和目标域均使用 $K = 1$。对于局部密度归一化，我们设置 $K = 16$；对于域级局部密度归一化，设置 $K_s = 16$，$K_t = 9$。

**数据集概要。** 该数据集包含三个子集：开发集、附加训练集和评估数据集。开发集包含七种机器类型，每种机器类型有一个 section，包含来自源域的 990 个正常片段、来自目标域的 10 个正常片段，以及 200 个带标签的测试片段（100 个正常和 100 个异常），并带有域标签。一些机器还包含属性注释（attribute annotations）。附加训练数据集引入了九种新的机器类型，每种机器类型具有相同的训练结构，但只有部分机器提供了属性。评估数据集包含与附加训练机器对应的测试片段，没有任何标签或域信息。参赛者必须仅使用每种机器类型的一个 section 来训练模型，不得在测试集上调优，也不得依赖属性信息 [1]。

**评估指标。** 域偏移下的性能使用 AUC、部分 AUC（pAUC）和官方分数进行评估。官方分数定义为源 AUC、目标 AUC 和混合 pAUC 在所有机器类型上的调和平均值 [1]。

---

## 3. 实验结果

**表 2：基线与提交系统在开发数据上的比较。每列最佳结果以粗体标出。**

| 系统 | AUC 源 | AUC 目标 | pAUC | 官方分数 |
|------|--------|----------|------|----------|
| Baseline [4] | 66.78 | 51.39 | 52.94 | 56.26 |
| System1 | **76.11** | 61.66 | 58.36 | **64.53** |
| System2 | 63.42 | **68.73** | **59.69** | 63.74 |
| System3 | 67.96 | 61.46 | 56.23 | 61.51 |
| System4 | 72.56 | 60.01 | 56.10 | 62.15 |

如表 2 所示，System1（应用域级局部密度归一化）以 64.53 的官方分数和最佳 AUC 源（76.11）取得了最高的整体性能。System2（使用不带域分离的局部密度归一化）以 63.74 的官方分数排名第二，同时在 AUC 目标（68.73）和 pAUC（59.69）上取得了最佳结果。System4（仅使用 ced tiny 配合域级局部密度归一化的轻量级变体）以 62.15 的强劲官方分数，略优于采用域级 Z 分数归一化并得分 61.51 的 System3。基线系统 [4] 表现最低，官方分数为 56.26，远低于所有提出的系统。

**图 2：归一化评分方法的比较。** 对于每个音频编码器，我们报告的是在所有机器类型上取得最佳综合性能的层的结果（而不是对每台单独机器挑选最佳表现层，那样会导致虚高的性能）。

| 音频编码器 | 域级 Z-score（蓝） | 域级局部密度归一化（橙） | 局部密度归一化（绿） |
|------------|-------------------|--------------------------|----------------------|
| BEATs_ft1 | ~60.5 | ~62.5 | ~61.5 |
| EAT | ~58.5 | ~61.0 | ~57.0 |
| EAT_large | ~60.0 | ~60.5 | ~60.0 |
| SSLAM | ~61.5 | ~61.5 | ~61.5 |
| ced_base | ~62.0 | ~62.0 | ~62.0 |
| ced_tiny | ~60.0 | ~61.5 | ~57.5 |
| m2d_clap | ~59.5 | ~60.5 | ~60.5 |

> 图 2 展示了三种归一化方法——域级 Z-score（蓝色）、域级局部密度归一化（橙色）和局部密度归一化（绿色）——在七种音频编码器上的官方分数。域级局部密度归一化（橙色）对大多数编码器普遍取得了最高或可比的分数。域级 Z-score 归一化（蓝色）保持了相对一致的表现，而不带域分离的局部密度归一化（绿色）在某些情况下显示出更大的波动性和更低的分数，例如 EAT 和 ced tiny。归一化方法之间的性能差距因编码器而异：某些编码器（如 EAT large、SSLAM 和 ced base）表现出微小差异，而其他编码器（如 EAT 和 ced tiny）则显示出更显著的变化。

---

## 4. 讨论与结论

我们认为，我们的方法为域泛化异常声音检测的未来工作提供了一个实用且有效的基础方案。尽管方法简单——仅使用冻结音频编码器和轻量级归一化技术——我们的系统在多种机器类型上取得了强劲的性能，而无需依赖测试时适应。值得注意的是，即使是紧凑模型如 CED tiny，在与域级局部密度归一化结合使用时，也超越了传统的基于自编码器的基线系统 [4]，突显了现成表示（off-the-shelf representations）在挑战性域偏移场景中的潜力。

**表 3：各系统和机器在开发集上的异常检测性能。**

| 机器 | 指标 | System 1 | System 2 | System 3 | System 4 |
|------|------|----------|----------|----------|----------|
| **ToyCar** | | | | | |
| | AUC source | 70.90 | 63.12 | 64.60 | 70.54 |
| | AUC target | 67.72 | 71.56 | 71.10 | 63.44 |
| | pAUC | 54.37 | 55.68 | 52.21 | 50.53 |
| | 官方分数 | 63.47 | 62.79 | 61.60 | 60.32 |
| **ToyTrain** | | | | | |
| | AUC source | 88.34 | 84.34 | 79.14 | 87.12 |
| | AUC target | 68.06 | 69.88 | 69.62 | 67.52 |
| | pAUC | 60.53 | 59.79 | 54.53 | 55.21 |
| | 官方分数 | 70.53 | 69.94 | 66.17 | 67.57 |
| **bearing** | | | | | |
| | AUC source | 71.12 | 70.12 | 65.12 | 64.62 |
| | AUC target | 60.28 | 61.86 | 57.94 | 53.98 |
| | pAUC | 60.21 | 60.32 | 60.84 | 56.74 |
| | 官方分数 | 63.48 | 63.82 | 61.16 | 58.11 |
| **fan** | | | | | |
| | AUC source | 72.44 | 34.26 | 58.88 | 67.88 |
| | AUC target | 47.28 | 74.72 | 47.26 | 48.98 |
| | pAUC | 51.63 | 56.95 | 50.95 | 49.00 |
| | 官方分数 | 55.23 | 49.89 | 51.93 | 53.00 |
| **gearbox** | | | | | |
| | AUC source | 70.30 | 66.92 | 63.66 | 67.04 |
| | AUC target | 59.38 | 69.70 | 56.56 | 58.42 |
| | pAUC | 56.21 | 58.74 | 52.84 | 57.63 |
| | 官方分数 | 61.41 | 64.77 | 57.35 | 60.75 |
| **slider** | | | | | |
| | AUC source | 82.00 | 82.54 | 77.78 | 78.78 |
| | AUC target | 57.60 | 57.92 | 57.86 | 59.30 |
| | pAUC | 56.58 | 57.05 | 55.47 | 59.89 |
| | 官方分数 | 63.52 | 63.95 | 62.28 | 64.86 |
| **valve** | | | | | |
| | AUC source | 81.52 | 82.24 | 71.50 | 76.96 |
| | AUC target | 82.46 | 80.68 | 82.72 | 76.00 |
| | pAUC | 73.58 | 72.00 | 71.47 | 67.58 |
| | 官方分数 | 78.98 | 78.04 | 74.88 | 73.26 |
| **All (Avg)** | | | | | |
| | AUC source | 76.11 | 63.42 | 67.96 | 72.56 |
| | AUC target | 61.66 | 68.73 | 61.46 | 60.01 |
| | pAUC | 58.36 | 59.69 | 56.23 | 56.10 |
| | 官方分数 | 64.53 | 63.74 | 61.51 | 62.15 |

> 官方分数是 AUC source、AUC target 和 pAUC 的调和平均值。

**表 4：模型复杂度比较。**

| 模型 | MACs (G) | 参数量 (M) |
|------|----------|-----------|
| Baseline | 0.17 | 0.27 |
| BEATs ft1 | 45.01 | 90.71 |
| m2d clap | 26.50 | 85.25 |
| EAT | 43.71 | 85.25 |
| SSLAM | 43.71 | 85.25 |
| ced tiny | 1.33 | 5.49 |
| ced base | 21.13 | 85.66 |
| EAT large | 155.16 | 302.57 |

---

## 5. 参考文献

[1] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2025 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring," In arXiv e-prints: 2506.10097, 2025.

[2] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, "ToyADMOS2: Another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions," in DCASE, Barcelona, Spain, November 2021.

[3] K. Dohi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, Y. Nikaido, and Y. Kawaguchi, "MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), Nancy, France, November 2022.

[4] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, and M. Yasuda, "First-shot anomaly detection for machine condition monitoring: A domain generalization baseline," Proceedings of 31st European Signal Processing Conference (EUSIPCO), pp. 191–195, 2023.

[5] K. Wilkinghoff, T. Fujimura, K. Imoto, J. L. Roux, Z.-H. Tan, and T. Toda, "Handling domain shifts for anomalous sound detection: A review of dcase-related work," 2025. [Online]. Available: https://arxiv.org/abs/2503.10435

[6] D. Hendrycks, M. Mazeika, and T. Dietterich, "Deep anomaly detection with outlier exposure," ICLR, 2019.

[7] K. Wilkinghoff, "Self-supervised learning for anomalous sound detection," in ICASSP. IEEE, 2024, pp. 276–280.

[8] Z. Lv, A. Jiang, B. Han, Y. Liang, Y. Qian, X. Chen, J. Liu, and P. Fan, "Aithu system for first-shot unsupervised anomalous sound detection," DCASE2024 Challenge, Tech. Rep., June 2024.

[9] A. Jiang, B. Han, Z. Lv, Y. Deng, W.-Q. Zhang, X. Chen, Y. Qian, J. Liu, and P. Fan, "Anopatch: Towards better consistency in machine anomalous sound detection," in Proc. Interspeech, 2024.

[10] A. Jiang, X. Zheng, B. Han, Y. Qiu, P. Fan, W.-Q. Zhang, C. Lu, and J. Liu, "Adaptive prototype learning for anomalous sound detection with partially known attributes," in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[11] T. Fujimura, I. Kuroyanagi, and T. Toda, "Improvements of discriminative feature space training for anomalous sound detection in unlabeled conditions," in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[12] K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, and J. L. Roux, "Keeping the balance: Anomaly score calculation for domain generalization," in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[13] P. Saengthong and T. Shinozaki, "Deep generic representations for domain-generalized anomalous sound detection," in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[14] K. Dohi, K. Imoto, N. Harada, D. Niizumi, Y. Koizumi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring," May 2023, arXiv:2305.07828 [cs, eess]. [Online]. Available: http://arxiv.org/abs/2305.07828

[15] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, "BEATs: Audio pre-training with acoustic tokenizers," in ICML, vol. 202. PMLR, July 2023. [Online]. Available: https://proceedings.mlr.press/v202/chen23ag.html

[16] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, M. Yasuda, S. Tsubaki, and K. Imoto, "M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation," to appear at Interspeech, 2024. [Online]. Available: https://arxiv.org/abs/2406.02032

[17] W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, "Eat: Self-supervised pre-training with efficient audio transformer," in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, K. Larson, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2024, pp. 3807–3815, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2024/421

[18] T. Alex, S. Atito, A. Mustafa, M. Awais, and P. J. B. Jackson, "SSLAM: Enhancing self-supervised models with audio mixtures for polyphonic soundscapes," in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=odU59TxdiB

[19] H. Dinkel, Y. Wang, Z. Yan, J. Zhang, and Y. Wang, "Ced: Consistent ensemble distillation for audio tagging," in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.
