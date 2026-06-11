# 基于预训练模型增强的异常声音检测系统用于DCASE2025 Task2

## 技术报告

**作者：Lei Wang**

**单位：MYPS，中国阜阳**

**联系方式：1022160842@qq.com**

---

## 摘要

本研究提出了一种鲁棒的方法来应对DCASE2025 Challenge Task2，该任务聚焦于面向机器状态监测的首次无监督异常声音检测（First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring）。该任务提出了一个独特的挑战：在有属性信息和没有属性信息两种场景下均需训练模型，要求模型在两种场景下都能表现出鲁棒的性能。

为了解决这一挑战，我们利用先进的预训练模型作为特征提取骨干网络，集成了属性分类网络和域分类网络，并在DCASE2025 Task2数据集上对其进行微调。最后，我们采用KNN模型作为后端来计算异常分数。得益于预训练模型强大的特征提取能力，我们的系统在开发集上取得了AUC和pAUC（p=0.1）的调和平均值60.9%的具有竞争力的结果。

**关键词：** 异常检测、预训练模型、微调、KNN

---

## 1. 引言

异常声音检测（Anomalous Sound Detection, ASD）是指识别目标机器发出的声音是正常还是异常的任务。DCASE 2025 Challenge Task 2[1, 2, 3, 4]——"面向机器状态监测的首次无监督异常声音检测"是DCASE 2020 Task 2至DCASE 2024 Task 2的延续。本次挑战的关键技术要点包括：

- **无监督学习（Unsupervised Learning）：** 仅依赖正常数据来进行异常表征。
- **域泛化（Domain Generalization）：** 算法对分布偏移的适应能力。
- **零样本适应（Zero-Shot Adaptation）：** 模型对新型机器类型的灵活性。
- **数据异质性（Data Heterogeneity）：** 处理有标签/无标签以及含噪/干净数据场景的能力。

预训练模型通常基于海量音频数据进行预训练，具有较强的泛化能力和通用的特征提取能力。在本工作中，我们利用预训练模型强大的泛化能力来解决源域与目标域之间以及不同机器之间的泛化挑战。我们将预训练模型用作特征提取器，后接一个分类头来区分属性或域。最后，采用KNN作为后端来计算异常分数。

本文按如下结构组织：第2节描述所提出的方法，并在该节末尾展示实验结果。第3节给出基于本报告的结论。

![图1：所提出ASD系统的架构](图1：所提出ASD系统的架构)

---

## 2. 所提出的ASD系统

### 2.1 骨干网络

EAT[5]是一种专为自监督音频学习设计的模型，致力于从无标签音频数据中进行高效的表示学习。它提出了一种新颖的目标函数，融合了全局语句级学习和局部帧级学习，从而增强了全面的音频理解能力。此外，EAT采用了一种针对音频领域定制的自举式自监督训练策略。我们使用在AudioSet-2M[9]上预训练的EAT基础模型，该模型包含8800万个参数。

图1展示了我们ASD系统的架构。我们采用EAT的编码器作为骨干网络，以梅尔频谱特征作为输入。这些特征随后被分割为16x16的patch，每个patch最终输出一个包含深层表征信息的嵌入向量。我们使用所有patch嵌入向量的平均池化作为分类器的输入。

### 2.2 微调

ASD系统通过在所有机器类型的数据上进行机器属性分类和域分类来进行微调。具体来说，由于部分机器缺少属性信息，我们对此类情况仅执行域分类。由于今年的数据额外提供了干净的机器数据和噪声数据，我们采取了进一步的措施：首先，对于提供了额外干净机器声音数据的机器，我们直接将其加入训练集作为扩充后的训练集；对于提供了噪声的机器类型，我们利用这些噪声对数据进行加噪处理以进行数据增强。

在微调过程中，我们使用ArcFace损失函数[6]，该任务的目标函数可用以下公式表示：

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s\cos(\theta_{y_i} + m)}}{e^{s\cos(\theta_{y_i} + m)} + \sum_{j=1, j\neq y_i}^{c}e^{s\cos\theta_j}}$$

其中 $y_i$ 是样本 $i$ 的标签，$s$ 和 $m$ 是两个超参数。$\theta_j$ 是样本 $i$ 的嵌入向量与第 $j$ 类的注册嵌入向量之间的夹角，即分类头权重 $W$ 的第 $j$ 列：

$$\theta_j = \arccos\left(\frac{x_i^T}{\|x_i\|_2 \cdot \|W_j\|_2} \cdot W_j\right)$$

其中 $x_i$ 是骨干网络的最终输出，$T$ 表示转置操作，$\|\cdot\|$ 表示L2范数距离。

#### 2.2.1 全微调（Full Fine-tune）

全微调（Full Fine-tune, FFT）是一种通过调整预训练模型的所有参数来使其适应特定任务或域需求的优化方法。我们对分类头和EAT骨干网络的所有参数进行了微调，使用AdamW优化器[11]训练20,000步，最大学习率为5e-5，批大小为32。该模型称为EAT-FFT。

#### 2.2.2 LoRA微调

除了FFT之外，我们还采用了低秩适应（Low-Rank Adaptation, LoRA）[10]方法进行模型参数调优，称之为EAT-LoRA。LoRA冻结预训练模型的权重，并在特定层（如注意力机制）中注入可训练的秩分解矩阵。这使得模型能够在保持大部分预训练知识的同时适应新任务。

LoRA的核心思想是利用低秩矩阵来近似参数更新，从而大幅减少可训练参数的数量。对于预训练模型的权重矩阵 $W$，LoRA将其更新分解为两个低秩矩阵 $A$ 和 $B$ 的乘积：

$$W = W_0 + \Delta W = W_0 + B \cdot A \cdot \alpha$$

其中 $W_0$ 代表预训练的权重矩阵，$B$ 和 $A$ 分别是维度为 $d \times r$ 和 $r \times k$ 的矩阵（$r \ll \min(d, k)$），$\alpha$ 是一个用于调整更新幅度的缩放因子。

### 2.3 后端

我们使用KNN[7]作为ASD系统的后端。我们将正常数据的嵌入向量作为库，计算评估集中每个样本的嵌入向量到所有正常数据嵌入向量的余弦距离，并取最小值作为异常分数（k=1）。

### 2.4 基于机理的分析

所有机器本质上都是由电动机驱动的，机器故障通常可归因于电机异常。因此，我们基于电机机理特性对机器进行异常检测，涉及AutoTrash、Polisher和ScrewFeeder。具体而言，对这三台机器进行了以下基于机理的分析：

**时域分析（Time-Domain Analysis）：** 通过计算均方根值（RMS）、峰值因子（Crest Factor, CF）、峭度（Kurtosis）和偏度（Skewness），对信号能量、冲击强度和分布特性进行定量表征。例如，在正常状态下，这些指标保持稳定（如峭度约等于3），而异常则表现为峭度显著增加（>6）或峰值因子急剧上升。

**频域分析（Frequency-Domain Analysis）：** 聚焦于频谱结构，该分析提取基频（旋转频率）和故障特征频率（如轴承内圈/外圈频率及边带频率）处的幅值。正常频谱表现出稳定的基频和谐波，而异常则表现为特征频率幅值的突然增加或边带的出现。

**时频分析（Time-Frequency Analysis）：** 采用小波变换和短时傅里叶变换来揭示非平稳信号中的瞬态特征。通过计算小波能量熵或时频聚集度来检测能量分布的局部突变。在正常条件下，能量分布均匀；异常表现为特定频带内能量熵的急剧下降或局部时频聚集，表明早期故障引起的瞬态冲击。

综合以上分析，我们建立了多域基线指标，并通过阈值比较实现了异常定位。对于使用基于机理分析的系统，我们用后缀"-MA"表示。

### 2.5 提交的系统

我们使用不同的学习率训练了ASD系统。此外，我们还使用了不同迭代步数的检查点。最后，我们采用集成学习策略[8]来整合上述提出的方法。为了平衡各个系统，我们基于不同检查点的分数分布进行z-score归一化，然后使用它们的加权和。

我们最终提交的System-1和System-3分别是EAT-LoRA和EAT-FFT，核心区别在于前者采用LoRA微调，而后者采用全微调策略。System-2和System-4分别命名为EAT-LoRA-MA和EAT-FFT-MA。在System-1和System-3的基础上，这两个系统将三台设备——AutoTrash、Polisher和ScrewFeeder——的异常分数替换为机理分析模型的输出结果。

### 2.6 实验结果

我们将我们的系统与DCASE 2025 Challenge Task 2的基线系统——AE-MSE和AE-MAHALA进行比较。我们最优系统的性能超越了基线系统，每台机器的AUC得分如表1所示。

**表1：开发集上各机器类型的AUC和pAUC**

| 机器类型 | 指标 | 我们的系统 | MSE基线 | Mahalanobis基线 |
|----------|------|-----------|---------|-----------------|
| **bearing** | AUC(source) | 66.53% | 63.63% | 65.32% |
| | AUC(target) | 53.15% | 59.03% | 47.82% |
| | pAUC | 61.12% | 61.86% | 50.95% |
| **fan** | AUC(source) | 70.96% | 77.99% | 52.80% |
| | AUC(target) | 38.75% | 38.56% | 58.68% |
| | pAUC | 49.46% | 50.82% | 53.79% |
| **gearbox** | AUC(source) | 64.80% | 73.26% | 69.36% |
| | AUC(target) | 50.49% | 51.61% | 72.82% |
| | pAUC | 52.49% | 55.07% | 57.42% |
| **slider** | AUC(source) | 70.10% | 73.79% | 72.12% |
| | AUC(target) | 48.77% | 50.27% | 57.12% |
| | pAUC | 52.32% | 53.61% | 52.53% |
| **ToyCar** | AUC(source) | 71.05% | 73.17% | 62.10% |
| | AUC(target) | 53.52% | 50.91% | 68.50% |
| | pAUC | 49.70% | 49.05% | 47.37% |
| **ToyTrain** | AUC(source) | 61.76% | 50.87% | 72.26% |
| | AUC(target) | 56.46% | 46.15% | 65.94% |
| | pAUC | 50.19% | 48.32% | 53.63% |
| **valve** | AUC(source) | 63.53% | 56.22% | 78.60% |
| | AUC(target) | 67.18% | 61.00% | 81.24% |
| | pAUC | 57.35% | 52.53% | 72.63% |
| **All (harmonic mean)** | AUC(source) | 66.78% | 65.51% | 66.54% |
| | AUC(target) | 51.39% | 50.05% | 62.91% |
| | pAUC | 52.94% | 52.72% | 54.60% |

---

## 3. 结论

本文提出了一种基于预训练模型增强的异常声音检测系统，用于DCASE2025 Task2，该任务聚焦于面向机器状态监测的首次无监督异常检测。该系统使用EAT作为特征提取骨干网络，通过属性分类和域分类进行微调，采用KNN进行异常评分，并通过集成学习整合模型。该系统优于基线系统，在开发集上取得了AUC和pAUC（p=0.1）的调和平均值60.9%。

---

## 4. 参考文献

1. Tomoya Nishida, Noboru Harada, Daisuke Niizumi, Davide Albertini, Roberto Sannino, Simone Pradolini, Filippo Augusti, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2025 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2506.10097, 2025.

2. N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, "ToyADMOS2: Another dataset of miniature machine operating sounds for anomalous sound detection under domain shift conditions," in Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), Barcelona, Spain, November 2021, pp. 1–5.

3. K. Dohi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, Y. Nikaido, and Y. Kawaguchi, "MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), Nancy, France, November 2022.

4. N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, and M. Yasuda, "First-shot anomaly detection for machine condition monitoring: A domain generalization baseline," Proceedings of 31st European Signal Processing Conference (EUSIPCO), pp. 191–195, 2023.

5. W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, "EAT: Self-supervised pre-training with efficient audio transformer," in Proceedings of the 33rd International Joint Conference on Artificial Intelligence, 2024.

6. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4690–4699.

7. S. Ramaswamy, R. Rastogi, and K. Shim, "Efficient algorithms for mining outliers from large data sets," in Proc. 2000 ACM SIGMOD Int. Conf. Manag. Data, 2000, pp. 427–438.

8. R. L. Sagi Omer, "Ensemble learning: A survey," Wiley interdisciplinary reviews. Data mining and knowledge discovery, vol. 8, 2018.

9. Jort Gemmeke, Daniel Ellis, Dylan Freedman, Aren Jansen, et al. Audio set: An ontology and human-labeled dataset for audio events. In Proc. ICASSP. IEEE, 2017.

10. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in International Conference on Learning Representations, 2022.

11. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in International Conference on Learning Representations, 2019.
