# 声学场景与事件检测与分类 2025 挑战赛

## 面向任务2的两阶段融合异常检测方法

### 技术报告

**Jie Yang**

中国 淮北

852549193@qq.com

---

## 摘要

本技术报告详细介绍了我们针对 DCASE 2025 挑战赛任务2所提出的方法。我们提出了一种两阶段融合自适应异常检测方案，该方案将自适应滤波去噪与分离技术同基于神经网络的分类器相结合。首先，采用传统信号分离与去噪技术对原始音频信号进行预处理，此阶段重点在于抑制噪声、隔离干扰声源并提升目标机器声音的信噪比（SNR）。在属性分类网络方面，我们利用 MobileFaceNet 的深度可分离卷积与瓶颈结构来高效学习异常声音的深度判别特征。最终，基于 k-means 和余弦距离计算异常分数。

结果表明，所提出的方法在不同的数据条件下均取得了显著的性能提升，并确保了稳健的适应性。此外，该框架在处理不同类型输入数据方面的灵活性增强了其在真实工业机器监测场景中的适用性。

**关键词**—— 两阶段融合，自适应异常检测，自适应滤波

---

## 1. 引言

随着社会和技术的进步，机械设备在工业生产中变得越来越关键。然而，各种因素往往导致设备在运行过程中发生故障，这可能影响生产效率与性能，甚至引发严重的安全事故。异常声音检测（ASD）是机器状态监测工具的重要组成部分，其任务是识别目标机器发出的声音是正常的还是异常的 [1]。

今年任务2的主题是面向机器状态监测的首次无监督异常声音检测（First-shot unsupervised anomalous sound detection for machine condition monitoring）[2]，该任务是 DCASE 2020 任务2至 DCASE 2024 任务2 [3] 的延续。

针对上述情况，我们提出了一种自适应异常检测框架，该框架能够确保无论是否存在属性信息，系统都能高效运行，更贴近工业实际情况。该检测框架是一种两阶段融合自适应异常检测方案，将自适应滤波去噪与分离技术同基于神经网络的分类器相结合。首先，我们考虑采用基于自适应滤波的预处理方案，对含噪机器声音进行降噪或分离操作，进一步提升机器声音的信噪比（SNR）。对于带有属性的机器，我们通过按所有属性进行细分来执行详细的异常检测。在缺乏属性信息的情况下，我们利用领域特定信息进行有效检测。最后，我们通过 MobileFaceNet [4] 提取嵌入向量，并使用 ArcFace [5] 作为模型的损失函数，后端则采用异常检测算法来评估异常程度。

---

## 2. 方法

### 2.1. 自适应滤波预处理

自适应滤波的特点在于其滤波器的参数（如权重）会随着输入信号的变化而不断调整。即使当信号的统计特性发生变化时，自适应滤波器也能够自动调整其参数以适应新的信号特性。因此，自适应滤波通常用于信号统计特性不可预测或不断变化的场景，如噪声消除、回声消除、信道估计等。受此启发，针对任务2中机器声音与噪声混合的场景，我们尝试滤除混入机器声音中的噪声和干扰信号。

自适应滤波器的原理框图如下图所示。输入信号 x(n) 经过一个参数可调的数字滤波器产生输出信号 y(n)，将 y(n) 与期望信号 d(n) 进行比较，形成误差信号 e(n)。通过自适应算法调整滤波器参数，最终使 e(n) 的均方值最小。自适应滤波可以利用前一时刻获得的滤波器参数结果，自动调整当前时刻的滤波器参数，以适应信号和噪声未知或时变的统计特性，从而实现最优滤波。自适应滤波器本质上是一种能够调整自身传输特性以达到最优效果的维纳滤波器。自适应滤波器不需要关于输入信号的先验知识，计算量小，特别适用于实时处理。维纳滤波器的参数是固定的，适用于平稳随机信号。卡尔曼滤波器的参数是时变的，适用于非平稳随机信号。

**图1：自适应滤波器原理框图。**

对于提供噪声数据的机器，我们设定：x(n) 为原始含噪音频，d(n) 为纯噪声音频，y(n) 为估计的噪声音频。在后续推理过程中，通过从 x(n) 中减去 y(n) 即可获得滤波后的音频。对于不提供噪声数据的机器，我们尝试从原始含噪音频中提取纯噪声片段，并执行上述滤波过程。如果无法提取纯噪声片段，则不执行滤波，直接使用原始音频。

### 2.2. 分类模型

我们使用 DCASE 2025 的数据，并将滤波后的数据作为异常声音检测系统的输入。整个异常声音检测系统由前端特征提取器和后端异常检测器组成。前端特征提取器作为一个有监督的属性分类模型，构建在 MobileFaceNet 网络结构之上。该属性分类模型同时受机器类型、属性和领域信息的监督。为了最小化类内距离并最大化类间距离，采用 ArcFace 作为模型的损失函数。

### 2.3. 特征提取

在特征提取阶段，我们深入分析了四种特征图像，包括语谱图（spectrogram）、原始波形图、梅尔语谱图（Mel-spectrogram）和短时傅里叶变换（STFT）语谱图。为确保所选特征的有效性和准确性，我们在开发集上从多个角度对这些特征进行了组合验证。经过严格的实验和分析，最终选择了梅尔语谱图和 STFT 语谱图这两种特征图像作为输入。同时，在物理层面上，这两种特征分别关注声音信息和振动信息。对于这两种特征表示，我们均使用了窗长为1024、窗移为512的幅度语谱图。将这两种特征分别输入基于 MobileFaceNet 的嵌入向量提取网络进行处理。通过这一过程，我们获得两种特征向量，并将它们组合起来形成最终的嵌入特征向量。

### 2.4. 后端异常检测器

系统的后端异常检测器由三个步骤组成 [6]。对于源域，应用 k-means 算法为每种机器类型获取多个类别中心，然后计算给定测试样本与这些类别中心之间的所有余弦距离。而对于目标域，则计算给定测试样本与同一机器类型的所有10个正常样本之间的余弦距离。最后，对于一个测试样本，将该样本计算得到的所有余弦距离中的最小值作为该样本的异常分数。因此，样本的异常分数可以指示该样本是否异常。

---

## 3. 结果

我们的系统包含不同类型的滤波器，并设置了不同的滤波参数。将所提出的系统与 DCASE 2025 挑战赛任务2的基准系统（即 Baseline MSE 和 Baseline MAHALA [7]）进行对比。如表1所示，在开发集上，我们的系统在大多数机器类型上优于基准系统。

**表1：所提出系统的异常检测结果**

| 方法 | Baseline MSE | Baseline MAHALA | Proposed system |
|------|-------------|-----------------|-----------------|
| **ToyCar** | | | |
| AUC(source) | 71.05% | 73.17% | 61.58% |
| AUC(target) | 53.52% | 50.91% | 67.04% |
| pAUC | 49.70% | 49.05% | 51.37% |
| score | 56.73% | 55.87% | 59.26% |
| **ToyTrain** | | | |
| AUC(source) | 61.76% | 50.87% | 75.76% |
| AUC(target) | 56.46% | 46.15% | 65.74% |
| pAUC | 50.19% | 48.32% | 50.37% |
| score | 55.73% | 48.37% | 62.16% |
| **bearing** | | | |
| AUC(source) | 66.53% | 63.63% | 63.18% |
| AUC(target) | 53.15% | 59.03% | 49.54% |
| pAUC | 61.12% | 61.86% | 49.00% |
| score | 59.75% | 61.45% | 53.17% |
| **fan** | | | |
| AUC(source) | 70.96% | 77.99% | 57.76% |
| AUC(target) | 38.75% | 38.56% | 61.60% |
| pAUC | 49.46% | 50.82% | 53.95% |
| score | 49.90% | 51.34% | 57.60% |
| **gearbox** | | | |
| AUC(source) | 64.80% | 73.26% | 75.84% |
| AUC(target) | 50.49% | 51.61% | 76.92% |
| pAUC | 52.49% | 55.07% | 59.63% |
| score | 55.26% | 58.61% | 69.84% |
| **slider** | | | |
| AUC(source) | 70.10% | 73.79% | 77.46% |
| AUC(target) | 48.77% | 50.27% | 65.04% |
| pAUC | 52.32% | 53.61% | 57.21% |
| score | 55.68% | 57.58% | 65.55% |
| **valve** | | | |
| AUC(source) | 63.53% | 56.22% | 70.92% |
| AUC(target) | 67.18% | 61.00% | 82.76% |
| pAUC | 57.35% | 52.53% | 54.05% |
| score | 62.42% | 56.37% | 67.14% |
| **All (hmean)** | | | |
| AUC(source) | 66.78% | 65.51% | 68.11% |
| AUC(target) | 51.39% | 50.05% | 65.42% |
| pAUC | 52.94% | 52.72% | 53.43% |
| score | 56.26% | 55.34% | 61.62% |

---

## 4. 结论

在本技术报告中，我们介绍了提交至 DCASE 2025 挑战赛任务2的系统。我们提出了一种基于自适应滤波预处理和 MobileFaceNet 网络的属性分类方案，结合后端 k-means 算法，实现了对异常声音的精确检测。当属性信息完整时，我们对每种属性进行详细划分，从而实现对异常情况的深度监测。即使在属性信息缺失的情况下，我们也将领域信息作为属性进行划分，以有效识别异常状况。

此外，在特征提取阶段，我们综合考虑了工业环境中声音和振动两个关键维度上机器的特性。我们融合了梅尔语谱图和短时傅里叶变换（STFT）语谱图的优势，这一创新举措显著提升了异常检测系统的普适性和有效性。这种集成化的特征提取方法确保了系统能够更准确地识别和处理来自不同来源和环境的异常声音数据。

---

## 5. 参考文献

[1] Tomoya Nishida, Noboru Harada, Daisuke Niizumi, Davide Albertini, Roberto Sannino, Simone Pradolini, Filippo Augusti, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2025 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2506.10097, 2025.

[2] Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1–5. Barcelona, Spain, November 2021.

[3] Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November 2022.

[4] C. D. Jones, A. B. Smith, and E. F. Roberts, "A sample paper in conference proceedings," in Proc. IEEE ICASSP, 2003, vol. II, pp. 803-806.

[5] A. B. Smith, C. D. Jones, and E. F. Roberts, "A sample paper in journals," IEEE Trans. Signal Process., vol. 62, pp. 291-294, Jan. 2000.

[6] Wilkinghoff K. Design Choices for Learning Embeddings from Auxiliary Tasks for Domain Generalization in Anomalous Sound Detection[C]//ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023: 1-5.

[7] Harada, Noboru and Niizumi, Daisuke and Ohishi, Yasunori and Takeuchi, Daiki and Yasuda, Masahiro. First-Shot Anomaly Sound Detection for Machine Condition Monitoring: A Domain Generalization Baseline. In 2023 31st European Signal Processing Conference (EUSIPCO), 191-195. 2023.
