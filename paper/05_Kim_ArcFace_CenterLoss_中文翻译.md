# DCASE 2025 声学场景与事件检测与分类挑战赛

# AISTAT实验室用于DCASE 2025任务2的系统：面向机器状态监测的首次无监督异常声音检测

## 技术报告

**Hyun Jun Kim¹, Min Jun Kim², Hyeon Gyu Bae², Changwon Lim¹,²,\***

¹ 韩国首尔中央大学智慧城市系  
² 韩国首尔中央大学应用统计学系  
{hyunjun0615, goodwill1669, amysst11, clim}@cau.ac.kr  
\* 通讯作者

---

## 摘要

本报告介绍了AISTAT团队在DCASE2025任务2——首次无监督异常声音检测中的提交方案。与往年的挑战不同，可用的训练数据涵盖了从2020年到2025年的训练数据集。为有效利用给定数据，我们采用了两阶段训练策略，由预训练和迁移学习组成。在迁移学习阶段，对缺乏属性信息的数据应用伪标签以分配近似标签并增强模型适应性。此外，我们联合使用ArcFace损失和Center损失来直接降低类内方差。另外，为提取更具信息量的音频表征，我们利用了多层聚合方法。通过这些技术，我们的单个最佳模型达到了66.12的调和平均分数，而最佳集成模型达到了66.78的调和平均分数。

**索引关键词**——异常声音检测、两阶段训练、伪标签、类内方差、多层聚合

---

## 1. 引言

首次无监督异常声音检测是一项仅使用正常声音数据进行训练的任务，其核心目标是有效学习正常数据的特征分布。DCASE2025任务2[1-4]的一个显著特点是，可以使用2020年至2025年的训练数据，而评估仅在2025年的测试数据上进行。此外，与DCASE2024任务2[5]类似，部分训练数据不包含属性信息。另外，干净的噪声数据和干净的机器声音数据作为补充数据提供，按机器随机分配。在本研究中，我们将问题构造为分类任务来训练特征提取器，其中机器属性被视为类别标签[6,7]。特征提取器训练完成后，我们在测试阶段使用K近邻（KNN）检测器，基于测试样本与由正常数据构成的训练集之间的距离来计算异常分数，从而识别异常实例。

尽管可用的训练数据横跨2020年至2025年，但测试仅限于2025年的数据。因此，我们将训练过程分为两个阶段。在第一阶段，在2020-2025年的数据上进行预训练，以学习机器声音的通用表征。在第二阶段，进行迁移学习，使模型专门适应2025年的数据。对于缺乏属性信息的数据，我们在迁移学习之前应用伪标签来补充缺失的标签。

由于在测试阶段异常分数是基于与正常训练样本的距离计算的，异常数据必须远离所有类别的正常特征。因此，特征提取器必须经过训练以最小化类内方差，同时最大化类间可分性。为达到这一目的，我们在预训练期间使用ArcFace损失来鼓励紧凑的类内表征和更大的类间距离，并在迁移学习期间进一步应用Center损失来直接降低类内方差。

在异常声音检测中，从音频中提取丰富且有信息量的表征对于有效区分正常和异常声音至关重要。为此，我们采用了多层聚合方法，整合了跨多个Transformer层的patch嵌入。我们的实验表明，与仅使用最后一层的patch嵌入相比，该方法能产生更优的性能。

本文的其余部分组织如下。第2节详细描述所提出的方法。第3节介绍实验设置、结果以及对我们系统的评估。最后，第4节总结我们的工作。

---

## 2. 方法

本节概述了为本挑战开发的方法。我们的方法利用两阶段框架来学习鲁棒且具有判别力的音频表征，以应对多样化的机器类型、变化的运行条件和领域偏移等挑战。我们系统的整体架构如图1所示，后续小节将详细阐述预训练和迁移学习阶段、特征聚合策略、损失函数、伪标签以及测试流程。

### 2.1 两阶段框架

在预训练阶段，我们利用2020年至2025年的整个数据集来学习机器声音的通用特征。该数据集包含多样化的机器类型，如ToyCar、Fan、Gearbox等，其中一些并未出现在2025年的数据中，从而有助于学习跨不同机器类型和运行条件的广泛声学特性。为确保数据完整性，我们删除了跨年份的重复数据。每个独特的机器类型和属性信息（如运行速度、环境噪声）的组合被视为一个独立的类别，这与我们基于分类的特征学习方法保持一致。这一策略鼓励音频编码器提取具有判别力的特征，以捕获每个机器属性对的独特声学特征。

然而，将目标域包含在预训练阶段会引入严重的类别不平衡，因为这些数据集可能包含不同的分布或某些机器/条件下的样本较少。这种不平衡对我们的基于分类的方法是有害的，因为它可能使模型偏向于过表征的类别，降低其有效泛化的能力。为缓解这一问题，我们从预训练阶段排除了所有目标域数据，仅专注于2020年至2025年的源域数据。这确保了平衡的类别分布，使模型能够学习鲁棒且可泛化的特征，而不会被目标域的特定特征所偏斜。

预训练使用ArcFace损失函数进行，该函数通过在特征空间中引入角度间隔来增强类别可分性，详见第2.4节。由此产生的预训练模型为捕获通用机器声音特征提供了坚实的基础，这对于后续的迁移学习阶段至关重要。

在迁移学习阶段，我们使用整个2025年训练数据集（包括源域和目标域数据）对预训练模型进行微调。与预训练阶段一致，每个机器类型和属性信息的组合被视为分类的一个独立类别。这种方法确保模型在适应2025年数据集的特定特征（包括潜在的领域偏移）的同时，继续利用预训练期间建立的判别特征空间。

为提升模型性能，我们探索了各种损失函数组合，将Center损失与广泛采用的ArcFace损失结合使用。ArcFace损失保持类别可分性，而Center损失通过将特征拉向其各自的类别中心来鼓励类内紧凑性，详见第2.4节。

此外，对于2025年数据集中缺乏属性信息的机器，我们应用了基于预训练权重的伪标签策略来分配临时标签。该过程涉及提取特征嵌入、使用UMAP进行可视化、并通过聚类确定伪标签类别，详见第2.5节。伪标签数据被整合到迁移学习阶段中，扩充了训练集，提高了模型对属性信息不完整的机器的泛化能力。

**图1：所提出系统的示意图。** 音频特征通过对音频编码器中所有Transformer层的输出进行层聚合得到。这些输出经过注意力统计池化层处理以生成最终嵌入。在训练过程中，使用基于分类的损失函数对该嵌入进行优化，以区分机器类型和属性信息。在测试阶段，最终的异常分数通过训练数据和测试数据的嵌入之间的K近邻（KNN）计算得到。注意，预训练阶段仅使用ArcFace损失，而迁移学习阶段额外引入Center损失。

### 2.2 多层聚合

多层聚合策略最早在说话人识别领域被提出[8]，是一种利用Transformer模型所有层的输出来构建全面特征表示的方法。在说话人识别中，该方法被用于捕捉局部特征，如音高、语调风格和发音模式，同时建模全局上下文以考虑变长语音序列中固有的长距离依赖关系[9]。通过整合这些多样化的特征，模型实现了鲁棒的表征，同时涵盖了细粒度和整体性的音频特性。

在我们的工作中，我们采用了这种方法，聚合来自低层和高层的特征以同时捕捉局部和全局音频特性。低层提取细粒度模式，如特定频率分量或短时时序变化，而高层则编码长期运行特征，如机器的周期性嗡鸣声。为实现这一目标，我们将所有Transformer层的输出沿特征维度进行拼接，然后依次经过层归一化、一个线性层、GELU激活和另一个线性层。这一过程将多尺度特征整合为统一的表征，增强了模型处理多样化音频模式的能力。

通过结合对不同时间尺度敏感的特征，多层聚合增强了模型检测多样化异常的能力，从短时事件（如突发的高频噪声）到长期偏离（如节奏模式的破坏）均能覆盖。此外，该方法提高了跨域泛化能力。低层提供域不变特征，如基本频率模式，而高层则捕捉域特定信息，如环境噪声特征。我们的实验表明，多层聚合显著提升了模型性能，在数据有限的目标域中改进尤为明显。

### 2.3 注意力统计池化

为有效聚合音频编码器的序列输出，我们采用了注意力统计池化，这是一种广泛应用于音频处理中的技术，用于捕捉变长序列的局部和全局特性[10]。该方法增强了模型从Transformer的时间分布输出中生成固定长度表征的能力，这对于下游的分类和异常检测任务至关重要。

在我们的实现中，注意力统计池化应用于Transformer编码器的输出，紧随第2.2节所述的多层聚合步骤之后。池化后的表征通过两个线性层进行处理，中间使用ReLU激活函数，dropout率为0.2，其中中间维度设置为输入维度的三倍，最终投影到768维空间以输入损失函数。

### 2.4 损失函数

ArcFace损失，即加性角度间隔损失，最早在人脸识别中被提出，此后因其增强判别能力的能力而被广泛应用于各种分类任务[11]。该损失函数通过在标准softmax损失的基础上添加角度间隔，强制不同类别在特征空间中具有更大的角度距离，从而提升类别可分性。

ArcFace损失在区分不同机器类型及其属性（如运行速度或环境噪声条件）方面特别有效。通过将每个独特的机器类型和属性信息组合视为一个独立类别，ArcFace损失确保音频编码器学习提取高度判别性的特征。这对于DCASE2025任务2至关重要，因为模型必须跨域泛化，特征空间中清晰的类别分离有助于实现鲁棒的异常检测。

在迁移学习阶段，我们的目标是使预训练模型适应当年度的训练数据集。为实现这一目标，我们在ArcFace损失的基础上引入了Center损失。Center损失旨在为每个类别学习一个中心，并将同一类别的特征向量拉向其各自的中心，从而在保持ArcFace损失所强制的类间分离的同时减少类内变化[12]。Center损失的数学定义如下：

$$
L_C = \frac{1}{2} \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2,
\tag{1}
$$

其中 $m$ 是mini-batch中的样本数，$x_i$ 表示第 $i$ 个样本的特征嵌入，$c_{y_i}$ 表示对应类别标签 $y_i$ 的可学习类别中心，$\|\cdot\|_2^2$ 是平方L2范数。该公式鼓励同一类别的特征嵌入紧紧围绕各自的中心聚集，增强模型对目标数据集的泛化能力。类别中心 $c_{y_i}$ 在训练过程中更新以反映不断变化的特征分布，补充了ArcFace损失提供的基于角度间隔的分离。

在迁移学习期间，我们探索了四种迁移学习损失配置。首先，我们单独使用ArcFace损失，如预训练阶段一样，以保持机器类型和属性的强大类别可分性。其次，按照Center损失的原始实现，我们使用交叉熵（CE）损失与Center损失的组合作为损失函数。第三，我们将Focal损失与Center损失结合。Focal损失是一种通过聚焦于难以分类的样本来解决类别不平衡问题的损失函数，同时降低对易于分类样本的权重[13]。其公式如下：

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t).
\tag{2}
$$

我们在所有实验中将 $\gamma$ 设为5。最后，我们将ArcFace损失与Center损失结合，Center损失权重为0.9，ArcFace损失权重为0.1以计算最终损失。判别性损失（CE、Focal或ArcFace）与Center损失的组合确保了类别可分性得以保留，同时Center损失通过鼓励特征向其类别中心收敛来增强类内紧凑性，从而显著提升性能。

**表1：基于预训练权重进行伪标签后的类别数量。**

| 机器 | EAT | BEATs |
|------|-----|-------|
| ToyTrain | 4 | 4 |
| Slider | 5 | 5 |
| Bearing | 5 | 4 |
| AutoTrash | 2 | 2 |
| Polisher | 4 | 4 |
| ScrewFeeder | 5 | 3 |
| ToyPet | 4 | 3 |
| **总计** | **29** | **25** |

### 2.5 伪标签

为利用缺乏属性信息的机器数据，我们实现了伪标签策略来提升模型性能。该方法为无标签数据分配临时标签，使其能够在迁移学习中使用。伪标签过程首先使用预训练模型从无标签机器中提取特征嵌入。为分析这些嵌入的结构，我们应用了统一流形逼近与投影（UMAP）[14]来降低维度并对其进行二维可视化。UMAP保留数据的局部和全局结构，使其能够有效识别可能对应于不同运行条件或机器状态的聚类。

基于UMAP可视化，我们采用双重方法来分配伪标签。对于在特征空间中表现出明显聚类的机器，伪标签类别的数量直接由观察到的聚类数量确定，反映了数据中的自然分组。对于聚类模糊或重叠的机器，我们采用凝聚层次聚类来生成树状图，该图表示数据的层次结构。使用自顶向下的方法，我们解读树状图并设置20到40之间的距离阈值以识别候选伪标签类别。该阈值通过经验选取，以平衡聚类的粒度，确保有意义且明确的类别分配。

为解决潜在的数据不平衡问题，我们计算了每个候选伪标签类别的样本数量，并为每种机器类型选择不平衡程度最小的配置。数据不平衡（即某些类别的样本数量远多于其他类别）会偏置模型训练，因此此步骤确保了伪标签的均衡表征。该过程针对每种机器类型单独应用，认识到它们独特的声学特性和聚类行为。伪标签产生的类别数量如表1所示。

### 2.6 测试阶段

在测试阶段，训练好的音频编码器为训练和测试音频样本生成嵌入。KNN算法用于计算测试样本嵌入与正常训练样本嵌入之间的距离。异常分数从这些距离中导出，通常取到k个最近邻的平均距离，距离越大表示异常可能性越高。我们在所有实验中使用k=1。此外，为解决类别不平衡问题，使用SMOTE为目标域中欠代表的正常样本生成合成嵌入，平衡用于KNN异常检测的数据集。此外，为考虑机器之间的差异，我们从训练数据集中计算每个机器的统计量，并使用对应的均值和标准差对每个机器的测试样本嵌入进行归一化。

---

## 3. 实验

本节概述实验设置和结果。关于补充数据这一今年竞赛的显著特征，我们尝试了使用干净噪声的数据增强技术，但未能获得有意义的结果。

### 3.1 实验设置

我们在实验中使用了EAT[15]和BEATs[16]模型。输入音频按照各模型预训练方法相同的方式进行预处理。为高效训练，我们对BEATs和EAT模型的query、value和projection层应用了低秩适配（LoRA）[17]，秩设为64。所有系统训练40个epoch，通过监控开发测试集上AUC源域、AUC目标域和pAUC的调和平均值（hmean）来选择表现最佳的epoch权重。

当保持与预训练相同的可训练参数数量时，我们观察到严重的过拟合。为解决这一问题，我们冻结了大部分模型参数，并专门对用于层聚合的适配器层应用LoRA。此外，在迁移学习阶段，我们冻结了现有模型参数，仅更新适配器的LoRA参数。然而，当使用Center损失更新EAT模型时，我们发现将所有模型参数设为可学习（而不使用适配器LoRA）能获得更好的性能。

学习率使用余弦预热重启调度进行更新。对于预训练，初始学习率设为1e-5，最小学习率设为1e-7，预热1个epoch，每5个epoch重启一次。权重衰减设为0.0001，批大小设为32，累积步数设为8。

对于迁移学习，初始学习率设为1e-7，最小学习率设为5e-10，权重衰减设为0.01，批大小设为16，累积步数设为16。其他设置与预训练保持一致。

### 3.2 实验结果

表2总结了我们系统的性能。具体而言，层聚合的引入显著提升了两种模型在目标域上的AUC性能。值得注意的是，EAT模型的目标域AUC从66.64提升至71.58，而BEATs模型从61.94提升至70.41。两种模型之间的调和平均值差异不大。然而，EAT在目标域上通常优于BEATs，而BEATs在源域上表现更好。基于EAT和BEATs的最佳系统分别达到了66.12和65.66的调和平均值。

**表2：单系统和集成系统的性能。** 每种模型的最佳性能已加粗标注。注意，System ID对应表3中描述的集成中使用的系统索引，而Submission ID指最终提交中使用的索引。

（表2数据略，见原文，主要内容如下：）

| 模型 | 阶段 | 聚合 | 损失 | AUC源域 | AUC目标域 | pAUC | hmean | System ID |
|------|------|------|------|---------|----------|------|-------|-----------|
| EAT | 1 | X | ArcFace | 68.44 | 66.64 | 56.62 | 63.45 | - |
| EAT | 1 | O | ArcFace | 67.24 | 71.58 | 57.41 | 64.85 | - |
| EAT | 2 | O | ArcFace | 67.80 | 72.70 | 57.73 | 65.52 | 1 |
| EAT | 2 | O | CE + Center | 69.33 | 72.71 | 58.07 | 66.09 | 2 |
| EAT | 2 | O | Focal + Center | 69.41 | 72.74 | 58.07 | **66.12** | - |
| EAT | 2 | O | ArcFace + Center | 68.55 | 73.31 | 57.61 | 65.81 | 3 |
| BEATs | 1 | X | ArcFace | 69.70 | 61.94 | 56.06 | 62.07 | - |
| BEATs | 1 | O | ArcFace | 69.74 | 70.41 | 56.71 | 64.97 | - |
| BEATs | 2 | O | ArcFace | 70.18 | 70.77 | 57.60 | 65.59 | 4 |
| BEATs | 2 | O | CE + Center | 69.84 | 70.83 | 57.62 | 65.52 | 5 |
| BEATs | 2 | O | Focal + Center | 70.01 | 70.74 | 57.55 | 65.55 | - |
| BEATs | 2 | O | ArcFace + Center | 69.90 | 70.92 | 57.95 | **65.66** | 6 |

| 集成 | 提交ID | AUC源域 | AUC目标域 | pAUC | hmean |
|------|--------|---------|----------|------|-------|
| Ensemble 1 | 1 | 69.49 | 72.82 | 59.51 | **66.78** |
| Ensemble 2 | 2 | 69.39 | 72.88 | 59.49 | 66.75 |
| Ensemble 3 | 3 | 69.63 | 72.86 | 59.21 | 66.70 |
| Ensemble 4 | 4 | 69.61 | 72.80 | 59.25 | 66.70 |

为确定最终提交的系统，我们进行了网格搜索以探索单系统的各种组合，使用的权重如表3所示。在此探索过程中，我们观察到使用Focal损失与Center损失组合的系统始终表现出相对较低的集成性能。因此，这些系统被排除在最终提交之外。提交1和2是通过为多个单系统找到最优集成权重生成的。相比之下，提交3和4分别通过对使用Center损失以及ArcFace损失与Center损失组合配置的系统进行模型级集成而创建。集成系统在每种机器上的结果如表4所示。

**表3：四个提交系统的组合系数。**

|  | System 1 | System 2 | System 3 | System 4 | System 5 | System 6 |
|------|------|------|------|------|------|------|
| Ensemble 1 | 0.18 | 0 | 0.42 | 0 | 0.04 | 0.36 |
| Ensemble 2 | 0.18 | 0.18 | 0.24 | 0.12 | 0.12 | 0.16 |
| Ensemble 3 | 0 | 0.6 | 0 | 0 | 0.4 | 0 |
| Ensemble 4 | 0 | 0 | 0.6 | 0 | 0 | 0.4 |

---

## 4. 结论与未来工作

本文描述了AISTAT实验室用于首次无监督异常声音检测的系统。针对与以往挑战相比可用的训练数据有所扩展这一变化，我们采用了两阶段训练框架以最大化数据利用率。此外，我们使用层聚合来整合多尺度音频表征，同时捕捉细粒度声学模式和长距离上下文依赖关系。在迁移学习阶段，我们使用Center损失增强了标准的ArcFace损失，在保持类别可分性的同时促进了类内凝聚性，从而获得了显著的性能提升。作为未来工作，我们计划探索整合本研究中未使用的干净噪声和干净机器声音数据，以进一步增强模型的鲁棒性和性能。

---

## 5. 致谢

本研究得到了中央大学2024年研究奖学金资助的支持。

---

## 6. 参考文献

[1] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2025 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring," In arXiv e-prints: 2506.10097, 2025.

[2] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, "ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions," in Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1–5. Barcelona, Spain, November, 2021.

[3] K. Dohi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, Y. Nikaido, and Y. Kawaguchi, "MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November, 2022.

[4] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, and M. Yasuda, "First-shot anomaly detection for machine condition monitoring: a domain generalization baseline," in Proceedings of 31st European Signal Processing Conference (EUSIPCO), pages 191–195, 2023.

[5] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2024 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring," In arXiv e-prints: 2406.07250, 2024.

[6] Z. Lv, A. Jiang, B. Han, Y. Liang, Y. Qian, X. Chen, J. Liu, P. Fan, "AITHU system for first-shot unsupervised anomalous sound detection," in Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), Tokyo, Japan, October 2024.

[7] A. Jiang, X. Zheng, Y. Qiu, W. Zhang, B. Chen, P. Fan, W. Q. Zhang, C. Lu, J. Liu, "THUEE system for first-shot unsupervised anomalous sound detection," in Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), Tokyo, Japan, October 2024.

[8] J.-w. Jung, Y. J. Kim, H.-S. Heo, B.-J. Lee, et al., "Pushing the limits of raw waveform speaker recognition," in Proc. Interspeech, 2022.

[9] Y. Zhang, Z. Lv, H. Wu, S. Zhang, et al., "MFA-Conformer: Multi-scale feature aggregation conformer for automatic speaker verification," in Proc. Interspeech, 2022.

[10] B. Desplanques, J. Thienpondt, and K. Demuynck, "Ecapatdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification," arXiv preprint arXiv:2005.07143, 2020.

[11] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "Arcface: Additive angular margin loss for deep face recognition," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4690–4699.

[12] Y. Wen, K. Zhang, Z. Li, Y. Qiao, "A discriminative feature learning approach for deep face recognition," in Computer vision–ECCV 2016: 14th European conference, amsterdam, the netherlands, proceedings, part VII 14 (pp. 499-515). Springer International Publishing, October 11–14, 2016.

[13] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in Proceedings of the IEEE international conference on computer vision (pp. 2980-2988), 2017.

[14] L. McInnes, J. Healy, J. Melville, "Umap: Uniform manifold approximation and projection for dimension reduction," arXiv preprint arXiv:1802.03426, 2018.

[15] W. Chen, Y. yang, Z. Ma, Z. Zheng, and X. Chen, "EAT: Self-supervised pre-training with efficient audio transformer," arXiv preprint arXiv:2401.03497, 2024.

[16] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, and F. Wei, "Beats: Audio pre-training with acoustic tokenizers," arXiv preprint arXiv:2212.09058, 2022.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, ... and W. Chen, "Lora: Low-rank adaptation of large language models," ICLR, 1(2), 3, 2022.

---

**表4：四个提交系统在开发集上的检测结果。**

| 机器 | 指标 | Ensemble 1 | Ensemble 2 | Ensemble 3 | Ensemble 4 |
|------|------|------------|------------|------------|------------|
| ToyCar | AUC_s | 62.52 | 62.84 | 62.48 | 62.48 |
| ToyCar | AUC_t | 69.72 | 69.76 | 68.96 | 68.96 |
| ToyCar | pAUC | 51.00 | 51.63 | 50.68 | 50.42 |
| ToyCar | hmean | 60.07 | 60.46 | 59.72 | 59.88 |
| ToyTrain | AUC_s | 73.72 | 73.36 | 74.24 | 74.40 |
| ToyTrain | AUC_t | 77.08 | 77.12 | 77.36 | 76.96 |
| ToyTrain | pAUC | 60.42 | 60.21 | 61.37 | 61.68 |
| ToyTrain | hmean | 69.62 | 69.43 | 70.27 | 70.35 |
| Bearing | AUC_s | 57.20 | 57.28 | 57.44 | 57.52 |
| Bearing | AUC_t | 74.84 | 75.32 | 75.08 | 74.56 |
| Bearing | pAUC | 56.16 | 56.21 | 55.84 | 55.58 |
| Bearing | hmean | 61.66 | 61.82 | 61.68 | 61.49 |
| Fan | AUC_s | 62.24 | 62.60 | 62.44 | 62.08 |
| Fan | AUC_t | 62.04 | 61.52 | 62.48 | 62.08 |
| Fan | pAUC | 52.00 | 51.79 | 51.84 | 51.84 |
| Fan | hmean | 58.35 | 58.21 | 58.47 | 58.25 |
| Gearbox | AUC_s | 68.76 | 68.72 | 70.16 | 70.76 |
| Gearbox | AUC_t | 79.16 | 79.36 | 79.40 | 78.84 |
| Gearbox | pAUC | 66.53 | 66.79 | 66.16 | 66.89 |
| Gearbox | hmean | 71.08 | 71.22 | 71.49 | 71.83 |
| Slider | AUC_s | 90.32 | 90.32 | 90.48 | 90.36 |
| Slider | AUC_t | 74.88 | 74.68 | 74.76 | 74.04 |
| Slider | pAUC | 64.63 | 64.05 | 63.84 | 63.79 |
| Slider | hmean | 75.19 | 74.86 | 74.83 | 74.53 |
| valve | AUC_s | 82.24 | 82.40 | 81.16 | 80.44 |
| valve | AUC_t | 75.40 | 75.20 | 74.88 | 75.80 |
| valve | pAUC | 71.58 | 71.63 | 70.26 | 70.42 |
| valve | hmean | 76.16 | 76.15 | 75.17 | 75.33 |
