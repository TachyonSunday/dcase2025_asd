# DCASE 2025 挑战赛声学场景与事件检测与分类

## SJTU-AITHU 系统用于 DCASE 2025 异常声音检测挑战赛

### 技术报告

**作者：** Xinhu Zheng¹, Anbai Jiang², Bing Han¹, Shuwei Zhang³, Wei-Qiang Zhang³, Xie Chen¹, Cheng Lu⁴, Pingyi Fan², Jia Liu²˒³, Yanmin Qian¹

¹ 上海交通大学，上海，中国  
² 清华大学，北京，中国  
³ 华控智加科技有限公司，北京，中国  
⁴ 华北电力大学，北京，中国  

Email: zhengxh24@sjtu.edu.cn, jab22@mails.tsinghua.edu.cn

---

## 摘要

本报告介绍了我们针对 DCASE 2025 任务 2：面向机器状态监测的首次无监督异常声音检测（First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring）提出的解决方案。在该领域中，预训练模型已展现出巨大的潜力，尤其是在处理领域偏移（domain shifts）方面。我们基于 BEATs 和 EAT 系列模型构建系统，并探索了多种训练策略以提升性能。采用子中心损失（Sub-center loss）和噪声感知训练（noise-aware training）来改善系统性能。通过融合多种模型与方法，我们在开发数据集上取得了 69.12% 的调和平均数（hmean）。

**关键词：** DCASE 挑战赛，异常检测，声音，预训练模型，噪声感知训练

---

## 1. 引言

在工业自动化领域，检测异常声音的能力对于确保运行可靠性和预防潜在故障依然至关重要。

DCASE 2025 挑战赛任务 2 [1, 2, 3, 4]，即面向机器状态监测的首次无监督异常声音检测，继续聚焦于识别来自特定机器类型声音中的异常。今年的挑战赛引入了额外的复杂性，将数据集扩展至同时包含干净数据与纯噪声样本，这进一步考验了算法在区分真实异常与正常运行噪声方面的鲁棒性。

该任务的复杂性在于准确区分正常运行噪声与真实异常，要求算法能够从多样的声学模式中学习。在实际生产环境中，设备类型的多样性、复杂的周围环境以及声音数据采集的挑战，使得开发能够在不同设备和环境下准确识别并分类异常声音的系统变得十分困难。主要挑战可概括如下：

*   **训练数据稀缺。** 尽管今年的数据集包含了更多的样本，但真实工业场景仍面临模型训练数据有限的基本问题。模型仍需克服在缺乏异常样本的情况下从相对稀缺的正常运行示例中学习的挑战。
*   **领域偏移。** 工业生产环境的复杂性、多样的背景噪声以及录音设备的差异，持续导致音频数据出现分布差异。今年数据集中新增的干净数据可能有助于缓解但无法消除这些领域偏移问题。
*   **训练标签不完整。** 数据采集过程仍然面临着并非所有机器类型都有可用的属性标签这一问题。尽管存在这些限制，模型必须保持良好的泛化性能。

延续我们之前的工作 [5, 6, 7, 8]，我们继续利用预训练模型来提供在不同机器之间所需的泛化能力。今年，由于比赛规则的变化，允许使用从 DCASE 2020 到 DCASE 2025 的所有可用数据进行训练，我们发现某些方法，如 LoRA [9] 微调和 SMOTE [10]，在数据规模增大时不再有效。因此，我们选择不采用这些方法。相反，我们引入了噪声感知训练和子中心损失 [11] 来增强模型的鲁棒性并解决标签缺失的问题。所有提交的系统均为集成系统，将多个单一模型的得分进行结合，我们最好的系统在开发集上取得了 69.12% 的调和平均数。

本文的结构安排如下。第 2 节介绍预训练模型与附加策略。第 3 节概述所有提交的系统。第 4 节展示检测结果。

---

## 2. 方法

### 2.1. BEATs

BEATs [12]，全称为来自音频 Transformer 的双向编码器表示（Bidirectional Encoder representation from Audio Transformers），相比其他预训练音频模型已展现出卓越的性能。该自监督学习框架在其声学分词器（acoustic tokenizer）与音频自监督学习（SSL）模型组件之间采用了一种迭代优化过程。该架构生成了丰富的语义离散标签，能够有效捕获音频表示，这对于我们的分类目标和异常检测特别有益。我们使用 BEATs-iter3 版本，该版本在 AudioSet [13] 上预训练，包含 90M 参数。

对于模型适配，我们对 DCASE 2020 至 DCASE 2025 的所有机器类型执行基于属性的微调。输入处理流水线将音频片段标准化为 10 秒时长，随后使用 25 ms 帧长、10 ms 帧移和 128 个 mel 频带转换为对数梅尔（log-mel）频谱图。为了增强鲁棒性，我们应用了 SpecAug [14]，在时间维度和频率维度上的最大掩蔽长度均为 80。

模型架构融合了来自 ECAPA-TDNN [15] 的一个注意力统计池化层（attentive statistics pooling layer），用于将帧级嵌入聚合为句子级嵌入。其后附加了两个全连接层来预测 logits。我们的分类方法动态适应标签的可用性：对于具有强属性标注的样本（DCASE 2022-2025 数据），我们使用这些属性标签作为分类目标；当仅有弱属性标签或无属性标签存在时（DCASE 2020-2021 数据），我们回退到使用 section 标签作为分类标准。这种混合策略最大限度地利用了所有可用标签。训练采用 ArcFace [16] 损失，使用 AdamW [17] 优化器训练 30,000 步，梯度累积步数为 8，预热步数为 360，批次大小为 32。

异常检测系统通过对嵌入向量之间的余弦距离进行计算，采用基于相似度得分的 1-最近邻方法，其中每个测试实例的异常得分为其与任意训练样本之间的最小距离。

### 2.2. 其他自监督学习（SSL）模型

除了 BEATs 之外，我们还研究了 EAT [18] 以及一个自研自监督学习模型的使用。该自研模型采用短时傅里叶变换（STFT）作为输入，并在一个教师-学生框架中对子带进行建模。该模型在来自 Audioset [19]、Freesound [1]、MTG-Jamendo [20] 和 Music4all [21] 的 17k 小时音频上训练。该自研模型将在后续研究论文中详细介绍。两个模型均在所有六个 DCASE 数据集上进行了微调。在检测过程中，我们提取 [CLS] 嵌入，并执行与 BEATs 相同的异常检测流程。

[1] https://freesound.org/

### 2.3. 噪声感知训练

为了提升模型对环境噪声的鲁棒性，我们利用 DCASE 2025 数据集中提供的纯噪声样本实现了噪声感知训练。在训练过程中，每个音频样本有 50% 的概率与随机选择的噪声在不同信噪比（SNR）下混合。具体来说，对于每个可能被损坏的样本，我们：

*   从提供的噪声集合中随机选择一个噪声样本
*   从 {5, 10, 15, 20} dB 中随机选择一个信噪比水平
*   在选定的信噪比水平下将原始音频与噪声混合
*   无论是否添加噪声，均保持原始标签不变

该方法有两个关键目的：(1) 对模型进行正则化，防止其过拟合原始训练数据；(2) 让系统更好地应对机器声音常与环境噪声共存的现实世界条件。所选的 SNR 范围覆盖了从具有挑战性（5dB）到较为适中（20dB）的噪声条件，代表了真实的工业场景。值得注意的是，我们仅使用 DCASE 2025 数据集官方提供的噪声样本，以确保与评估环境的一致性。

### 2.4. 子中心损失

我们采用子中心 ArcFace 损失（Sub-Center ArcFace loss）[11] 来处理标签歧义性并提高特征的判别力。该损失函数通过为选定类别引入多个子中心来增强传统的角度间隔方法：

$$ L = -\log \frac{e^{s(\cos(\theta_y + m))}}{e^{s(\cos(\theta_y + m))} + \sum_{j \neq y} e^{s \cos \theta_j}} \tag{1} $$

其中，$\theta_y$ 表示嵌入向量与其目标类别 $y$ 的最近子中心之间的角度，$s = 30$ 为缩放因子，$m = 0.2$ 为角度间隔。关键实现细节包括：

*   子中心（$k = 16$）仅对以下情况激活：(1) 所有 DCASE 2020-2021 机器类型，以及 (2) 无属性标签的 DCASE 2024-2025 机器类型
*   其他情况使用标准 ArcFace（单中心）
*   在训练过程中，每个样本自动关联其最近的子中心
*   间隔惩罚有助于创建更具判别力的特征空间

这种子中心的选择性应用提供了两个好处：(1) 通过引入额外的表示，为标注不完善或属性缺失的样本提供鲁棒性，同时 (2) 为标注良好的类别保持更简单的判别边界。

---

## 3. 提交的系统

我们提交的四个系统是由 13 个系统组成的集成系统，其中包括一个基准系统（结合了 BEATs、EAT 和自研模型，仅应用子中心损失），以及 12 个基于 BEATs 的系统，这些系统采用了不同的方法组合和超参数。对于每个系统，我们选取训练过程中表现最好的前 3 个检查点（checkpoints）作为内部集成。

*   **系统 1** 实现了所有基于 BEATs 的系统的融合。
*   **系统 2** 展示了所有 13 个系统的融合。
*   **系统 3** 将基准系统与两个在 DCASE 2025 评估集上显示出特别稳健得分分布的系统相结合。
*   **系统 4** 将基准系统与两个基于定量指标表现最优的系统合并。

模型通过线性组合不同模型的异常得分来进行集成，其中系数的获取方式为网格搜索或贝叶斯优化。系统 1 中的贝叶斯方法自动确定权重，而系统 3-4 使用网格搜索来寻找能平衡性能和鲁棒性的系数。这种多策略集成框架既提供了全面的模型平均，又提供了针对性的性能优化。

---

## 4. 实验结果

检测性能使用 DCASE 2025 挑战赛指定的标准指标进行评估：接收者操作特征（ROC）曲线下面积（AUC）、在假阳性率范围为 0-0.1 内的部分 AUC（pAUC），以及它们的调和平均数。对于每种机器类型，我们计算源域与目标域的 AUC 得分以及 pAUC 值，然后根据官方评估基线通过调和平均进行组合。

**表 1：四个提交系统在开发集上的结果**

| Machine | Metric | 系统 1 | 系统 2 | 系统 3 | 系统 4 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **bearing** | AUC s | 68.26 | 66.76 | 65.74 | 66.22 |
| | AUC t | 67.96 | 68.44 | 68.06 | 68.42 |
| | pAUC | 60.68 | 58.79 | 58.00 | 58.11 |
| | hmean | 65.44 | 64.38 | 63.63 | 63.93 |
| **fan** | AUC s | 61.54 | 61.42 | 61.24 | 61.36 |
| | AUC t | 62.40 | 62.50 | 62.56 | 62.06 |
| | pAUC | 55.42 | 55.89 | 55.42 | 55.79 |
| | hmean | 59.62 | 59.79 | 59.57 | 59.60 |
| **gearbox** | AUC s | 80.96 | 82.92 | 83.10 | 82.90 |
| | AUC t | 75.20 | 82.22 | 82.94 | 82.50 |
| | pAUC | 68.79 | 67.53 | 67.79 | 67.53 |
| | hmean | 74.65 | 76.86 | 77.24 | 76.94 |
| **slider** | AUC s | 88.40 | 91.98 | 91.92 | 92.12 |
| | AUC t | 76.22 | 77.82 | 77.30 | 77.92 |
| | pAUC | 60.63 | 59.95 | 59.68 | 60.05 |
| | hmean | 73.30 | 74.25 | 73.94 | 74.36 |
| **ToyCar** | AUC s | 67.82 | 69.04 | 69.12 | 69.10 |
| | AUC t | 63.22 | 63.26 | 62.76 | 63.20 |
| | pAUC | 51.79 | 53.16 | 52.84 | 53.11 |
| | hmean | 60.15 | 61.10 | 60.82 | 61.07 |
| **ToyTrain** | AUC s | 75.76 | 76.92 | 76.50 | 77.22 |
| | AUC t | 74.70 | 70.34 | 69.20 | 70.26 |
| | pAUC | 62.11 | 57.63 | 56.53 | 57.47 |
| | hmean | 70.28 | 67.31 | 66.35 | 67.29 |
| **valve** | AUC s | 88.16 | 87.58 | 87.32 | 87.44 |
| | AUC t | 93.92 | 94.06 | 93.60 | 94.04 |
| | pAUC | 78.11 | 84.11 | 84.63 | 84.11 |
| | hmean | 86.22 | 88.39 | 88.36 | 88.34 |
| **hmean** | AUC s | 74.59 | 75.19 | 74.91 | 75.13 |
| | AUC t | 72.16 | 72.70 | 72.36 | 72.63 |
| | pAUC | 61.53 | 61.17 | 60.75 | 61.02 |
| | hmean | 68.94 | **69.12** | 68.76 | 69.02 |

*注：AUC s 和 AUC t 分别为源域和目标域的 AUC。*

表 1 展示了四个提交系统的详细性能。为每个系统和每种机器类型计算了 AUC s、AUC t、pAUC 以及调和平均数。在 DCASE 2025 开发数据集上的最佳结果由系统 2 取得，调和平均数为 69.12%。

---

## 5. 结论

本文介绍了 SJTU-AITHU 系统在 DCASE 2025 任务 2“首次无监督异常声音检测”上的工作。我们的方法利用 BEATs 和 EAT 预训练模型，并通过噪声感知训练增强了对环境干扰的鲁棒性，同时采用子中心损失来解决不同机器类型和数据集中的标签缺失和不一致问题。最终，所提出的系统在开发集上取得了 69.12% 的最佳调和平均数。

---

## 6. 参考文献

[1] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, and M. Yasuda, "First-shot anomaly detection for machine condition monitoring: A domain generalization baseline," *Proceedings of 31st European Signal Processing Conference (EUSIPCO)*, pp. 191–195, 2023.

[2] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2025 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring," *In arXiv e-prints: 2506.10097*, 2025.

[3] K. Dohi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, Y. Nikaido, and Y. Kawaguchi, "MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in *Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)*, Nancy, France, November 2022.

[4] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, "ToyADMOS2: Another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions," in *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE)*, Barcelona, Spain, November 2021, pp. 1–5.

[5] Z. Lv, A. Jiang, B. Han, Y. Liang, Y. Qian, X. Chen, J. Liu, and P. Fan, "AITHU system for first-shot unsupervised anomalous sound detection," *DCASE2024 Challenge, Tech. Rep.*, June 2024.

[6] A. Jiang, X. Zheng, Y. Qiu, W. Zhang, B. Chen, P. Fan, W.-Q. Zhang, C. Lu, and J. Liu, "THUEE system for first-shot unsupervised anomalous sound detection," *DCASE2024 Challenge, Tech. Rep.*, June 2024.

[7] A. Jiang, B. Han, Z. Lv, Y. Deng, W.-Q. Zhang, X. Chen, Y. Qian, J. Liu, and P. Fan, "AnoPatch: Towards better consistency in machine anomalous sound detection," in *Interspeech 2024*, 2024, pp. 107–111.

[8] X. Zheng, A. Jiang, B. Han, Y. Qian, P. Fan, J. Liu, and W.-Q. Zhang, "Improving anomalous sound detection via low-rank adaptation fine-tuning of pre-trained audio models," in *2024 IEEE Spoken Language Technology Workshop (SLT)*. IEEE, 2024, pp. 969–974.

[9] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in *International Conference on Learning Representations*, 2022.

[10] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," *Journal of artificial intelligence research*, vol. 16, pp. 321–357, 2002.

[11] A. Jiang, X. Zheng, B. Han, Y. Qiu, P. Fan, W.-Q. Zhang, C. Lu, and J. Liu, "Adaptive prototype learning for anomalous sound detection with partially known attributes," in *ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2025, pp. 1–5.

[12] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, and F. Wei, "BEATs: Audio pre-training with acoustic tokenizers," *arXiv preprint arXiv:2212.09058*, 2022.

[13] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio set: An ontology and human-labeled dataset for audio events," in *2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)*. IEEE, 2017, pp. 776–780.

[14] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A simple data augmentation method for automatic speech recognition," *arXiv preprint arXiv:1904.08779*, 2019.

[15] B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification," *arXiv preprint arXiv:2005.07143*, 2020.

[16] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, pp. 4690–4699.

[17] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *International Conference on Learning Representations*, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[18] W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, "EAT: Self-supervised pre-training with efficient audio transformer," *arXiv preprint arXiv:2401.03497*, 2024.

[19] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, "Audio set: An ontology and human-labeled dataset for audio events," in *2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)*. IEEE, 2017, pp. 776–780.

[20] D. Bogdanov, M. Won, P. Tovstogan, A. Porter, and X. Serra, "The MTG-Jamendo dataset for automatic music tagging," in *Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019)*, Long Beach, CA, United States, 2019. [Online]. Available: http://hdl.handle.net/10230/42015

[21] I. A. P. Santana, F. Pinhelli, J. Donini, L. Catharin, R. B. Mangolin, V. D. Feltrim, M. A. Domingues, et al., "Music4all: A new music database and its applications," in *2020 International Conference on Systems, Signals and Image Processing (IWSSIP)*. IEEE, 2020, pp. 399–404.
