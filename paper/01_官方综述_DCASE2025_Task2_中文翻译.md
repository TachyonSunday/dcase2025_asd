# 声学场景与事件检测与分类 2025
> 2025年10月30-31日，西班牙巴塞罗那

# DCASE 2025 挑战赛任务 2 描述与讨论：面向机器状态监测的首次推理无监督异常声音检测

**Tomoya Nishida¹, Noboru Harada², Daisuke Niizumi², Davide Albertini³, Roberto Sannino³, Simone Pradolini³, Filippo Augusti³, Keisuke Imoto⁴, Kota Dohi¹, Harsh Purohit¹, Takashi Endo¹, and Yohei Kawaguchi¹**

¹ 株式会社日立制作所，日本，tomoya.nishida.ax@hitachi.com  
² NTT 株式会社，日本，harada.noboru@ntt.com  
³ 意法半导体（STMicroelectronics），瑞士  
⁴ 京都大学，日本，keisuke.imoto@ieee.org

## 摘要

本文介绍声学场景与事件检测与分类（DCASE）2025 挑战赛任务 2 的任务描述，该任务题为"面向机器状态监测的首次推理无监督异常声音检测（ASD）"。在 DCASE 2024 挑战赛任务 2 的基础上，本任务被构建为领域泛化框架下的一个首次推理问题。首次推理方法的主要目标是在无需针对特定机器类型进行超参数调优的情况下，促进 ASD 系统对新机器类型的快速部署。在 DCASE 2025 挑战赛任务 2 中，来自先前未见过的机器类型的声音已被采集并提供作为评估数据集。我们收到了来自 35 支队伍的 119 份提交，本文对这些提交进行了分析。分析表明，多种方法都可以具有竞争力，例如微调预训练模型、使用冻结的预训练模型以及从头训练小型模型，只要搭配适当的代价函数、异常分数标准化以及使用干净机器声音和噪声声音，均能取得良好效果。

**索引词**——异常检测，声学状态监测，领域偏移，首次推理问题，DCASE 挑战赛

## 1. 引言

异常声音检测（ASD）[1–7] 涉及判断目标机器发出的声音是正常的还是异常的。这种能力在实现机械故障的自动化检测中起着关键作用，这在第四次工业革命和 AI 驱动的工厂自动化时代至关重要。

开发 ASD 系统的关键挑战之一在于可用于训练的异常样本的稀缺性和有限多样性。为了解决这一问题，首个 ASD 任务在 DCASE 2020 挑战赛任务 2 [8] 中被引入，聚焦于"无监督 ASD（UASD）"，其目标是仅使用正常声音样本进行训练来检测未知的异常声音。在此基础上，2021 年和 2022 年的后续挑战赛 [9,10] 解决了领域偏移问题，以扩展 ASD 系统的更广泛应用。领域偏移指的是源域和目标域数据之间的差异，这些差异源于机器运行状态或环境噪声的变化。

DCASE 2023/2024 任务 2（"首次推理" UASD）[11,12] 针对的是一种真实场景：系统必须在无法获得同类型数据进行训练或超参数调优的情况下，对全新的机器类型进行异常检测。这反映了快速部署场景，其中收集多样化的训练或测试数据——特别是异常样本——是不可行的，因此基于测试数据的人工调优也是不现实的。相应地，评估数据包含开发集中不存在的机器类型，以强制施加这一约束。

DCASE 2025 挑战赛任务 2 保持了此前任务设定，即在领域泛化条件下的首次推理问题，并使用新录制的机器声音数据作为评估数据集。此外，还有一些修改：我们为每类机器提供了额外的补充数据，包括干净机器录音或噪声样本，这些数据可选择性地用于增强噪声环境下的 ASD 性能。同时，参赛者被要求提供其解决方案的计算复杂度。虽然该分数不用于官方排名，但它有助于阐明模型复杂度与性能之间的平衡——这是面向边缘设备的轻量级 ASD 应用的关键因素。在本文中，我们对本任务进行说明并讨论挑战赛结果。

## 2. 领域偏移条件下的首次推理无监督异常声音检测

考虑一个音频片段 **x**，其中包含机器产生的声音。ASD 任务的目标是通过使用参数为 θ 的异常分数计算器 *A* 计算异常分数 *A*<sub>θ</sub>(**x**)，将机器分类为正常或异常。*A* 的输入可以是音频片段 **x**，也可以带有或不带有额外信息，例如指示机器运行状态的标签。当 *A*<sub>θ</sub>(**x**) 超过预定义的阈值 ϕ 时，机器被判定为异常，即：

$$ \text{Decision} = \begin{cases} \text{Anomaly} & (A_\theta(\mathbf{x}) > \phi) \\ \text{Normal} & (\text{otherwise}) \end{cases} \tag{1} $$

本任务的主要困难在于仅使用正常声音来训练异常分数计算器（UASD）。DCASE 2020 挑战赛任务 2 [8] 正是为解决这一问题而设计的，此后所有任务都立足于这一 UASD 设定。

解决领域偏移问题对于 ASD 系统的实际部署同样至关重要。领域偏移是指训练阶段和测试阶段之间条件的变化，这些变化会改变观测声音数据的分布。这些变化可能源于运行速度、机器负载、加热温度、麦克风布置、环境噪声以及其他因素的差异。定义两个域：源域，表示具有充足训练数据的原始条件；目标域，表示仅有有限样本可用的另一种条件。今年的任务遵循 2022 至 2024 年任务 2 [10–12] 的设定，其中域信息在测试阶段被假定为未知，且来自两个域的异常都必须通过单一阈值来检测。在这种情况下，需要领域泛化来实现良好性能。

为了进一步推动 ASD 系统在真实场景中的快速开发，解决 ASD 在以下条件下的问题至关重要：(a) 对完全新颖的机器类型，(b) 仅使用一个 section 的训练数据，(c) 不依赖于测试数据的人工调优。这是因为在真实场景中，客户可能仅拥有一台新颖的机器，而收集用于人工调优的测试数据——特别是异常样本——可能是不可行的。这一问题设定被称为"首次推理问题"（first-shot problem），2023 年和 2024 年的任务 2 [11,12] 正是基于这一问题设定组织的。首次推理问题通过为数据集引入两个关键特征来实现：(i) 开发集和评估集由完全不同的机器类型组成，(ii) 数据集中的每种机器类型仅包含一个 section。需要注意的是，截至 2022 年任务 2，提供的数据集为每种机器类型包含多个 section，且开发集和评估集共享相同的机器类型。

DCASE 2025 挑战赛任务 2 在保持此前任务设定（即领域泛化条件下的首次推理问题）的同时，引入了若干修改。首先，我们提供了额外的补充数据，包括干净机器录音和噪声样本。这些资源可能反映实际场景——例如在工厂闲置时收集干净机器数据，或在机器不运行时采集噪声录音。参赛者可以自由地整合这些额外来源来提高其模型的准确率。其次，尽管大规模模型——如预训练网络和集成模型——在本任务中日益流行，但能够在边缘设备上运行的轻量级模型仍然是一个重要的研究领域。为确认这一点，参赛者被可选地要求报告其解决方案在乘加操作数（MACs）方面的计算复杂度。虽然这一指标不影响官方排名，但它为模型复杂度与性能之间的平衡提供了有价值的见解。

## 3. 任务设定

### 3.1. 数据集

本任务的数据集分为三类：开发数据集、附加训练数据集和评估数据集。开发数据集包含七种机器类型，而附加训练数据集和评估数据集包含九种机器类型，每种机器类型由一个 section 组成。机器类型指的是机器的类别，如风扇或齿轮箱，section 表示与每种机器类型相关的数据的子集或全部。

所有录音均为单通道，持续时间为 6 到 10 秒，采样率为 16 kHz。在实验室录制的机器声音与在工厂和郊区录制的环境噪声混合，以创建数据集中的每个样本。关于录音过程的更多细节，请参阅 ToyADMOS2 [13] 和 MIMII DG [14] 的论文。

开发数据集提供了七种机器类型（fan、gearbox、bearing、slide rail、valve、ToyCar、ToyTrain），每种机器类型有一个 section，包含完整的训练和测试数据集。每个 section 包含：(i) 来自源域的 990 个正常片段用于训练，(ii) 来自目标域的 10 个正常片段用于训练，(iii) 100 个补充声音数据片段，包含源域中干净的正常机器声音或仅含噪声的声音，(iv) 来自两个域的 100 个正常片段和 100 个异常片段用于测试。为辅助参赛者，域信息（源/目标）被包含在测试数据中。对于四种机器类型（fan、gearbox、valve 和 ToyCar），有关运行状态或环境条件的详细信息在文件名和属性 CSV 文件中提供。然而，对于其余三种机器类型，这些属性未被披露。

附加训练数据集提供了九种新颖的机器类型（AutoTrash、HomeCamera、ToyPet、ToyRCCar、BandSealer、Polisher、ScrewFeeder、CoffeeGrinder）。每个 section 包含：(i) 源域中的 990 个正常片段用于训练，(ii) 目标域中的 10 个正常片段用于训练，(iii) 100 个补充声音数据片段，包含源域中干净的正常机器声音或仅含噪声的声音。对于五种机器类型（HomeCamera、ToyRCCar、BandSealer 和 CoffeeGrinder），本数据集提供了属性信息。对于其他四种机器类型，属性被隐藏。评估数据集提供与附加训练数据集相对应的测试片段，即与附加训练数据集相同的机器类型的数据。每个 section 包含 200 个测试片段，这些片段均不带有状态标签（即正常或异常）、域信息或属性信息。参赛者需要仅为每种新机器类型使用单个 section 来训练模型。

### 3.2. 评估指标

为评估整体检测性能，我们采用受试者工作特征曲线下面积（AUC）。此外，我们使用部分 AUC（pAUC）来评估低误报率范围 [0, p] 内的性能，其中我们设置 p = 0.1。为在领域泛化设定下评估每个系统，我们按如下方式计算每个域的 AUC 和每个 section 的 pAUC：

$$ \text{AUC}_{m,n,d} = \frac{1}{N^{-}_{d} N^{+}_{n}} \sum_{i=1}^{N^{-}_{d}} \sum_{j=1}^{N^{+}_{n}} \mathcal{H}(A_\theta(\mathbf{x}^{+}_{j}) - A_\theta(\mathbf{x}^{-}_{i})) \tag{2} $$

$$ \text{pAUC}_{m,n} = \frac{1}{\lfloor pN^{-}_{n} \rfloor N^{+}_{n}} \sum_{i=1}^{\lfloor pN^{-}_{n} \rfloor N^{+}_{n}} \sum_{j=1}^{N^{+}_{n}} \mathcal{H}(A_\theta(\mathbf{x}^{+}_{j}) - A_\theta(\mathbf{x}^{-}_{i})) \tag{3} $$

其中 *m* 和 *n* 分别表示机器类型和 section 的索引，d ∈ {source, target} 表示域，⌊·⌋ 为取整函数，H(*y*) 在 *y* > 0 时返回 1，否则返回 0。此处，{**x**<sup>-</sup><sub>i</sub>}<sup>*N*<sup>-</sup><sub>d</sub></sup><sub>i=1</sub> 是机器类型 *m* 的 section *n* 中域 *d* 的正常测试片段，{**x**<sup>+</sup><sub>j</sub>}<sup>*N*<sup>+</sup><sub>n</sub></sup><sub>j=1</sub> 是机器类型 *m* 的 section *n* 中的所有异常测试片段。*N*<sup>-</sup><sub>d</sub>、*N*<sup>-</sup><sub>n</sub>、*N*<sup>+</sup><sub>n</sub> 分别表示域 *d* 中的正常测试片段数、section *n* 中的正常测试片段数以及 section *n* 中的异常测试片段数。

官方分数 Ω 由所有机器类型和 section 上的 AUC 和 pAUC 分数的调和平均值给出：

$$ \Omega = h\left(\left\{ \text{AUC}_{m,n,d}, \text{pAUC}_{m,n} \mid m \in \mathcal{M}, n \in \mathcal{S}(m), d \in \{\text{source}, \text{target}\} \right\}\right) \tag{4} $$

其中 *h*{·} 表示调和平均，**M** 是给定的机器类型集合，**S**(m) 表示机器类型 *m* 的 section 集合。具体而言，对于 2024-2025 年的数据集，**S**(m) = {00}。此外，虽然不纳入官方排名，但参赛者被可选地要求提供其模型在 MAC 操作数方面的计算复杂度信息。建议使用文献 [15] 中的开源实现来计算该指标。

### 3.3. 基线系统与结果

任务组织者提供了一个使用自编码器（Autoencoders, AEs）的基线系统，具有两种运行模式，与 2023 年任务 2 的基线相同。虽然两种模式在训练时都使用自编码器，但它们在异常分数计算上有所不同。本文介绍该系统及其检测性能；详细信息可参见文献 [16]。

#### 3.3.1. 自编码器训练

对于两种运行模式，AE 均使用训练声音片段的对数梅尔频谱图 **X** = [**X**<sub>1</sub>, ..., **X**<sub>T</sub>] 进行训练，其中 **X**<sub>t</sub> ∈ R<sup>F</sup>，t = 1, ..., *T* 表示第 t 帧的帧级特征向量，*F* = 128 为梅尔滤波器数量，*T* 为时间帧数。对于输入，将 *P* = 5 个连续帧拼接为 ψ<sub>t</sub> = [**X**<sup>T</sup><sub>t</sub>, ..., **X**<sup>T</sup><sub>t+P−1</sub>]<sup>T</sup> ∈ R<sup>D</sup>，其中 D = P × F = 640。模型参数通过最小化输入 ψ<sub>t</sub> 与重构输出 r<sub>θ</sub>(ψ<sub>t</sub>) 之间针对训练数据中所有输入的均方误差（MSE）来训练。

#### 3.3.2. 简单自编码器模式

该模式使用给定声音片段所有特征的均值 MSE 作为其异常分数，即：

$$ A_\theta(\mathbf{X}) = \frac{1}{DK} \sum_{k=1}^{K} \|\psi_k - r_\theta(\psi_k)\|^2_2 \tag{5} $$

其中 *K* = *T* − *P* + 1，∥·∥<sub>2</sub> 表示 ℓ<sub>2</sub> 范数。

#### 3.3.3. 选择性马氏距离模式

在该模式中，系统输入与重构特征之间的马氏距离被用于计算异常分数。异常分数定义为：

$$ A_\theta(\mathbf{X}) = \frac{1}{DK} \sum_{k=1}^{K} \min \left\{ D_s(\psi_k, r_\theta(\psi_k)), D_t(\psi_k, r_\theta(\psi_k)) \right\} \tag{6} $$

$$ D_s(\cdot) = \text{Mahalanobis}(\psi_k, r_\theta(\psi_k), \Sigma^{-1}_s) \tag{7} $$

$$ D_t(\cdot) = \text{Mahalanobis}(\psi_k, r_\theta(\psi_k), \Sigma^{-1}_t) \tag{8} $$

其中 Σ<sup>-1</sup><sub>s</sub> 和 Σ<sup>-1</sup><sub>t</sub> 分别是每种机器类型的源域和目标域数据中 r<sub>θ</sub>(ψ<sub>k</sub>) − ψ<sub>k</sub> 的协方差矩阵。

#### 3.3.4. 结果

表 1 展示了两种基线系统在开发数据集上的 AUC 和 pAUC 结果，平均值和标准差由五次独立试验计算得出。

**表 1：开发数据集的基线结果**

| 机器类型 | 模式 | AUC [%]<br>源域 | AUC [%]<br>目标域 | pAUC [%] |
|:---|:---|:---:|:---:|:---:|
| ToyCar | MSE | 71.05 ± 0.50 | 53.32 ± 0.56 | 49.79 ± 0.49 |
| ToyCar | MAHALA | 73.17 ± 0.39 | 50.91 ± 0.85 | 49.05 ± 0.05 |
| ToyTrain | MSE | 61.76 ± 0.74 | 56.46 ± 0.47 | 50.19 ± 0.25 |
| ToyTrain | MAHALA | 50.87 ± 2.88 | 46.15 ± 1.77 | 48.32 ± 0.05 |
| bearing | MSE | 66.53 ± 2.63 | 53.15 ± 1.99 | 61.12 ± 0.59 |
| bearing | MAHALA | 63.63 ± 1.15 | 59.03 ± 1.79 | 61.86 ± 0.36 |
| fan | MSE | 70.96 ± 0.94 | 38.75 ± 0.74 | 49.46 ± 0.53 |
| fan | MAHALA | 77.99 ± 0.23 | 38.56 ± 0.58 | 50.82 ± 0.06 |
| gearbox | MSE | 64.80 ± 1.48 | 50.49 ± 1.22 | 52.49 ± 0.37 |
| gearbox | MAHALA | 73.26 ± 0.78 | 51.61 ± 0.52 | 55.07 ± 0.47 |
| slider | MSE | 70.10 ± 1.01 | 48.77 ± 1.07 | 52.32 ± 0.36 |
| slider | MAHALA | 73.79 ± 1.95 | 50.27 ± 1.15 | 53.61 ± 0.26 |
| valve | MSE | 63.53 ± 2.90 | 67.18 ± 1.75 | 57.35 ± 1.96 |
| valve | MAHALA | 56.22 ± 2.22 | 61.00 ± 2.98 | 52.53 ± 1.32 |

## 4. 挑战赛结果

### 4.1. 总体结果

我们收到了来自 35 支队伍的 119 份提交。20 支队伍超越了两种基线，相比去年的任务（27 支队伍中的 11 支）略有增加。分别查看每个域的结果，有 6 支队伍在源域 AUC 上超越了基线，而有 25 支队伍在目标域 AUC 上超越了基线。4 支队伍在两个域的 AUC 上均高于基线。这显示了同时提高源域和目标域性能的难度。图 1 展示了前 10 名队伍的 AUC 值。在源域中，每支队伍是否能击败基线高度依赖于机器类型，许多队伍在平均值上难以超越基线。具体而言，属性信息可用的机器类型往往表现不佳，尽管这一因素是否确实相关尚不清楚。相比之下，所有前 10 名队伍在目标域的调和平均 AUC 上均超越了基线。

图 2 比较了前 20 名队伍在开发数据集和评估数据集上的 AUC 值。可以看出，在开发数据集上取得高 AUC 值并不意味着在评估数据集上也能取得高 AUC。这是自 2023 年开始的首次推理问题设定中的一个典型趋势，并显示了在没有测试数据的情况下找到对机器类型具有鲁棒性的方法有多么困难。因此，构建一个对未知机器类型有效的 ASD 系统仍然是一个困难且重要的挑战。在图 3 中，我们比较了前 20 名队伍在评估数据集源域和目标域中的调和平均 AUC 值。该图显示了源域 AUC 和目标域 AUC 之间的负相关关系。虽然前 3 名队伍取得了非常接近的官方分数（差异在 1.0% 以内），但它们在源域和目标域 AUC 之间的平衡各有不同。在源域和目标域之间取得良好平衡的性能可能是取得高排名的重要因素。

### 4.2. 模型规模趋势

我们分析了提交系统的计算复杂度趋势。21 支队伍报告了 MACs。图 4 绘制了每份提交的 MAC 数量与其官方分数的关系；如果一支队伍提交了多个具有相同 MAC 值的系统，则仅保留排名最高的系统。

该图显示各提交的 MAC 数量分布范围广泛。它同样突出表明，更大的计算预算并不必然带来更高的分数。值得注意的是，来自不同队伍的两份提交 [17,18] 在使用更少 MACs 的情况下取得了超越基线的分数。这证明了针对首次推理 UASD 的计算高效解决方案的可行性，这也可能成为未来的研究方向之一。

### 4.3. 排名靠前队伍中观察到的新方法

#### a. 预训练模型的使用

今年，延续前几年的趋势，许多参赛者在其异常检测流程中采用了预训练模型。其中许多队伍通过属性或域分类辅助任务对这些模型进行了微调，例如排名第 1、第 4、第 5 的队伍 [19–21]。另一方面，有趣的是，一些排名靠前的队伍使用冻结的预训练网络取得了强劲的性能，通过利用中间层特征并配合异常分数标准化 [22]。第 2 名 [23] 和第 8 名 [24] 的队伍在提交中仅使用了冻结模型，而第 4 名队伍 [20] 则将冻结网络集成到了其微调网络中。然而，通过基于分类的任务从头训练轻量级模型的队伍也取得了高排名，包括第 3 名队伍 [17,18]，这表明预训练模型并非获得有竞争力性能的绝对前提。总体而言，今年多种方法都具有竞争力，每种方法可能仍有进一步研究空间。

#### b. 补充数据的使用

参赛者尝试利用新发布的补充干净机器录音和噪声录音，以两种不同的方式加以应用。第一种方式是将其用于数据增强。第 1 名 [19] 和其他几支前十队伍 [24–26] 将补充片段作为辅助分类器中的额外类别注入，将噪声信号与训练样本混合，或在对比学习 [18] 中利用它们来丰富特征空间多样性。另一方面，第 3 名和第 4 名队伍 [17,20] 使用干净/噪声数据来构建增强模块，从带噪训练数据中提取或去噪目标机器声音，并将这些提取到的信号提供给其主异常检测网络。

## 5. 结论

我们概述了 DCASE 2025 挑战赛任务 2。本任务的目标是开发对新颖机器类型有效的 ASD 系统，每种机器类型仅有一个 section，同时提供了干净机器声音或仅含噪声的声音等补充数据。我们讨论了挑战赛中观察到的若干新方法，例如预训练模型是如何被使用（或未被使用）的，以及新提供的补充数据的使用方式。虽然我们无法讨论所有新方法，但我们希望所有技术报告都能为异常声音检测领域的进步做出贡献。

## 6. 参考文献

[1] Y. Koizumi, S. Saito, H. Uematsu, and N. Harada, "Optimizing acoustic feature extractor for anomalous sound detection based on Neyman-Pearson lemma," in *Proc. EUSIPCO*, 2017, pp. 698–702.

[2] Y. Kawaguchi and T. Endo, "How can we detect anomalies from subsampled audio signals?" in *Proc. IEEE MLSP*, 2017.

[3] Y. Koizumi, S. Saito, H. Uematsu, Y. Kawachi, and N. Harada, "Unsupervised detection of anomalous sound based on deep learning and the Neyman-Pearson lemma," *IEEE/ACM TASLP*, vol. 27, no. 1, pp. 212–224, Jan. 2019.

[4] Y. Kawaguchi, R. Tanabe, T. Endo, K. Ichige, and K. Hamada, "Anomaly detection based on an ensemble of dereverberation and anomalous sound extraction," in *Proc. IEEE ICASSP*, 2019, pp. 865–869.

[5] Y. Koizumi, S. Saito, M. Yamaguchi, S. Murata, and N. Harada, "Batch uniformization for minimizing maximum anomaly score of DNN-based anomaly detection in sounds," in *Proc. IEEE WASPAA*, 2019, pp. 6–10.

[6] K. Suefusa, T. Nishida, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, "Anomalous sound detection based on interpolation deep neural network," in *Proc. IEEE ICASSP*, 2020, pp. 271–275.

[7] H. Purohit, R. Tanabe, T. Endo, K. Suefusa, Y. Nikaido, and Y. Kawaguchi, "Deep autoencoding GMM-based unsupervised anomaly detection in acoustic signals and its hyper-parameter optimization," in *Proc. DCASE Workshop*, 2020, pp. 175–179.

[8] Y. Koizumi, Y. Kawaguchi, K. Imoto, T. Nakamura, Y. Nikaido, R. Tanabe, H. Purohit, K. Suefusa, T. Endo, M. Yasuda, and N. Harada, "Description and discussion on DCASE2020 challenge task2: Unsupervised anomalous sound detection for machine condition monitoring," in *Proc. DCASE Workshop*, 2020, pp. 81–85.

[9] Y. Kawaguchi, K. Imoto, Y. Koizumi, N. Harada, D. Niizumi, K. Dohi, R. Tanabe, H. Purohit, and T. Endo, "Description and discussion on DCASE 2021 challenge task 2: Unsupervised anomalous detection for machine condition monitoring under domain shifted conditions," in *Proc. DCASE Workshop*, 2021, pp. 186–190.

[10] K. Dohi, K. Imoto, N. Harada, D. Niizumi, Y. Koizumi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, and Y. Kawaguchi, "Description and discussion on DCASE 2022 challenge task 2: Unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques," in *Proc. DCASE Workshop*, 2022, pp. 26–30.

[11] K. Dohi, K. Imoto, N. Harada, D. Niizumi, Y. Koizumi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2023 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring," in *Proc. DCASE Workshop*, 2023, pp. 31–35.

[12] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, "Description and discussion on DCASE 2024 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring," in *Proc. DCASE Workshop*, 2024, pp. 111–115.

[13] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, "ToyADMOS2: Another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions," in *Proc. DCASE Workshop*, 2021, pp. 1–5.

[14] K. Dohi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, M. Yamamoto, Y. Nikaido, and Y. Kawaguchi, "MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in *Proc. DCASE Workshop*, 2022.

[15] L. Zhu, "Thop: Pytorch-opcounter," https://github.com/Lyken17/pytorch-OpCounter, 2019.

[16] N. Harada, N. Daisuke, T. Daiki, O. Yasunori, and Y. Masahiro, "First-shot anomaly detection for machine condition monitoring: A domain generalization baseline," in *Proc. EUSIPCO*, 2023, pp. 191–195.

[17] J. Yang, "A two stage fusion anomaly detection approach for Task2," DCASE2025 Challenge, Tech. Rep., June 2025.

[18] Q. Zhou and S. Wu, "Machine anomalous sound detection combining convolutional auto-encoder and contrastive learning," DCASE2025 Challenge, Tech. Rep., June 2025.

[19] L. Wang, "Pre-trained model enhanced anomalous sound detection system for DCASE2025 Task2," DCASE2025 Challenge, Tech. Rep., June 2025.

[20] F. Takuya, I. Kuroyanagi, and T. Toda, "The NU systems for DCASE 2025 Challenge Task 2," DCASE2025 Challenge, Tech. Rep., June 2025.

[21] A. Jiang, W. Liang, S. Feng, Y. Qiu, Y. Zhao, J. Li, P. Fan, W.-Q. Zhang, C. Lu, X. Chen, Y. Qian, and J. Liu, "THUEE system for DCASE 2025 anomalous sound detection challenge," DCASE2025 Challenge, Tech. Rep., June 2025.

[22] K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, and J. Le Roux, "Keeping the balance: Anomaly score calculation for domain generalization," in *Proc. IEEE ICASSP*. IEEE, 2025, pp. 1–5.

[23] P. Saengthong and T. Shinozaki, "GenRep for first-shot unsupervised anomalous sound detection of DCASE2025 Challenge," DCASE2025 Challenge, Tech. Rep., June 2025.

[24] T. Shiraga, K. Ozeki, T. Masuzaki, N. Tanaka, and T. Kuriyama, "Anomalous sound detection method using contrastive learning," DCASE2025 Challenge, Tech. Rep., June 2025.

[25] X. Zheng, A. Jiang, B. Han, S. Zhang, W.-Q. Zhang, X. Chen, C. Lu, P. Fan, J. Liu, and Y. Qian, "SJTU-AITHU system for DCASE 2025 anomalous sound detection challenge," DCASE2025 Challenge, Tech. Rep., June 2025.

[26] S. Zhang, F. Xiao, S. Fan, Q. Zhu, W. Wang, and J. Guan, "Anomalous sound detection using pre-trained model with statistical feature difference representation," DCASE2025 Challenge, Tech. Rep., June 2025.
