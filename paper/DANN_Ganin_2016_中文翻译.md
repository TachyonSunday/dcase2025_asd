# 神经网络的域对抗训练

**期刊：** Journal of Machine Learning Research 17 (2016) 1-35  
**投稿日期：** 2015年5月；**发表日期：** 2016年4月

**作者：**
- Yaroslav Ganin（ganin@skoltech.ru）
- Evgeniya Ustinova（evgeniya.ustinova@skoltech.ru）  
  斯科尔科沃科学技术研究所（Skoltech），斯科尔科沃，莫斯科地区，俄罗斯
- Hana Ajakan（hana.ajakan.1@ulaval.ca）
- Pascal Germain（Pascal.Germain@ift.ulaval.ca）
- Mario Marchand（Mario.Marchand@ift.ulaval.ca）  
  拉瓦尔大学，计算机科学与软件工程系，魁北克，加拿大，G1V 0A6
- Hugo Larochelle（hugo.larochelle@usherbrooke.ca）  
  舍布鲁克大学，计算机科学系，魁北克，加拿大，J1K 2R1
- Francois Laviolette（Francois.Laviolette@ift.ulaval.ca）
- Victor Lempitsky（lempitsky@skoltech.ru）  
  斯科尔科沃科学技术研究所（Skoltech），斯科尔科沃，莫斯科地区，俄罗斯

**编辑：** Urun Dogan, Marius Kloft, Francesco Orabona, and Tatiana Tommasi

## 摘要

我们提出一种新的域适应表示学习方法，用于训练和测试数据来自相似但不同分布的场景。我们的方法直接受到域适应理论的启发，该理论表明：为了实现有效的域迁移，预测必须基于无法区分训练（源）域和测试（目标）域的特征。

该方法在神经网络架构的背景下实现这一思想，利用源域的标注数据和目标域的未标注数据（不需要标注目标域数据）进行训练。随着训练的进行，该方法促进了具有以下两方面特征的出现：（i）对源域的主要学习任务具有判别性，以及（ii）对域之间的偏移不可区分。我们证明，这种适应行为可以通过在几乎任何前馈模型中添加少量标准层和一个新的梯度反转层来实现。由此得到的增强架构可以使用标准的反向传播和随机梯度下降进行训练，因此几乎不需要额外工作量即可在任何深度学习包中实现。

我们在两个不同的分类问题（文档情感分析和图像分类）上证明了该方法的成功，在标准基准上取得了最先进的域适应性能。我们还在行人重识别应用环境中的描述子学习任务上验证了该方法。

**关键词：** 域适应，神经网络，表示学习，深度学习，合成数据，图像分类，情感分析，行人重识别

(c) 2016 Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, et al.  
arXiv:1505.07818v4 [stat.ML] 2016年5月26日

---

## 1. 引言

为新机器学习任务生成标注数据的成本往往是应用机器学习方法的一个障碍。特别地，这是深度神经网络架构进一步发展的限制因素，而深度神经网络已经为各种机器学习任务和应用带来了令人瞩目的最先进进展。对于缺乏标注数据的问题，仍然可能获得足够大以训练大规模深度模型的训练集，但这些训练集与"测试时"实际遇到的数据之间存在数据分布偏移。一个重要的例子是在合成或半合成图像上训练图像分类器，这些图像可能大量存在且有完整标注，但不可避免地具有与真实图像不同的分布（Liebelt and Schmid, 2010; Stark et al., 2010; Vazquez et al., 2014; Sun and Saenko, 2014）。另一个例子是书面评论的情感分析，其中可能有一类产品（如电影）评论的标注数据，而需要对其他产品（如书籍）的评论进行分类。

在训练分布和测试分布存在偏移的情况下学习判别性分类器或其他预测器被称为域适应（DA）。已有方法在源（训练时）域和目标（测试时）域之间建立映射，使得为源域学习的分类器在与所学的域间映射组合后也可以应用于目标域。域适应方法的吸引力在于能够在目标域数据完全未标注（无监督域适应）或仅有少量标注样本（半监督域适应）的情况下学习域间映射。下文中，我们将聚焦于更难的无监督情形，尽管所提出的方法（域对抗学习）可以相当直接地推广到半监督情形。

与许多先前使用固定特征表示的域适应论文不同，我们专注于在一个训练过程中结合域适应和深度特征学习。我们的目标是将域适应嵌入表示学习的过程中，使得最终的分类决策基于既具有判别性又对域变化不变的特征，即特征在源域和目标域中具有相同或非常相似的分布。通过这种方式，所获得的前馈网络可以适用于目标域，而不受两个域之间偏移的阻碍。我们的方法受到域适应理论（Ben-David et al., 2006, 2010）的启发，该理论表明：跨域迁移的良好表示应当使得一个算法无法学习识别输入观测的域来源。

因此，我们专注于学习结合了（i）判别性和（ii）域不变性的特征。这通过联合优化底层特征以及两个在这些特征上运行的判别性分类器来实现：（i）标签预测器，用于预测类别标签，在训练和测试时都使用；（ii）域分类器，在训练期间区分源域和目标域。虽然分类器的参数被优化以最小化其在训练集上的误差，但底层深度特征映射的参数被优化以最小化标签分类器的损失并最大化域分类器的损失。因此，后一种更新对域分类器起对抗作用，它促使域不变特征在优化过程中出现。

关键的是，我们证明这三种训练过程可以嵌入到一个适当组合的深度前馈网络——称为域对抗神经网络（DANN）（见图1，第12页）——中使用标准层和损失函数，并且可以使用基于随机梯度下降或其变体（例如带动量的SGD）的标准反向传播算法进行训练。该方法是通用的，因为几乎任何可通过反向传播训练的前馈架构都可以创建DANN版本。在实践中，所提出架构中唯一的非标准组件是一个相当平凡的梯度反转层，它在前向传播中保持输入不变，在反向传播中将梯度乘以一个负标量来反转梯度。

我们对所提出的域对抗学习思想在一系列深度架构和应用上提供了实验评估。我们首先考虑最简单的DANN架构，其中三个部分（标签预测器、域分类器和特征提取器）是线性的，并展示了域对抗学习在此类架构上的成功。评估在合成数据以及自然语言处理的情感分析问题上进行，其中DANN在常见的Amazon评论基准上改进了Chen et al. (2012)的最先进边缘化堆叠自编码器（mSDA）。

我们进一步对图像分类任务进行了广泛的评估，并在传统深度学习图像数据集上呈现结果——如MNIST（LeCun et al., 1998）和SVHN（Netzer et al., 2011）——以及Office基准（Saenko et al., 2010），其中域对抗学习使我们能够获得一个深度架构，该架构比先前的最先进准确率有显著提升。

最后，我们在行人重识别应用（Gong et al., 2014）的背景下评估了域对抗描述子学习，该任务要求获得适合检索和验证的良好行人图像描述子。我们应用了域对抗学习，使用以Siamese式损失训练的预测器描述子来代替以分类损失训练的标签预测器。在一系列实验中，我们证明域对抗学习可以显著改善跨数据集重识别。

## 2. 相关工作

实现域适应的通用方法已在多个方面进行了探索。多年来，大部分文献主要关注线性假设（参见例如Blitzer et al., 2006; Bruzzone and Marconcini, 2010; Germain et al., 2013; Baktashmotlagh et al., 2013; Cortes and Mohri, 2014）。最近，非线性表示受到越来越多的研究，包括神经网络表示（Glorot et al., 2011; Li et al., 2014），尤其值得注意的是最先进的mSDA（Chen et al., 2012）。该文献主要专注于利用基于去噪自编码器范式（Vincent et al., 2008）的鲁棒表示原理。

与此同时，针对无监督域适应提出了多种匹配源域和目标域中特征分布的方法。一些方法通过对源域样本重新加权或选择来实现（Borgwardt et al., 2006; Huang et al., 2006; Gong et al., 2013），而另一些方法则寻求一种显式的特征空间变换，将源分布映射到目标分布（Pan et al., 2011; Gopalan et al., 2011; Baktashmotlagh et al., 2013）。分布匹配方法的一个重要方面是测量分布之间（不）相似性的方式。这里一个流行的选择是在核再生希尔伯特空间中匹配分布均值（Borgwardt et al., 2006; Huang et al., 2006），而Gong et al. (2012)和Fernando et al. (2013)则映射与每个分布相关的主轴。

我们的方法也尝试匹配特征空间分布，然而这是通过修改特征表示本身来实现的，而不是通过重新加权或几何变换。此外，我们的方法使用了一种相当不同的方式来测量分布之间的差异，即基于深层判别性训练分类器的可分性。还应注意，有几种方法通过逐渐改变训练分布来进行从源域到目标域的过渡（Gopalan et al., 2011; Gong et al., 2012）。在这些方法中，Chopra et al. (2013)以一种"深度"方式，通过逐层训练一系列深度自编码器，同时逐渐用目标域样本替换源域样本。这改进了Glorot et al. (2011)的类似方法，后者简单地训练单个深度自编码器用于两个域。在这两种方法中，实际的分类器/预测器是使用自编码器学习的特征表示在单独的步骤中学习的。与Glorot et al. (2011); Chopra et al. (2013)不同，我们的方法在一个统一的架构中联合执行特征学习、域适应和分类器学习，并且使用单一学习算法（反向传播）。因此，我们认为我们的方法更简单（无论是在概念上还是在实现上）。我们的方法在流行的Office基准上也取得了更好的结果。

虽然上述方法执行的是无监督域适应，但也有方法通过利用目标域的标注数据执行有监督域适应。在深度前馈架构的背景下，这些数据可用于"微调"在源域上训练的网络（Zeiler and Fergus, 2013; Oquab et al., 2014; Babenko et al., 2014）。我们的方法不需要标注的目标域数据。同时，当这类数据可用时，它可以很容易地纳入这些数据。

Goodfellow et al. (2014)描述了一个与我们相关的想法。虽然他们的目标相当不同（构建能够合成样本的生成式深度网络），但他们测量和最小化训练数据分布与合成数据分布之间差异的方式，与我们架构测量和最小化两个域特征分布之间差异的方式非常相似。此外，作者提到了饱和sigmoid的问题，这可能由于域的显著不相似而在训练早期阶段出现。他们用来避免这个问题的技术（梯度的"对抗"部分被替换为关于适当代价计算的梯度）直接适用于我们的方法。

此外，Tzeng et al. (2014); Long and Wang (2015)最近和同时进行的报告关注前馈网络中的域适应。他们的一套技术测量并最小化跨域的数据分布均值之间的距离（可能在将分布嵌入RKHS之后）。因此，他们的方法与我们通过使分布对判别性分类器不可区分来匹配分布的思想不同。下文中，我们在Office基准上将我们的方法与Tzeng et al. (2014); Long and Wang (2015)进行了比较。另一种深度域适应方法由Chen et al. (2015)并行开发，可以说与我们更加不同。

从理论角度来看，我们的方法直接源自Ben-David et al. (2006, 2010)的开创性理论工作。事实上，DANN直接优化了H-divergence的概念。我们确实注意到Huang and Yates (2012)的工作，其中使用受Ben-David等人工作启发的后验正则化器学习词标注的HMM表示。除了任务不同——Huang and Yates (2012)关注词标注问题——我们认为DANN的学习目标更直接地优化了H-divergence，而Huang and Yates (2012)出于效率原因依赖于更粗糙的近似。

本文的一部分已作为会议论文发表（Ganin and Lempitsky, 2015）。此版本在整合报告Ajakan et al. (2014)（在第二届迁移与多任务学习研讨会上发表）的基础上，非常显著地扩展了Ganin and Lempitsky (2015)，引入了新的术语、深入的理论分析和方法的论证，以及在合成数据和自然语言处理任务（情感分析）上对浅层DANN案例的广泛实验。此外，在这个版本中，我们超越了分类，在行人重识别应用中评估了描述子学习场景的域对抗学习。

## 3. 域适应

我们考虑分类任务，其中 $X$ 是输入空间，$Y = \{0, 1, \ldots, L-1\}$ 是 $L$ 个可能标签的集合。此外，我们在 $X \times Y$ 上有两个不同的分布，称为源域 $D_S$ 和目标域 $D_T$。一个无监督域适应学习算法被给予从 $D_S$ 独立同分布抽取的标注源样本 $S$，以及从 $D_X^T$ 独立同分布抽取的未标注目标样本 $T$，其中 $D_X^T$ 是 $D_T$ 在 $X$ 上的边际分布。

$$S = \{(x_i, y_i)\}_{i=1}^n \sim (D_S)^n ; \quad T = \{x_i\}_{i=n+1}^N \sim (D_X^T)^{n'},$$

其中 $N = n + n'$ 是样本总数。学习算法的目标是构建一个具有低目标风险的分类器 $\eta : X \to Y$：

$$R_{D_T}(\eta) = \Pr_{(x,y) \sim D_T} \left[ \eta(x) \neq y \right],$$

同时对 $D_T$ 的标签没有任何信息。

### 3.1 域散度

为了应对具有挑战性的域适应任务，许多方法将目标误差界定为源误差与源分布和目标分布之间距离概念之和。这些方法通过一个简单的假设得到直观证明：当两个分布相似时，源风险预期是目标风险的良好指标。针对域适应提出了若干距离概念（Ben-David et al., 2006, 2010; Mansour et al., 2009a,b; Germain et al., 2013）。在本文中，我们聚焦于Ben-David et al. (2006, 2010)使用的H-divergence，该概念基于Kifer et al. (2004)的早期工作。注意，我们在下面的定义1中假设假设类 $H$ 是一组（离散或连续的）二值分类器 $\eta : X \to \{0, 1\}$。[^1]

[^1]: 正如Ben-David et al. (2006)提到的，对于多类设置，同样的分析成立。然而，当 $|Y| > 2$ 时，为获得相同结果，应假设 $H$ 是对称假设类。即，对于所有 $h \in H$ 和任何标签排列 $c : Y \to Y$，有 $c(h) \in H$。注意，对于大多数常用的神经网络架构，情况确实如此。

**定义1**（Ben-David et al., 2006, 2010; Kifer et al., 2004）给定 $X$ 上的两个域分布 $D_X^S$ 和 $D_X^T$，以及假设类 $H$，$D_X^S$ 和 $D_X^T$ 之间的H-divergence定义为：

$$d_H(D_X^S, D_X^T) = 2 \sup_{\eta \in H} \left| \Pr_{x \sim D_X^S} [\eta(x) = 1] - \Pr_{x \sim D_X^T} [\eta(x) = 1] \right|.$$

也就是说，H-divergence依赖于假设类 $H$ 区分由 $D_X^S$ 生成的样本和由 $D_X^T$ 生成的样本的能力。Ben-David et al. (2006, 2010)证明，对于对称假设类 $H$，可以通过计算以下公式来计算两个样本 $S \sim (D_X^S)^n$ 和 $T \sim (D_X^T)^{n'}$ 之间的经验H-divergence：

$$\hat{d}_H(S, T) = 2 \left( 1 - \min_{\eta \in H} \left[ \frac{1}{n} \sum_{i=1}^{n} I[\eta(x_i) = 0] + \frac{1}{n'} \sum_{i=n+1}^{N} I[\eta(x_i) = 1] \right] \right), \tag{1}$$

其中 $I[a]$ 是指示函数，如果谓词 $a$ 为真则取值为1，否则为0。

### 3.2 Proxy距离

Ben-David et al. (2006)建议，即使通常很难精确计算 $\hat{d}_H(S, T)$（例如，当 $H$ 是 $X$ 上线性分类器的空间时），我们可以通过运行为区分源样本和目标样本问题而运行学习算法来轻松近似它。为此，我们构造一个新的数据集：

$$U = \{(x_i, 0)\}_{i=1}^n \cup \{(x_i, 1)\}_{i=n+1}^N, \tag{2}$$

其中源样本的样本标签为0，目标样本的样本标签为1。然后，在区分源样本和目标样本的问题上，在新数据集 $U$ 上训练的分类器的风险近似于公式(1)中的"min"部分。给定区分源样本和目标样本问题的泛化误差 $\epsilon$，H-divergence近似为：

$$\hat{d}_A = 2 (1 - 2\epsilon). \tag{3}$$

在Ben-David et al. (2006)中，值 $\hat{d}_A$ 被称为Proxy A-distance（PAD）。A-distance定义为 $d_A(D_X^S, D_X^T) = 2 \sup_{A \in \mathcal{A}} |\Pr_{D_X^S}(A) - \Pr_{D_X^T}(A)|$，其中 $\mathcal{A}$ 是 $X$ 的一个子集。注意，通过选择 $\mathcal{A} = \{A_\eta | \eta \in H\}$，其中 $A_\eta$ 是由特征函数 $\eta$ 表示的集合，A-distance和定义1的H-divergence是相同的。

在本文的实验部分，我们遵循Glorot et al. (2011); Chen et al. (2012)的方法计算PAD值，即，我们在 $U$（公式2）的一个子集上训练一个线性SVM或更深层的MLP分类器，并使用获得的分类器在另一个子集上的误差作为公式(3)中 $\epsilon$ 的值。关于线性SVM情形的更多细节和说明在第5.1.5节中提供。

### 3.3 目标风险的泛化界

Ben-David et al. (2006, 2010)的工作还表明，H-divergence $d_H(D_X^S, D_X^T)$ 的上界是其经验估计 $\hat{d}_H(S, T)$ 加上一个依赖于 $H$ 的VC维和样本 $S$、$T$ 大小的常数复杂度项。通过将这一结果与对源风险的类似界结合，获得了以下定理。

**定理2**（Ben-David et al., 2006）设 $H$ 是VC维为 $d$ 的假设类。以概率 $1 - \delta$，在选择样本 $S \sim (D_S)^n$ 和 $T \sim (D_X^T)^n$ 的情况下，对每个 $\eta \in H$：

$$R_{D_T}(\eta) \leq R_S(\eta) + \sqrt{\frac{4}{n} \left( d \log \frac{2e n}{d} + \log \frac{4}{\delta} \right)} + \hat{d}_H(S, T) + 4 \sqrt{\frac{1}{n} \left( d \log \frac{2n}{d} + \log \frac{4}{\delta} \right)} + \beta,$$

其中 $\beta \geq \inf_{\eta^* \in H} [R_{D_S}(\eta^*) + R_{D_T}(\eta^*)]$，且 $R_S(\eta) = \frac{1}{n} \sum_{i=1}^{m} I [\eta(x_i) \neq y_i]$ 是经验源风险。

上述结果告诉我们，$R_{D_T}(\eta)$ 仅当 $\beta$ 项很小时才可能很小，即仅当存在一个可以在两个分布上都实现低风险的分类器时。它还告诉我们，要在给定的固定VC维类中找到具有较小 $R_{D_T}(\eta)$ 的分类器，学习算法应该（在该类中）最小化源风险 $R_S(\eta)$ 和经验H-divergence $\hat{d}_H(S, T)$ 之间的权衡。正如Ben-David et al. (2006)所指出的，控制H-divergence的一个策略是找到一个样本的表示，其中源域和目标域尽可能不可区分。在这种表示下，根据定理2，具有低源风险的假设将在目标数据上表现良好。在本文中，我们提出了一个直接利用这一思想的算法。

## 4. 域对抗神经网络（DANN）

我们方法的一个原创性方面是显式地将定理2所展示的思想实现在神经网络分类器中。也就是说，为了学习一个能够从一个域很好地泛化到另一个域的模型，我们确保神经网络的内部表示不包含关于输入来源（源域或目标域）的判别信息，同时保持对源域（标注）样本的低风险。

在本节中，我们详细描述了将"域适应组件"引入神经网络的方法。在第4.1节中，我们首先对最简单的情况——即单隐藏层、全连接神经网络——展开这一思想。然后，我们描述如何将该方法推广到任意（深层）网络架构。

### 4.1 浅层神经网络的示例案例

让我们首先考虑一个具有单隐藏层的标准神经网络（NN）架构。为简单起见，我们假设输入空间由 $m$ 维实向量组成。因此，$X = \mathbb{R}^m$。隐藏层 $G_f$ 学习一个函数 $G_f : X \to \mathbb{R}^D$，将样本映射到新的 $D$ 维表示[^2]，并由矩阵-向量对 $(W, b) \in \mathbb{R}^{D \times m} \times \mathbb{R}^D$ 参数化：

[^2]: 为了记号简洁，我们有时会省略 $G_f$ 对其参数 $(W, b)$ 的依赖，将 $G_f(x; W, b)$ 简写为 $G_f(x)$。

$$G_f(x; W, b) = \text{sigm}(Wx + b), \tag{4}$$

其中 $\text{sigm}(a) = \left[ \frac{1}{1 + \exp(-a_i)} \right]_{i=1}^{|a|}$。

类似地，预测层 $G_y$ 学习一个函数 $G_y : \mathbb{R}^D \to [0, 1]^L$，由对 $(V, c) \in \mathbb{R}^{L \times D} \times \mathbb{R}^L$ 参数化：

$$G_y(G_f(x); V, c) = \text{softmax}(V G_f(x) + c),$$

其中 $\text{softmax}(a) = \left[ \frac{\exp(a_i)}{\sum_{j=1}^{|a|} \exp(a_j)} \right]_{i=1}^{|a|}$。

这里 $L = |Y|$。通过使用softmax函数，向量 $G_y(G_f(x))$ 的每个分量表示神经网络将 $x$ 分配给由该分量表示的 $Y$ 中类别的条件概率。给定一个源样本 $(x_i, y_i)$，自然使用的分类损失是正确标签的负对数概率：

$$L_y \big( G_y(G_f(x_i)), y_i \big) = \log \frac{1}{G_y(G_f(x))_{y_i}}.$$

训练神经网络则导致如下源域上的优化问题：

$$\min_{W, b, V, c} \left[ \frac{1}{n} \sum_{i=1}^{n} L^i_y(W, b, V, c) + \lambda \cdot R(W, b) \right], \tag{5}$$

其中 $L^i_y(W, b, V, c) = L_y \big( G_y(G_f(x_i; W, b); V, c), y_i \big)$ 是第 $i$ 个样本上预测损失的简写记号，$R(W, b)$ 是由超参数 $\lambda$ 加权的可选正则化器。

我们方法的核心是设计一个直接源自定义1的H-divergence的域正则化器。为此，我们将隐藏层 $G_f(\cdot)$（公式4）的输出视为神经网络的内部表示。因此，我们将源样本的表示记为：

$$S(G_f) = \{ G_f(x) \mid x \in S \}.$$

类似地，给定来自目标域的未标注样本，我们将对应的表示记为：

$$T(G_f) = \{ G_f(x) \mid x \in T \}.$$

基于公式(1)，对称假设类 $H$ 在样本 $S(G_f)$ 和 $T(G_f)$ 之间的经验H-divergence由下式给出：

$$\hat{d}_H \big( S(G_f), T(G_f) \big) = 2 \Bigg( 1 - \min_{\eta \in H} \Bigg[ \frac{1}{n} \sum_{i=1}^{n} I \big[ \eta(G_f(x_i)) = 0 \big] + \frac{1}{n'} \sum_{i=n+1}^{N} I \big[ \eta(G_f(x_i)) = 1 \big] \Bigg] \Bigg). \tag{6}$$

让我们将 $H$ 视为表示空间中的超平面类。受Proxy A-distance启发（见第3.2节），我们建议通过一个域分类层 $G_d$ 来估计公式(6)的"min"部分，该层学习一个逻辑回归器 $G_d : \mathbb{R}^D \to [0, 1]$，由向量-标量对 $(u, z) \in \mathbb{R}^D \times \mathbb{R}$ 参数化，建模给定输入来自源域 $D_X^S$ 或目标域 $D_X^T$ 的概率。因此：

$$G_d(G_f(x); u, z) = \text{sigm}(u^\top G_f(x) + z). \tag{7}$$

因此，函数 $G_d(\cdot)$ 是一个域回归器。我们定义其损失为：

$$L_d \big( G_d(G_f(x_i)), d_i \big) = d_i \log \frac{1}{G_d(G_f(x_i))} + (1 - d_i) \log \frac{1}{1 - G_d(G_f(x_i))},$$

其中 $d_i$ 表示第 $i$ 个样本的二元变量（域标签），指示 $x_i$ 是来自源分布（$x_i \sim D_X^S$ 时 $d_i = 0$）还是目标分布（$x_i \sim D_X^T$ 时 $d_i = 1$）。

回想，对于来自源分布的样本（$d_i = 0$），对应的标签 $y_i \in Y$ 在训练时是已知的。对于来自目标域的样本，我们在训练时不知道标签，并且在测试时我们希望预测这些标签。这使得我们能够将域适应项添加到公式(5)的目标中，得到以下正则化器：

$$R(W, b) = \max_{u, z} \left[ -\frac{1}{n} \sum_{i=1}^{n} L^i_d(W, b, u, z) - \frac{1}{n'} \sum_{i=n+1}^{N} L^i_d(W, b, u, z) \right], \tag{8}$$

其中 $L^i_d(W, b, u, z) = L_d \big( G_d(G_f(x_i; W, b); u, z), d_i \big)$。这个正则化器试图近似公式(6)的H-divergence，因为 $2(1 - R(W, b))$ 是 $\hat{d}_H(S(G_f), T(G_f))$ 的代理。根据定理2，公式(5)和(8)给出的优化问题实现了源风险 $R_S(\cdot)$ 的最小化与散度 $\hat{d}_H(\cdot, \cdot)$ 之间的权衡。超参数 $\lambda$ 用于在学习过程中调节这两个量之间的权衡。

对于学习，我们首先注意到可以将公式(5)的完整优化目标重写如下：

$$E(W, V, b, c, u, z) = \frac{1}{n} \sum_{i=1}^{n} L^i_y(W, b, V, c) - \lambda \left( \frac{1}{n} \sum_{i=1}^{n} L^i_d(W, b, u, z) + \frac{1}{n'} \sum_{i=n+1}^{N} L^i_d(W, b, u, z) \right), \tag{9}$$

其中我们寻求参数 $\hat{W}, \hat{V}, \hat{b}, \hat{c}, \hat{u}, \hat{z}$ 给出以下鞍点：

$$(\hat{W}, \hat{V}, \hat{b}, \hat{c}) = \operatorname{argmin}_{W, V, b, c} E(W, V, b, c, \hat{u}, \hat{z}),$$

$$(\hat{u}, \hat{z}) = \operatorname{argmax}_{u, z} E(\hat{W}, \hat{V}, \hat{b}, \hat{c}, u, z).$$

因此，优化问题涉及对某些参数的最小化以及对其他参数的最大化。

**算法1：浅层DANN -- 随机训练更新**

**输入：**
- 样本 $S = \{(x_i, y_i)\}_{i=1}^n$ 和 $T = \{x_i\}_{i=1}^{n'}$
- 隐藏层大小 $D$
- 适应参数 $\lambda$
- 学习率 $\mu$

**输出：** 神经网络 $\{W, V, b, c\}$

1. $W, V \leftarrow$ 随机初始化(D)
2. $b, c, u, d \leftarrow 0$
3. **while** 停止条件未满足 **do**
4.     **for** $i$ 从 1 到 $n$ **do**
5.         **# 前向传播**
6.         $G_f(x_i) \leftarrow \text{sigm}(b + W x_i)$
7.         $G_y(G_f(x_i)) \leftarrow \text{softmax}(c + V G_f(x_i))$
8.         **# 反向传播**
9.         $\Delta_c \leftarrow -(e(y_i) - G_y(G_f(x_i)))$
10.         $\Delta_V \leftarrow \Delta_c G_f(x_i)^\top$
11.         $\Delta_b \leftarrow (V^\top \Delta_c) \odot G_f(x_i) \odot (1 - G_f(x_i))$
12.         $\Delta_W \leftarrow \Delta_b \cdot (x_i)^\top$
13.         **# 域适应正则化器...**
14.         **# ...来自当前域**
15.         $G_d(G_f(x_i)) \leftarrow \text{sigm}(d + u^\top G_f(x_i))$
16.         $\Delta_d \leftarrow \lambda(1 - G_d(G_f(x_i)))$
17.         $\Delta_u \leftarrow \lambda(1 - G_d(G_f(x_i))) G_f(x_i)$
18.         $tmp \leftarrow \lambda(1 - G_d(G_f(x_i))) \times u \odot G_f(x_i) \odot (1 - G_f(x_i))$
19.         $\Delta_b \leftarrow \Delta_b + tmp$
20.         $\Delta_W \leftarrow \Delta_W + tmp \cdot (x_i)^\top$
21.         **# ...来自另一个域**
22.         $j \leftarrow$ 均匀整数$(1, \ldots, n')$
23.         $G_f(x_j) \leftarrow \text{sigm}(b + W x_j)$
24.         $G_d(G_f(x_j)) \leftarrow \text{sigm}(d + u^\top G_f(x_j))$
25.         $\Delta_d \leftarrow \Delta_d - \lambda G_d(G_f(x_j))$
26.         $\Delta_u \leftarrow \Delta_u - \lambda G_d(G_f(x_j)) G_f(x_j)$
27.         $tmp \leftarrow -\lambda G_d(G_f(x_j)) \times u \odot G_f(x_j) \odot (1 - G_f(x_j))$
28.         $\Delta_b \leftarrow \Delta_b + tmp$
29.         $\Delta_W \leftarrow \Delta_W + tmp \cdot (x_j)^\top$
30.         **# 更新神经网络参数**
31.         $W \leftarrow W - \mu \Delta_W$
32.         $V \leftarrow V - \mu \Delta_V$
33.         $b \leftarrow b - \mu \Delta_b$
34.         $c \leftarrow c - \mu \Delta_c$
35.         **# 更新域分类器**
36.         $u \leftarrow u + \mu \Delta_u$
37.         $d \leftarrow d + \mu \Delta_d$
38.     **end for**
39. **end while**

**注：** 在此伪代码中，$e(y)$ 指"独热"向量，除位置 $y$ 处为1外全为0，$\odot$ 是逐元素乘积。

我们建议使用简单的随机梯度过程来解决这个问题，其中最小化参数的更新沿公式(9)梯度的相反方向进行，最大化参数的更新沿梯度方向进行。梯度的随机估计使用一部分训练样本来计算平均值。算法1提供了这一学习过程的完整伪代码。[^3] 换句话说，在训练过程中，神经网络（由 $W, b, V, c$ 参数化）和域回归器（由 $u, z$ 参数化）在公式(9)的目标上以对抗方式相互竞争。因此，我们将按照此目标训练的网络称为域对抗神经网络（DANN）。DANN将有效地尝试学习一个隐藏层 $G_f(\cdot)$，它将样本（源域或目标域）映射到一个表示中，该表示允许输出层 $G_y(\cdot)$ 准确分类源样本，但削弱域回归器 $G_d(\cdot)$ 检测每个样本是属于源域还是目标域的能力。

[^3]: 我们在 http://graal.ift.ulaval.ca/dann/ 提供了浅层DANN算法的实现。

### 4.2 推广到任意架构

为了说明的目的，我们到目前为止聚焦于单隐藏层DANN的情况。然而，将其推广到其他复杂的架构是直接的，这些架构可能更适合手头的数据。例如，深度卷积神经网络以学习图像判别性特征的最先进模型而闻名（Krizhevsky et al., 2012）。

让我们现在对DANN的不同组件使用更一般的记号。具体来说，设 $G_f(\cdot; \theta_f)$ 为 $D$ 维的神经网络特征提取器，参数为 $\theta_f$。此外，设 $G_y(\cdot; \theta_y)$ 是计算网络标签预测输出层的DANN部分，参数为 $\theta_y$，而 $G_d(\cdot; \theta_d)$ 现在对应网络的域预测输出计算，参数为 $\theta_d$。注意，为了保持定理2的理论保证，域预测组件 $G_d$ 生成的假设类 $\mathcal{H}_d$ 应包含标签预测组件 $G_y$ 生成的假设类 $\mathcal{H}_y$。因此，$\mathcal{H}_y \subseteq \mathcal{H}_d$。

我们分别记预测损失和域损失为：

$$L^i_y(\theta_f, \theta_y) = L_y \big( G_y(G_f(x_i; \theta_f); \theta_y), y_i \big),$$

$$L^i_d(\theta_f, \theta_d) = L_d \big( G_d(G_f(x_i; \theta_f); \theta_d), d_i \big).$$

训练DANN与单层情况类似，包括优化：

$$E(\theta_f, \theta_y, \theta_d) = \frac{1}{n} \sum_{i=1}^{n} L^i_y(\theta_f, \theta_y) - \lambda \left( \frac{1}{n} \sum_{i=1}^{n} L^i_d(\theta_f, \theta_d) + \frac{1}{n'} \sum_{i=n+1}^{N} L^i_d(\theta_f, \theta_d) \right), \tag{10}$$

通过找到鞍点 $\hat{\theta}_f, \hat{\theta}_y, \hat{\theta}_d$ 使得：

$$(\hat{\theta}_f, \hat{\theta}_y) = \operatorname{argmin}_{\theta_f, \theta_y} E(\theta_f, \theta_y, \hat{\theta}_d), \tag{11}$$

$$\hat{\theta}_d = \operatorname{argmax}_{\theta_d} E(\hat{\theta}_f, \hat{\theta}_y, \theta_d). \tag{12}$$

如前所述，由公式(11-12)定义的鞍点可作为以下梯度更新的驻点找到：

$$\theta_f \leftarrow \theta_f - \mu \left( \frac{\partial L^i_y}{\partial \theta_f} - \lambda \frac{\partial L^i_d}{\partial \theta_f} \right), \tag{13}$$

$$\theta_y \leftarrow \theta_y - \mu \frac{\partial L^i_y}{\partial \theta_y}, \tag{14}$$

$$\theta_d \leftarrow \theta_d - \mu \lambda \frac{\partial L^i_d}{\partial \theta_d}, \tag{15}$$

其中 $\mu$ 是学习率。我们通过从数据集中采样样本来使用这些梯度的随机估计。

**图1：** 所提出的架构包括一个深度特征提取器（绿色）和一个深度标签预测器（蓝色），它们共同构成一个标准的前馈架构。无监督域适应通过添加一个域分类器（红色）实现，该域分类器通过梯度反转层连接到特征提取器，梯度反转层在基于反向传播的训练期间将梯度乘以某个负常数。除此之外，训练按标准方式进行，最小化标签预测损失（针对源样本）和域分类损失（针对所有样本）。梯度反转确保两个域上的特征分布变得相似（对域分类器而言尽可能不可区分），从而产生域不变特征。

公式(13-15)的更新与包含特征提取器馈入标签预测器和域分类器（损失由 $\lambda$ 加权）的前馈深度模型的随机梯度下降（SGD）更新非常相似。唯一的区别是在公式(13)中，来自类别预测器和域预测器的梯度相减，而不是相加（这个区别很重要，否则SGD将试图使特征跨域不相似以最小化域分类损失）。由于SGD——及其许多变体，如ADAGRAD（Duchi et al., 2010）或ADADELTA（Zeiler, 2012）——是大多数深度学习库中实现的主要学习算法，将我们的随机鞍点过程实现框架化为SGD将非常方便。

幸运的是，这种归约可以通过引入一个特殊的梯度反转层（GRL）来实现，定义如下。梯度反转层本身没有参数。在前向传播期间，GRL充当恒等变换。然而，在反向传播期间，GRL从后续层获取梯度并改变其符号，即在将其传递到前一层之前乘以 $-1$。使用现有的面向对象的深度学习包实现这样的层很简单，只需要定义前向传播（恒等变换）和反向传播（乘以 $-1$）的过程。该层不需要参数更新。

如上定义的GRL被插入在特征提取器 $G_f$ 和域分类器 $G_d$ 之间，产生了图1所示的架构。当反向传播过程经过GRL时，GRL下游（即 $L_d$）的损失对GRL上游层参数（即 $\theta_f$）的偏导数被乘以 $-1$，即 $\frac{\partial L_d}{\partial \theta_f}$ 实际上被替换为 $-\frac{\partial L_d}{\partial \theta_f}$。因此，在结果模型中运行SGD实现了公式(13-15)的更新，并收敛到公式(10)的鞍点。

数学上，我们可以将梯度反转层形式化地处理为一个"伪函数" $R(x)$，由两个（不相容的）方程描述其前向和反向传播行为：

$$R(x) = x, \tag{16}$$

$$\frac{dR}{dx} = -I, \tag{17}$$

其中 $I$ 是单位矩阵。然后，我们可以定义关于 $(\theta_f, \theta_y, \theta_d)$ 的目标"伪函数"，该函数在我们的方法中通过随机梯度下降来优化：

$$\tilde{E}(\theta_f, \theta_y, \theta_d) = \frac{1}{n} \sum_{i=1}^{n} L_y \big( G_y(G_f(x_i; \theta_f); \theta_y), y_i \big) - \lambda \Bigg( \frac{1}{n} \sum_{i=1}^{n} L_d \big( G_d(R(G_f(x_i; \theta_f)); \theta_d), d_i \big) + \frac{1}{n'} \sum_{i=n+1}^{N} L_d \big( G_d(R(G_f(x_i; \theta_f)); \theta_d), d_i \big) \Bigg). \tag{18}$$

运行更新(13-15)可以作为对(18)执行SGD来实现，并导致既域不变又具有判别性的特征的出现。学习之后，标签预测器 $G_y(G_f(x; \theta_f); \theta_y)$ 可用于预测目标域（以及源域）样本的标签。注意，我们发布了梯度反转层的源代码及使用示例，作为Caffe（Jia et al., 2014）的扩展。[^4]

[^4]: http://sites.skoltech.ru/compvision/projects/grl/

## 5. 实验

在本节中，我们呈现浅层域对抗神经网络（第5.1节）和深层域对抗神经网络（第5.2节和第5.3节）的各种实证结果。

### 5.1 浅层神经网络实验

在第一个实验部分中，我们评估了第4.1节描述的简单版DANN的行为。注意，本节报告的结果是使用算法1获得的。因此，这里的随机梯度下降方法包括采样一对源样本和目标样本，并对DANN的所有参数执行梯度步更新。关键的是，虽然常规参数的更新照常遵循梯度的相反方向，但对于对抗参数，步必须遵循梯度方向（因为我们关于这些参数最大化，而不是最小化）。

#### 5.1.1 玩具问题实验

作为第一个实验，我们研究所提出算法在交叉双月二维问题变体上的行为，其中目标分布是源分布的旋转。作为源样本 $S$，我们生成下弦月和上弦月，分别标记为0和1，每个包含150个样本。目标样本 $T$ 通过以下过程获得：（1）我们以与生成 $S$ 相同的方式生成一个样本 $S'$；（2）将每个样本旋转35度；（3）移除所有标签。因此，$T$ 包含300个未标注样本。我们在图2中表示了这些样本。

我们通过将DANN与标准神经网络（NN）进行比较来研究DANN的适应能力。在这些玩具实验中，两种算法共享相同的网络架构，隐藏层大小为15个神经元。我们使用与DANN相同的程序训练NN。也就是说，我们继续使用目标样本 $T$ 更新域回归器组件（超参数 $\lambda = 6$；DANN也使用相同的值），但我们禁用进入隐藏层的对抗反向传播。为此，我们通过省略第22行和第31行来执行算法1。这允许恢复NN学习算法——基于公式(5)的源风险最小化而不使用任何正则化器——并同时训练公式(7)的域回归器以区分源域和目标域。通过这个玩具实验，我们将首先说明与NN相比，DANN如何调整其决策边界。此外，我们还将说明，与NN相比，DANN的隐藏层表示对源域任务的适应程度更低（这就是我们在NN实验中需要域回归器的原因）。我们回忆，这是我们提出算法的基本思想。实验分析出现在图2中，上方的图与标准NN有关，下方的图与DANN有关。通过成对地观察下方和上方的图，我们从四个不同的角度比较NN和DANN，详细描述如下。

**图2的"标签分类"列**显示了DANN和NN在预测源样本和目标样本标签问题上的决策边界。如预期的那样，NN准确分类了源样本 $S$ 的两个类别，但没有完全适应目标样本 $T$。相反，DANN的决策边界完美地分类了源样本和目标样本。在所研究的任务中，DANN明确地适应了目标分布。

**"PCA表示"列**研究了域适应正则化器如何影响网络隐藏层提供的表示 $G_f(\cdot)$。这些图通过对所有源和目标数据点表示集合，即 $S(G_f) \cup T(G_f)$，应用主成分分析（PCA）获得。因此，给定训练好的网络（NN或DANN），来自 $S$ 和 $T$ 的每个点通过隐藏层映射到一个15维的特征空间，并通过PCA变换投影回二维平面。在DANN-PCA表示中，我们观察到目标点均匀地散布在源点之间；在NN-PCA表示中，许多目标点属于不含源点的簇。因此，给定DANN-PCA表示，标记目标点似乎是一个更容易的任务。

为了进一步推进分析，PCA图用字母A、B、C和D标记了四个关键数据点，它们对应原始空间中月牙的端点（注意，原始点位置在第一列图中标记）。我们观察到，点A和B在NN-PCA表示中彼此非常接近，但它们明显属于不同的类别。点C和D也是如此。相反，这四个点在DANN-PCA表示中处于相反的四个角。还需注意，目标点A（对应D）——在原始空间中难以分类——在DANN-PCA表示中位于"+"簇（对应"-"簇）。因此，DANN促进的表示更适合适应问题。

**"域分类"列**显示了域分类问题上的决策边界，由公式(7)的域回归器 $G_d$ 给出。更准确地说，一个样本 $x$ 在 $G_d(G_f(x)) \geq 0.5$ 时被分类为源样本，否则被分类为域样本。记住，在DANN的学习过程中，$G_d$ 回归器努力区分源域和目标域，而隐藏表示 $G_f(\cdot)$ 以对抗方式被更新以阻止其成功。如上所述，我们在NN的学习过程中也训练了一个域回归器，但不允许它影响学习的表示 $G_f(\cdot)$。一方面，DANN域回归器明显未能泛化源和目标分布拓扑。另一方面，NN域回归器显示出更好（尽管不完美）的泛化能力。其中，它似乎大致捕捉到了目标分布的旋转角度。这再次证实了DANN表示不允许区分域。

**"隐藏神经元"列**显示了隐藏层神经元的配置（根据公式4，每个神经元实际上是一个线性回归器）。换句话说，十五条线中的每一条对应坐标 $x \in \mathbb{R}^2$，使得 $G_f(x)$ 的第 $i$ 个分量等于 $\frac{1}{2}$，对于 $i \in \{1, \ldots, 15\}$。我们观察到标准NN神经元被分为三个簇，每个簇允许为标签分类问题生成锯齿形决策边界的一条直线。然而，这些神经元中的大多数也能够（粗略地）捕捉域分类问题的旋转角度。因此，我们观察到DANN的适应正则化器阻止了这类神经元的产生。确实引人注目的是，NN神经元中两个主要模式（即从左上到右下穿过平面的两条平行线）在DANN神经元中消失了。

#### 5.1.2 无监督超参数选择

要执行无监督域适应，应该提供以无监督方式设置超参数（如域正则化参数 $\lambda$、学习率、我们方法的网络架构）的方法，即不依赖目标域的标注数据。在第5.1.3和5.1.4节的以下实验中，我们通过使用Zhong et al. (2010)提出的反向交叉验证方法的变体（我们称之为反向验证）来选择每个算法的超参数。

为了评估与超参数元组相关的反向验证风险，我们如下进行。给定标注的源样本 $S$ 和未标注的目标样本 $T$，我们将每个集合分为训练集（分别为 $S'$ 和 $T'$，包含原始样本的90%）和验证集（分别为 $S_V$ 和 $T_V$）。我们使用标注集 $S'$ 和未标注目标集 $T'$ 来学习一个分类器 $\eta$。然后，使用相同的算法，我们使用自标记集 $\{(x, \eta(x))\}_{x \in T'}$ 和 $S'$ 的未标注部分作为目标样本来学习一个反向分类器 $\eta_r$。最后，反向分类器 $\eta_r$ 在源样本的验证集 $S_V$ 上进行评估。然后我们说分类器 $\eta$ 具有反向验证风险 $R_{S_V}(\eta_r)$。使用多个超参数值重复此过程，所选参数对应于具有最低反向验证风险的分类器。

注意，当我们训练神经网络架构时，验证集 $S_V$ 也在学习 $\eta$ 期间用作早停标准，自标记验证集 $\{(x, \eta(x))\}_{x \in T_V}$ 在学习 $\eta_r$ 期间用作早停标准。我们还观察到，当我们用网络 $\eta$ 学习到的配置初始化反向分类器 $\eta_r$ 的学习时，获得了更好的准确率。

#### 5.1.3 情感分析数据集实验

我们现在将所提出的DANN算法与公式(5)描述的具有一个隐藏层的标准神经网络（NN）以及线性核支持向量机（SVM）进行性能比较。我们在经过Chen et al. (2012)预处理的Amazon评论数据集上比较这些算法。这个数据集包含四个域，每个域由特定类型产品（书籍、DVD光盘、电子产品和厨房电器）的评论组成。评论编码为5000维的unigrams和bigrams特征向量，标签是二元的：如果产品评分不超过3星则为"0"，如果产品评分为4或5星则为"1"。

我们执行十二个域适应任务。所有学习算法被给予2000个标注源样本和2000个未标注目标样本。然后，我们在单独的目标测试集（3000到6000个样本之间）上评估它们。注意，NN和SVM不使用未标注目标样本进行学习。

以下是每个学习算法所使用过程的更多细节，用于产生表1的实证结果。

- **对于DANN算法**，适应参数 $\lambda$ 在 $10^{-2}$ 到 $1$ 之间的9个值中以对数尺度选择。隐藏层大小 $l$ 要么是50要么是100。最后，学习率 $\mu$ 固定为 $10^{-3}$。
- **对于NN算法**，我们使用与上述DANN完全相同的超参数网格和训练过程，只是我们不需要适应参数。注意，可以通过使用 $\lambda = 0$ 的DANN实现（算法1）来训练NN。
- **对于SVM算法**，超参数 $C$ 在 $10^{-5}$ 到 $1$ 之间的10个值中以对数尺度选择。这个值范围与Chen et al. (2012)在其实验中使用的范围相同。

如第5.1.2节所述，我们对所有三种学习算法使用反向交叉验证选择超参数，并使用早停作为DANN和NN的停止准则。

**表1a的"原始数据"部分**显示了所有算法的目标测试准确率，**表1b**报告了一个算法根据Poisson二项检验（Lacoste et al., 2012）显著优于其他算法的概率。我们注意到，DANN相比NN和SVM具有显著更好的性能，概率分别为0.87和0.83。由于DANN和NN之间的唯一区别是域适应正则化器，我们得出结论：我们的方法成功地帮助找到了适合目标域的表示。

#### 5.1.4 将DANN与去噪自编码器结合

我们现在研究DANN算法是否可以改进由Chen et al. (2012)提出的最先进边缘化堆叠去噪自编码器（mSDA）学习到的表示。简言之，mSDA是一种无监督算法，学习训练样本的新鲁棒特征表示。它取源样本和目标样本的未标注部分，学习从输入空间 $X$ 到新表示空间的特征映射。作为一种去噪自编码器算法，它找到一种特征表示，从中可以（近似地）从其噪声对应物重建样本的原始特征。Chen et al. (2012)表明，将mSDA与线性SVM分类器一起使用可在Amazon评论数据集上达到最先进的性能。作为SVM的替代方案，我们建议将我们的浅层DANN算法应用于mSDA生成的相同表示（使用源样本和目标样本的表示）。注意，即使mSDA和DANN是两种表示学习方法，它们优化不同的目标，这可以互补。

我们在前一节描述的相同Amazon评论数据集上执行此实验。对于每个源-目标域对，我们使用50%的损坏概率和5层数量生成mSDA表示。然后在这些表示上执行三种学习算法（DANN、NN和SVM）。更准确地说，遵循Chen et al. (2012)的实验过程，我们使用5层输出和原始输入的串联作为新表示。因此，每个样本现在被编码为30000维的向量。注意，我们使用与第5.1.3节相同的网格搜索，但对DANN和NN都使用学习率 $\mu = 10^{-4}$。表1a中"mSDA表示"列的结果证实，将mSDA和DANN结合是一种合理的方法。事实上，Poisson二项检验显示，如**表1b**所报告的，DANN优于NN和SVM的概率分别为0.92和0.88。然而，我们注意到标准NN和SVM分别在第二个和第四个任务上找到了最佳解决方案。这表明DANN和mSDA的适应策略并不完全互补。

#### 5.1.5 Proxy距离

DANN算法的理论基础是Ben-David et al. (2006, 2010)的域适应理论。我们主张DANN找到一个源样本和目标样本几乎不可区分的表示。我们在第5.1.1节的玩具实验已经指出了这一点的一些证据，这里我们在真实数据上进行分析。为此，我们比较了Amazon评论数据集各种表示上的Proxy A-distance（PAD）；这些表示是通过运行NN、DANN、mSDA或mSDA与DANN组合获得的。回想，如第3.2节所述，PAD是一个估计源表示和目标表示相似性的度量。更准确地说，为了获得PAD值，我们使用以下过程：（1）我们使用训练样本的源表示和目标表示构造公式(2)的数据集 $U$；（2）将 $U$ 随机分成两个大小相等的子集；（3）使用大范围的 $C$ 值在 $U$ 的第一个子集上训练线性SVM；（4）计算所有获得的分类器在 $U$ 的第二个子集上的误差；（5）使用最低误差计算公式(3)的PAD值。

**首先，图3a**比较了在第5.1.3节实验中获得的DANN表示的PAD（使用导致表1结果的超参数值）与原始数据上计算的PAD。如预期的那样，PAD值被DANN表示拉低。

**其次，图3b**比较了DANN表示的PAD与标准NN表示的PAD。由于PAD受隐藏层大小的影响（判别能力倾向于随表示长度增加），我们这里将两种算法的隐藏层大小固定为100个神经元。我们还将DANN的适应参数固定为 $\lambda \simeq 0.31$；这是我们在Amazon评论数据集前述实验中最常选择的值。再次地，DANN明确地导致最低的PAD值。

**最后，图3c**呈现了两组与第5.1.4节实验相关的结果。一方面，我们重现了Chen et al. (2012)的结果，他们注意到mSDA表示的PAD值大于原始（原始）数据。虽然mSDA方法明确有助于适应目标任务，但这似乎与Ben-David等人的理论相矛盾。另一方面，我们观察到，当在mSDA之上运行DANN时（使用导致表1结果的超参数值），获得的表示具有低得多的PAD值。这些观察可能解释了DANN在与mSDA过程结合时提供的改进。

### 5.2 图像分类的深层网络实验

我们现在对深层版DANN（见第4.2节）在若干流行的图像数据集及其修改上进行广泛评估。这些包括深度学习方法中流行的大规模小图像数据集，以及Office数据集（Saenko et al., 2010），这些是计算机视觉域适应的事实标准，但图像数量少得多。

#### 5.2.1 基线

本节实验中评估了以下基线。**仅源域模型**（source-only）在不考虑目标域数据的情况下训练（网络中不包含域分类器分支）。**在目标域上训练模型**（train-on-target）在目标域上训练，类别标签已知。该模型作为DA方法的上界，假设目标数据丰富且域间偏移显著。

此外，我们将我们的方法与最近提出的基于子空间对齐（SA）的无监督DA方法（Fernando et al., 2013）进行比较，该方法容易设置并在新数据集上测试，同时在与其他"浅层"DA方法的实验比较中表现出非常好的性能。为了提升该基线的性能，我们从范围 $\{2, \ldots, 60\}$ 中挑选其最重要的自由参数（主成分数量），使得目标域上的测试性能最大化。为在我们的设置中应用SA，我们训练一个仅源域模型，然后将标签预测器中最后一个隐藏层（在最终线性分类器之前）的激活视为描述子/特征，并学习源域和目标域之间的映射（Fernando et al., 2013）。由于SA基线在适应特征后需要训练一个新分类器，并且为使所有比较的设置处于同等地位，我们对所有四种考虑的方法（包括我们的方法；重训练后目标域的性能保持大致相同）使用标准线性SVM（Fan et al., 2008）重训练标签预测器的最后一层。

对于Office数据集（Saenko et al., 2010），我们使用先前发表的结果直接将我们的完整网络（特征提取器和标签预测器）的性能与最近的DA方法进行比较。

#### 5.2.2 CNN架构和训练过程

通常，我们使用两个或三个卷积层组成特征提取器，从先前工作中挑选其确切配置。更准确地说，在我们的实验中使用了四种不同的架构。前三种如图4所示。对于Office域，我们使用来自Caffe包（Jia et al., 2014）的预训练AlexNet。适应架构与Tzeng et al. (2014)相同。[^5]

[^5]: 一个两层域分类器（x→1024→1024→2）附加到fc7的256维瓶颈层。

对于域适应组件，我们使用三个（x→1024→1024→2）全连接层，除了MNIST我们使用更简单的（x→100→2）架构以加速实验。诚然，这些域分类器的选择是任意的，如果这部分架构经过调优，可能会获得更好的适应性能。

**图4：实验中使用的CNN架构。** 方框对应应用于数据的变换。颜色编码与图1相同。

- (a) MNIST架构；受经典LeNet-5（LeCun et al., 1998）启发。
- (b) SVHN架构；采用自Srivastava et al. (2014)。
- (c) GTSRB架构；我们使用Ciresan et al. (2012)的单CNN基线作为起点。

对于损失函数，我们分别设置 $L_y$ 和 $L_d$ 为逻辑回归损失和二项交叉熵。遵循Srivastava et al. (2014)，我们在训练SVHN架构时也使用dropout和 $\ell_2$-范数约束。

其他超参数没有像第5.1节小规模实验中那样通过网格搜索选择，这计算成本高昂。相反，在随机梯度下降期间使用以下公式调整学习率：

$$\mu_p = \frac{\mu_0}{(1 + \alpha \cdot p)^\beta},$$

其中 $p$ 是训练进度，从0到1线性变化，$\mu_0 = 0.01$，$\alpha = 10$ 和 $\beta = 0.75$（该调度被优化以促进收敛和源域上的低误差）。还使用了0.9的动量项。

域适应参数 $\lambda$ 初始化为0，并使用以下调度逐渐变为1：

$$\lambda_p = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1,$$

其中 $\gamma$ 在所有实验中设置为10（该调度未被优化/调整）。该策略允许域分类器在训练过程的早期阶段对噪声信号不那么敏感。然而，注意这些 $\lambda_p$ 仅用于更新特征提取器组件 $G_f$。对于更新域分类组件，我们使用固定的 $\lambda = 1$，以确保后者与标签预测器 $G_y$ 训练得一样快。[^6]

[^6]: 等价地，可以对特征提取器和域分类组件使用相同的 $\lambda_p$，但对后者使用学习率 $\mu / \lambda_p$。

最后，注意模型在128大小的批次上训练（图像通过均值减法预处理）。每个批次的一半由源域样本填充（标签已知），其余构成目标域（标签不对算法揭示，除了在目标域上训练基线）。

#### 5.2.3 可视化

我们使用t-SNE（van der Maaten, 2013）投影来可视化网络不同点的特征分布，同时对域进行颜色编码（图5）。正如我们在浅层版DANN中已经观察到的（见图2），适应在目标域分类准确率方面的成功与这些可视化中域分布之间的重叠之间存在很强的对应关系。

**图5：适应对提取特征分布的影响（彩色效果最佳）。** 该图显示了CNN激活的t-SNE可视化：（a）未进行适应的情况和（b）将我们的适应过程纳入训练的情况。蓝点对应源域样本，红点对应目标域样本。在所有情况下，我们方法中的适应使两个特征分布更加接近。

#### 5.2.4 图像数据集结果

我们现在讨论实验设置和结果。在每种情况下，我们在源数据集上训练，在具有显著域间偏移的不同目标域数据集上测试（见图6）。结果总结在表2和表3中。

**图6：实验中使用的域对示例。** 详见第5.2.4节。

**MNIST → MNIST-M。** 我们的第一个实验涉及MNIST数据集（LeCun et al., 1998）（源域）。为了获得目标域（MNIST-M），我们将原始集中的数字与从BSDS500（Arbelaez et al., 2011）彩色照片中随机提取的斑块混合。此操作正式定义为对于两幅图像 $I^1, I^2$ 有 $I^{out}_{ijk} = |I^1_{ijk} - I^2_{ijk}|$，其中 $i, j$ 是像素坐标，$k$ 是通道索引。换句话说，输出样本通过取照片中的一个斑块并在对应数字像素的位置反转其像素来产生。对于人类而言，分类任务相比原始数据集仅略微变难（数字仍然清晰可辨），而对于在MNIST上训练的CNN，此域相当不同，因为背景和笔画不再是恒定的。因此，仅源域模型表现不佳。我们的方法成功地配准了特征分布（图5），这导致了成功的适应结果（考虑到适应是无监督的）。同时，子空间对齐（SA）（Fernando et al., 2013）实现的改进相比仅源域模型相当有限，从而突出了适应任务的困难。

**合成数字 → SVHN。** 为了应对在合成数据上训练和在真实数据上测试的常见场景，我们使用街景门牌号数据集SVHN（Netzer et al., 2011）作为目标域，合成数字作为源域。后者（Syn Numbers）由我们自己从Windows字体生成的约500,000张图像组成，通过改变文本（包括不同的一、二和三位数字）、定位、方向、背景和笔画颜色以及模糊程度。变化程度被手动选择以模拟SVHN，然而这两个数据集仍然相当不同，最大的差异是SVHN图像背景中的结构化杂乱。

所提出的基于反向传播的技术效果良好，覆盖了仅使用源数据训练和使用已知目标标签在目标域数据上训练之间差距的近80%。相比之下，SA（Fernando et al., 2013）导致分类准确率轻微下降（可能是由于降维期间的信息损失），表明适应任务比MNIST实验更具挑战性。

**MNIST ↔ SVHN。** 在这个实验中，我们进一步增加了分布之间的差距，并在外观显著不同的MNIST和SVHN上进行测试。在SVHN上即使没有适应的训练也很有挑战性——在前150个epoch中分类误差保持高位。为了避免陷入较差的局部最小值，我们因此在这里不使用学习率退火。显然，两个方向（MNIST → SVHN和SVHN → MNIST）难度不等。由于SVHN更多样化，在SVHN上训练的模型预计更具通用性，并在MNIST数据集上表现合理。这确实是事实，并得到特征分布外观的支持。当我们将域馈入仅在MNIST上训练的CNN时，我们观察到域之间有相当强的分离，而对于SVHN训练的网络，特征更加混合。这一差异可能解释了为什么我们的方法成功地在SVHN → MNIST场景中通过适应改进了性能（见表2），但在相反方向上则不然（SA在这种情况下也无法执行适应）。从MNIST到SVHN的无监督适应给出了我们方法的一个失败案例：它未能改进未适应模型的性能，后者达到约0.25的准确率（我们不知道有任何无监督DA方法能够执行这种适应）。

**合成标志 → GTSRB。** 总体而言，此设置与Syn Numbers → SVHN实验类似，只是特征分布因类别数量显著增加（43个而不是10个）而更加复杂。对于源域，我们获得了100,000张模拟各种成像条件的合成图像（我们称之为Syn Signs）。在目标域中，我们使用31,367个随机训练样本进行无监督适应，其余用于评估。我们的方法再次实现了显著的性能提升，证明了其适用于合成到真实数据的适应。

作为额外的实验，我们还评估了所提出算法在半监督域适应中的表现，即当额外提供少量标注目标数据时。在这里，我们揭示430个标注样本（每类10个样本）并将其添加到标签预测器的训练集中。图7显示了整个训练过程中验证误差的变化。虽然图表清楚地表明我们的方法在半监督设置中可能是有益的，但半监督设置的彻底验证留待未来的工作。

**Office数据集。** 我们最后在Office数据集上评估我们的方法，该数据集是三个不同域的集合：Amazon、DSLR和Webcam。与之前讨论的数据集不同，Office是小规模的，在最大的域中仅有2817张标注图像分布在31个不同类别中。可用数据的数量对深度模型的成功训练至关重要，因此我们选择对在ImageNet上预训练的CNN进行微调（来自Caffe包的AlexNet，参见Jia et al., 2014），正如一些最近的DA工作所做的那样（Donahue et al., 2014; Tzeng et al., 2014; Hoffman et al., 2013; Long and Wang, 2015）。我们通过使用完全相同的网络架构，用域分类器替换基于域均值的正则化，使我们的方法与Tzeng et al. (2014)更具可比性。

遵循先前的工作，我们评估了方法在三个最常用于评估的迁移任务上的性能。我们的训练协议采用自Gong et al. (2013); Chopra et al. (2013); Long and Wang (2015)，因为在适应过程中我们使用所有可用的标注源样本和未标注目标样本（我们方法的前提是目标域中有丰富的未标注数据）。此外，所有源域数据都用于训练。在此"完全传导式"设置下，我们的方法能够非常显著地改进先前报告的无监督适应最先进准确率（表3），尤其是在最具挑战性的Amazon → Webcam场景中（这两个域具有最大的域偏移）。

**表2：不同源域和目标域的数字图像分类准确率。** MNIST-M对应非均匀背景上的差分混合数字。第一行对应性能下界（即未进行适应）。最后一行对应在目标域数据上使用已知类别标签进行训练（DA性能的上界）。对于两种DA方法（我们的和Fernando et al., 2013），我们显示了在上下界之间覆盖了多少差距（括号内）。在所有五种情况下，我们的方法远优于Fernando et al. (2013)，并覆盖了差距的很大一部分。

**表3：标准Office（Saenko et al., 2010）数据集上不同DA方法的准确率评估。** 所有方法（除SA外）在"完全传导式"协议下评估（一些结果转载自Long and Wang, 2015）。我们的方法（最后一行）优于竞争对手，设定了新的最先进水平。

有趣的是，在所有三个实验中，我们观察到随着训练的进行有轻微的过拟合（目标域的性能下降，而源域的准确率继续提高），但它没有破坏验证准确率。此外，关闭域分类器分支使这种效果更加明显，从中我们得出结论，我们的技术起到了正则化器的作用。

**图7：半监督设置中交通标志分类的结果。** Syn和Real分别表示可用的标注数据（分别100,000张合成图像和430张真实图像）；Adapted表示约31,000张未标注目标域图像用于适应。通过使用标注样本和目标域中大量未标注语料，获得最佳性能。

### 5.3 深度图像描述子用于重识别实验

在本节中，我们讨论所描述的适应方法在行人重识别（re-id）问题中的应用。行人重识别的任务是将来自不同摄像头视角的行人关联起来。更正式地说，它可以定义如下：给定来自不同摄像头的两组图像（探针集和图库集），使得探针集中描绘的每个人在图库集中都有一张图像，对于探针集中每个人的图像，在图库集中找到同一个人的图像。不相交的摄像头视角、不同的光照条件、各种姿态和数据的低质量使得这个问题即使对人类来说也很困难（例如，Liu et al., 2013报告人类性能为Rank1=71.08%）。

与上述讨论的分类问题不同，重识别问题意味着每个图像被映射到一个向量描述子。然后使用描述子之间的距离来匹配来自探针集和图库集的图像。为了评估重识别方法的结果，通常使用累积匹配特征（CMC）曲线。它是rank-k处识别率（召回率）的图，即匹配的图库图像在距离探针图像最近的k个图像内的概率（就描述子距离而言）。

大多数现有工作在包含来自具有相似成像条件的特定摄像头网络图像的数据集内训练描述子映射并评估它们。然而，有几篇论文观察到，所得到的重识别系统的性能在描述子在一个数据集上训练并在另一个数据集上测试时会非常显著地下降。因此，将这种跨域评估作为域适应问题来处理是很自然的，其中每个摄像头网络（数据集）构成一个域。

**图8：来自不同行人重识别数据集的匹配和不匹配的探针-图库图像对。** 三个数据集在我们的实验中被视为不同的域。

最近，几篇论文提出了显著改进的重识别性能（Zhang and Saligrama, 2014; Zhao et al., 2014; Paisitkriangkrai et al., 2015），其中Ma et al. (2015)在跨数据集评估场景中报告了良好结果。目前，深度学习方法（Yi et al., 2014）尚未达到最先进的结果，可能是由于训练集规模有限。因此，域适应代表了改进深度重识别描述子的可行方向。

#### 5.3.1 数据集和协议

遵循Ma et al. (2015)，我们使用PRID（Hirzer et al., 2011）、VIPeR（Gray et al., 2007）、CUHK（Li and Wang, 2013）作为实验的目标数据集。PRID数据集存在两个版本，与Ma et al. (2015)一样，我们使用单次变体。它包含来自摄像头A的385人图像和来自摄像头B的749人图像，其中200人出现在两个摄像头中。VIPeR数据集也包含用两个摄像头拍摄的图像，总共捕获632人，每个人在每个摄像头视角下有一张图像。CUHK数据集由五对摄像头的图像组成，每个人在每个摄像头下有两张图像。我们将仅包含第一对摄像头的此数据集的子集称为CUHK/p1（因为大多数论文使用此子集）。这些数据集的样本见图8。

我们对各种数据集对进行广泛的实验，其中一个数据集作为源域，即它用于以有监督方式训练描述子映射，已知探针和图库图像之间的对应关系。第二个数据集用作目标域，因此该数据集的图像在没有探针-图库对应关系的情况下使用。

更详细地说，当CUHK作为目标域时不使用CUHK/p1，而当CUHK作为源域时使用两种设置（"whole CUHK"和CUHK/p1）。给定PRID作为目标数据集，我们随机选择出现在两个摄像头视角中的100人作为训练集。摄像头A中其他100人的图像用作探针，来自摄像头B的所有图像（不包括训练中使用的那些，总共649张）在测试时用作图库。对于VIPeR，我们使用随机316人进行训练，所有其他人进行测试。对于CUHK，971人被分为485人用于训练和486人用于测试。与Ma et al. (2015)不同，我们使用CUHK第一对摄像头中的所有图像，而不是从每个摄像头视角选择一个人的一张图像。我们还进行了两次实验，将整个CUHK数据集的所有图像作为源域，VIPeR和PRID数据集作为目标域，如原始论文（Yi et al., 2014）中那样。

遵循Yi et al. (2014)，我们用镜像图像增强了我们的数据，在测试时，我们将两张图像之间的相似度分数计算为两张比较图像的四种不同翻转对应分数的均值。在CUHK的情况下，每个人在每个摄像头视角下有4张图像（包括镜像图像），所有16种组合的分数取平均。

#### 5.3.2 CNN架构和训练过程

在我们的实验中，我们使用Yi et al. (2014)（深度度量学习或DML）中描述的Siamese架构在源数据集上学习深度图像描述子。该架构包含两个卷积层（具有 $7 \times 7$ 和 $5 \times 5$ 的滤波器组），后接ReLU和最大池化，以及一个全连接层，输出500维描述子。CNN内有三条并行流用于处理图像的三部分：上部、中部和下部。第一卷积层在三部分之间共享参数，第二卷积层的输出被串联。在训练期间，我们遵循Yi et al. (2014)，计算每个批次内500维特征之间的成对余弦相似度，并对批次内的所有对反向传播损失。

为执行域对抗训练，我们构建了一个DANN架构。特征提取器包括上述讨论的两个卷积层（后接最大池化和ReLU）。此情况下的标签预测器被替换为包括一个全连接层的描述子预测器。域分类器包括两个全连接层，中间表示有500个单元（x→500→1）。

对于描述子预测器中的验证损失函数，我们使用Yi et al. (2014)中定义的二项偏差损失，参数相似：$\alpha = 2$，$\beta = 0.5$，$c = 2$（负对的非对称代价参数）。域分类器与第5.2.2节一样使用逻辑损失训练。

我们使用固定为0.001的学习率和0.9的动量。使用了与第5.2.2节所述类似的适应调度。我们还在第二个最大池化层输出串联之后插入了率为0.5的dropout层。源数据使用128大小的批次，目标数据使用128大小的批次。

#### 5.3.3 重识别数据集结果

**图9：在VIPeR、PRID和CUHK/p1上有和没有域对抗学习的结果。** 在八个域对中，域对抗学习改进了重识别准确率。对于某些域对，改进是显著的。

图9以CMC曲线形式显示了八个数据集对的结果。根据标注问题的难度，我们训练了50,000次迭代（CUHK/p1 → VIPeR, VIPeR → CUHK/p1, PRID → VIPeR）或20,000次迭代（其他五对）。经过足够的迭代次数后，域对抗训练持续改进了重识别的性能。对于涉及PRID数据集的对——该数据集与其他两个数据集更不相似——改进是显著的。总体而言，这证明了域对抗学习在分类问题之外的可应用性。

**图10：在VIPeR → CUHK/p1实验对中，适应对源域和目标域描述子分布的影响，通过t-SNE可视化显示。** VIPeR用绿色表示，CUHK/p1用红色表示。与图像分类情况一样，域对抗学习确保源分布和目标分布之间更紧密的匹配。

图10进一步展示了在VIPeR → CUHK/p1实验中适应对学习到的描述子在源集和目标集中分布的影响，其中域对抗学习再次实现了两个域更好的混合。

## 6. 结论

本文提出了一种前馈神经网络域适应的新方法，允许基于源域中大量标注数据和目标域中大量未标注数据进行大规模训练。与许多先前的浅层和深层DA技术类似，适应是通过配准跨域的特征分布实现的。然而，与先前方法不同，配准是通过标准反向传播训练完成的。

该方法受到Ben-David et al. (2006, 2010)的域适应理论的启发和支持。DANN背后的主要思想是促使网络隐藏层学习一种表示，该表示对源样本标签具有预测性，但对输入的域（源域或目标域）不包含信息。我们在浅层和深层前馈架构中都实现了这一新方法。后者允许通过引入一个简单的梯度反转层，在几乎任何深度学习包中简单实现。我们已经证明我们的方法是灵活的，并在域适应的各种基准上取得了最先进的结果，即情感分析和图像分类任务。

我们方法的一个便利方面是域适应组件可以添加到几乎任何可通过反向传播训练的神经网络架构中。为此，我们实验证明该方法不仅限于分类任务，还可以用于其他前馈架构，例如用于行人重识别的描述子学习。

## 致谢

本工作得到国家科学与工程研究委员会（NSERC）发现基金262067和0122405以及俄罗斯科学与教育部基金RFMEFI57914X0071的支持。计算在拉瓦尔大学Colosse超级计算机集群上执行，由Calcul Quebec和Compute Canada主持。Colosse的运营由NSERC、加拿大创新基金会（CFI）、NanoQuebec和魁北克自然与技术研究基金（FRQNT）资助。我们还要感谢莫斯科国立罗蒙诺索夫大学计算数学与控制论学院图形与多媒体实验室提供合成道路标志数据集。

## 参考文献

[1] Hana Ajakan, Pascal Germain, Hugo Larochelle, Francois Laviolette, and Mario Marchand. Domain-adversarial neural networks. *NIPS 2014 Workshop on Transfer and Multi-task Learning: Theory Meets Practice*, 2014. URL http://arxiv.org/abs/1412.4446.

[2] Pablo Arbelaez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. *IEEE Transaction Pattern Analysis and Machine Intelligence*, 33, 2011.

[3] Artem Babenko, Anton Slesarev, Alexander Chigorin, and Victor S. Lempitsky. Neural codes for image retrieval. In *ECCV*, pages 584-599, 2014.

[4] Mahsa Baktashmotlagh, Mehrtash Tafazzoli Harandi, Brian C. Lovell, and Mathieu Salzmann. Unsupervised domain adaptation by domain invariant projection. In *ICCV*, pages 769-776, 2013.

[5] Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. In *NIPS*, pages 137-144, 2006.

[6] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. *Machine Learning*, 79(1-2):151-175, 2010.

[7] John Blitzer, Ryan T. McDonald, and Fernando Pereira. Domain adaptation with structural correspondence learning. In *Conference on Empirical Methods in Natural Language Processing*, pages 120-128, 2006.

[8] Karsten M. Borgwardt, Arthur Gretton, Malte J. Rasch, Hans-Peter Kriegel, Bernhard Scholkopf, and Alexander J. Smola. Integrating structured biological data by kernel maximum mean discrepancy. In *ISMB*, pages 49-57, 2006.

[9] Lorenzo Bruzzone and Mattia Marconcini. Domain adaptation problems: A DASVM classification technique and a circular validation strategy. *IEEE Transaction Pattern Analysis and Machine Intelligence*, 32(5):770-787, 2010.

[10] Minmin Chen, Zhixiang Eddie Xu, Kilian Q. Weinberger, and Fei Sha. Marginalized denoising autoencoders for domain adaptation. In *ICML*, pages 767-774, 2012.

[11] Qiang Chen, Junshi Huang, Rogerio Feris, Lisa M. Brown, Jian Dong, and Shuicheng Yan. Deep domain adaptation for describing people based on fine-grained clothing attributes. In *CVPR*, June 2015.

[12] S. Chopra, S. Balakrishnan, and R. Gopalan. DLID: Deep learning for domain adaptation by interpolating between domains. In *ICML Workshop on Challenges in Representation Learning*, 2013.

[13] Dan Ciresan, Ueli Meier, Jonathan Masci, and Jurgen Schmidhuber. Multi-column deep neural network for traffic sign classification. *Neural Networks*, 32:333-338, 2012.

[14] Corinna Cortes and Mehryar Mohri. Domain adaptation and sample bias correction theory and algorithm for regression. *Theor. Comput. Sci.*, 519:103-126, 2014.

[15] Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. DeCAF: A deep convolutional activation feature for generic visual recognition. In *ICML*, 2014.

[16] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Technical report, EECS Department, University of California, Berkeley, Mar 2010.

[17] Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, and Chih-Jen Lin. LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9:1871-1874, 2008.

[18] Basura Fernando, Amaury Habrard, Marc Sebban, and Tinne Tuytelaars. Unsupervised visual domain adaptation using subspace alignment. In *ICCV*, 2013.

[19] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In *ICML*, pages 325-333, 2015. URL http://jmlr.org/proceedings/papers/v37/ganin15.html.

[20] Pascal Germain, Amaury Habrard, Francois Laviolette, and Emilie Morvant. A PAC-Bayesian approach for domain adaptation with specialization to linear classifiers. In *ICML*, pages 738-746, 2013.

[21] Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Domain adaptation for large-scale sentiment classification: A deep learning approach. In *ICML*, pages 513-520, 2011.

[22] Boqing Gong, Yuan Shi, Fei Sha, and Kristen Grauman. Geodesic flow kernel for unsupervised domain adaptation. In *CVPR*, pages 2066-2073, 2012.

[23] Boqing Gong, Kristen Grauman, and Fei Sha. Connecting the dots with landmarks: Discriminatively learning domain-invariant features for unsupervised domain adaptation. In *ICML*, pages 222-230, 2013.

[24] Shaogang Gong, Marco Cristani, Shuicheng Yan, and Chen Change Loy. *Person Re-identification*. Springer, 2014.

[25] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *NIPS*, 2014.

[26] Raghuraman Gopalan, Ruonan Li, and Rama Chellappa. Domain adaptation for object recognition: An unsupervised approach. In *ICCV*, pages 999-1006, 2011.

[27] Doug Gray, Shane Brennan, and Hai Tao. Evaluating appearance models for recognition, reacquisition, and tracking. In *IEEE International Workshop on Performance Evaluation for Tracking and Surveillance*, Rio de Janeiro, 2007.

[28] Martin Hirzer, Csaba Beleznai, Peter M. Roth, and Horst Bischof. Person re-identification by descriptive and discriminative classification. In *SCIA*, 2011.

[29] Judy Hoffman, Eric Tzeng, Jeff Donahue, Yangqing Jia, Kate Saenko, and Trevor Darrell. One-shot adaptation of supervised deep convolutional models. *CoRR*, abs/1312.6204, 2013. URL http://arxiv.org/abs/1312.6204.

[30] Fei Huang and Alexander Yates. Biased representation learning for domain adaptation. In *Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning*, pages 1313-1323, 2012.

[31] Jiayuan Huang, Alexander J. Smola, Arthur Gretton, Karsten M. Borgwardt, and Bernhard Scholkopf. Correcting sample selection bias by unlabeled data. In *NIPS*, pages 601-608, 2006.

[32] Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional architecture for fast feature embedding. *CoRR*, abs/1408.5093, 2014.

[33] Daniel Kifer, Shai Ben-David, and Johannes Gehrke. Detecting change in data streams. In *Very Large Data Bases*, pages 180-191, 2004.

[34] Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. ImageNet classification with deep convolutional neural networks. In *NIPS*, pages 1097-1105, 2012.

[35] Alexandre Lacoste, Francois Laviolette, and Mario Marchand. Bayesian comparison of machine learning algorithms on single and multiple datasets. In *AISTATS*, pages 665-675, 2012.

[36] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278-2324, November 1998.

[37] Wei Li and Xiaogang Wang. Locally aligned feature transforms across views. In *CVPR*, pages 3594-3601, 2013.

[38] Yujia Li, Kevin Swersky, and Richard Zemel. Unsupervised domain adaptation by domain invariant projection. In *NIPS 2014 Workshop on Transfer and Multitask Learning*, 2014.

[39] Joerg Liebelt and Cordelia Schmid. Multi-view object class detection with a 3D geometric model. In *CVPR*, 2010.

[40] Chunxiao Liu, Chen Change Loy, Shaogang Gong, and Guijin Wang. POP: Person re-identification post-rank optimisation. In *ICCV*, pages 441-448, 2013.

[41] Mingsheng Long and Jianmin Wang. Learning transferable features with deep adaptation networks. *CoRR*, abs/1502.02791, 2015.

[42] Andy Jinhua Ma, Jiawei Li, Pong C. Yuen, and Ping Li. Cross-domain person reidentification using domain adaptation ranking SVMs. *IEEE Transactions on Image Processing*, 24(5):1599-1613, 2015.

[43] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. In *COLT*, 2009a.

[44] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Multiple source adaptation and the Renyi divergence. In *UAI*, pages 367-374, 2009b.

[45] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*, 2011.

[46] M. Oquab, L. Bottou, I. Laptev, and J. Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In *CVPR*, 2014.

[47] Sakrapee Paisitkriangkrai, Chunhua Shen, and Anton van den Hengel. Learning to rank in person re-identification with metric ensembles. *CoRR*, abs/1503.01543, 2015. URL http://arxiv.org/abs/1503.01543.

[48] Sinno Jialin Pan, Ivor W. Tsang, James T. Kwok, and Qiang Yang. Domain adaptation via transfer component analysis. *IEEE Transactions on Neural Networks*, 22(2):199-210, 2011.

[49] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In *ECCV*, pages 213-226, 2010.

[50] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. *The Journal of Machine Learning Research*, 15(1):1929-1958, 2014.

[51] Michael Stark, Michael Goesele, and Bernt Schiele. Back to the future: Learning shape models from 3D CAD data. In *BMVC*, pages 1-11, 2010.

[52] Baochen Sun and Kate Saenko. From virtual to reality: Fast adaptation of virtual object detectors to real domains. In *BMVC*, 2014.

[53] Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, and Trevor Darrell. Deep domain confusion: Maximizing for domain invariance. *CoRR*, abs/1412.3474, 2014. URL http://arxiv.org/abs/1412.3474.

[54] Laurens van der Maaten. Barnes-Hut-SNE. *CoRR*, abs/1301.3342, 2013. URL http://arxiv.org/abs/1301.3342.

[55] David Vazquez, Antonio Manuel Lopez, Javier Marin, Daniel Ponsa, and David Geronimo Gomez. Virtual and real world adaptation for pedestrian detection. *IEEE Transaction Pattern Analysis and Machine Intelligence*, 36(4):797-809, 2014.

[56] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In *ICML*, pages 1096-1103, 2008.

[57] Dong Yi, Zhen Lei, and Stan Z. Li. Deep metric learning for practical person re-identification. *CoRR*, abs/1407.4979, 2014. URL http://arxiv.org/abs/1407.4979.

[58] Matthew D. Zeiler. ADADELTA: an adaptive learning rate method. *CoRR*, abs/1212.5701, 2012. URL http://arxiv.org/abs/1212.5701.

[59] Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. *CoRR*, abs/1311.2901, 2013. URL http://arxiv.org/abs/1311.2901.

[60] Ziming Zhang and Venkatesh Saligrama. Person re-identification via structured prediction. *CoRR*, abs/1406.4444, 2014. URL http://arxiv.org/abs/1406.4444.

[61] Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Person re-identification by saliency learning. *CoRR*, abs/1412.1908, 2014. URL http://arxiv.org/abs/1412.1908.

[62] Erheng Zhong, Wei Fan, Qiang Yang, Olivier Verscheure, and Jiangtao Ren. Cross validation framework to choose amongst models and datasets for transfer learning. In *Machine Learning and Knowledge Discovery in Databases*, pages 547-562. Springer, 2010.
