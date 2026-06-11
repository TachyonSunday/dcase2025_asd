# 堆叠卷积自编码器用于分层特征提取

> Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

**作者**：Jonathan Masci, Ueli Meier, Dan Cireşan, and Jürgen Schmidhuber

**机构**：Istituto Dalle Molle di Studi sull'Intelligenza Artificiale (IDSIA), Lugano, Switzerland

**联系方式**：{jonathan, ueli, dan, juergen}@idsia.ch

*发表于：T. Honkela et al. (Eds.): ICANN 2011, Part I, LNCS 6791, pp. 52–59, 2011. ©Springer-Verlag Berlin Heidelberg 2011*

---

## 摘要 (Abstract)

我们提出了一种新颖的卷积自编码器（CAE），用于无监督特征学习。由CAE堆叠而成的结构构成一个卷积神经网络（CNN）。每个CAE使用常规在线梯度下降进行训练，无需额外的正则化项。最大池化层对于学习与先前方法所发现的相一致的、具有生物学合理性的特征至关重要。用训练好的CAE堆叠的滤波器初始化CNN，在数字识别（MNIST）和物体识别（CIFAR10）基准数据集上均取得了优越的性能。

**关键词**：卷积神经网络，自编码器，无监督学习，分类。

---

## 1 引言 (Introduction)

无监督学习方法的主要目的是从无标签数据中提取普遍有用的特征，检测并消除输入冗余，并仅保留数据在鲁棒且具有判别力的表示中的本质方面。无监督方法已被常规应用于许多科学和工业应用中。在神经网络架构的背景下，无监督层可以逐层堆叠以构建深层层次结构[7]。输入层的激活被馈送至第一层，第一层再馈送至下一层，以此类推，覆盖层次结构中的所有层。深度架构可以通过无监督的逐层方式进行训练，之后再通过反向传播进行微调，从而成为分类器[9]。无监督初始化倾向于避免局部极小值，并提高网络的性能稳定性[6]。

大多数方法基于编码器-解码器范式，例如[20]。输入首先被变换到一个通常维度更低的空间（编码器），然后被扩展以重现原始数据（解码器）。一旦一个层训练完成，其编码将馈送至下一层，以更好地建模输入中高度非线性的依赖关系。使用这一范式的方法包括：低复杂度编码解码机（LOCOCODE）[10] 的堆叠、可预测性最小化层[23, 24] 的堆叠、受限玻尔兹曼机（RBM）[8] 的堆叠、自编码器[20] 的堆叠以及基于能量的模型[15]。

在视觉物体识别中，CNN [1, 3, 4, 14, 26] 通常表现出色。与基于图像块的方法[19]不同，CNN在其潜在的高层特征表示中保留了输入的邻域关系和空间局部性。虽然常见的全连接深度架构在处理现实尺寸的高维图像时，在计算复杂度方面难以扩展，但CNN能够很好地扩展，因为描述其共享权重的自由参数数量不依赖于输入维度[16, 18, 28]。

本文引入了卷积自编码器，一种能够良好扩展至高维输入的分层无监督特征提取器。它使用普通的随机梯度下降学习非平凡的特征，并发现能够避免几乎所有深度学习问题中大量高度非凸目标函数的不同局部极小值的良好CNN初始化。

---

## 2 预备知识 (Preliminaries)

### 2.1 自编码器 (Auto-Encoder)

我们回顾自编码器模型的基本原理，例如[2]。一个自编码器接受输入 $\mathbf{x} \in \mathbb{R}^d$，首先使用形如 $\mathbf{h} = f_\theta = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$ 的确定性函数将其映射到潜在表示 $\mathbf{h} \in \mathbb{R}^{d'}$，其中参数 $\theta = \{\mathbf{W}, \mathbf{b}\}$。然后，该"编码"用于通过 $f$ 的反向映射重建输入：$\mathbf{y} = f_{\theta'}(\mathbf{h}) = \sigma(\mathbf{W}'\mathbf{h} + \mathbf{b}')$，其中 $\theta' = \{\mathbf{W}', \mathbf{b}'\}$。两组参数通常被约束为 $\mathbf{W}' = \mathbf{W}^T$ 的形式，即使用相同的权重对输入进行编码和对潜在表示进行解码。每个训练样本 $\mathbf{x}_i$ 随后被映射到其编码 $\mathbf{h}_i$ 及其重建 $\mathbf{y}_i$。通过最小化训练集 $D_n = \{(\mathbf{x}_0, \mathbf{t}_0), ..., (\mathbf{x}_n, \mathbf{t}_n)\}$ 上的适当代价函数来优化参数。

### 2.2 去噪自编码器 (Denoising Auto-Encoder)

在没有任何额外约束的情况下，传统自编码器会学习恒等映射。这个问题可以通过使用概率RBM方法、稀疏编码、或试图重建含噪输入的去噪自编码器（DA）[27] 来避免。后者的性能与RBM相当甚至更好[2]。训练涉及从部分损坏的输入中重建干净的输入。输入 $\mathbf{x}$ 通过添加根据输入图像特征分布的可变噪声量 $v$ 而变为损坏的输入 $\bar{\mathbf{x}}$。常见的选择包括用于黑白图像的二项噪声（开启或关闭像素），或用于彩色图像的不相关高斯噪声。参数 $v$ 表示允许的损坏百分比。自编码器被训练为通过首先找到潜在表示 $\mathbf{h} = f_\theta(\bar{\mathbf{x}}) = \sigma(\mathbf{W}\bar{\mathbf{x}} + \mathbf{b})$，然后从中重建原始输入 $\mathbf{y} = f_{\theta'}(\mathbf{h}) = \sigma(\mathbf{W}'\mathbf{h} + \mathbf{b}')$ 来对输入进行去噪。

### 2.3 卷积神经网络 (Convolutional Neural Networks)

CNN是层次化模型，其卷积层与下采样层交替出现，这让人联想到初级视觉皮层中的简单细胞和复杂细胞[11]。网络架构由三个基本构建块组成，可以根据需要堆叠和组合。这三个构建块是：卷积层、最大池化层和分类层[14]。CNN是用于有监督图像分类的最成功模型之一，在许多基准测试中达到了最先进的水平[13, 14]。

---

## 3 卷积自编码器 (Convolutional Auto-Encoder, CAE)

全连接的AE和DAE都忽略了图像的二维结构。这不仅在处理现实尺寸的输入时是一个问题，而且还会在参数中引入冗余，迫使每个特征都是全局的（即跨越整个视觉场）。然而，最成功的模型[17, 25]在视觉和物体识别中采用的趋势是发现那些在输入各处重复出现的局部化特征。CAE与传统AE的不同之处在于，其权重在输入的所有位置上共享，从而保留了空间局部性。因此，重建是由基于潜在编码的基本图像块进行线性组合得到的。

CAE的架构在直观上与第 2.2 节描述的类似，不同之处在于权重是共享的。对于单通道输入 $\mathbf{x}$，第 $k$ 个特征图的潜在表示由下式给出：

$$
\mathbf{h}^k = \sigma(\mathbf{x} * \mathbf{W}^k + b^k) \tag{1}
$$

其中偏置被广播到整个特征图，$\sigma$ 是一个激活函数（我们在所有实验中使用了缩放双曲正切函数），$*$ 表示二维卷积。每个潜在特征图仅使用一个偏置，因为我们希望每个滤波器专门化于整个输入的特征（每个像素一个偏置会引入过多的自由度）。重建通过下式获得：

$$
\mathbf{y} = \sigma\left(\sum_{k \in H} \mathbf{h}^k * \tilde{\mathbf{W}}^k + c\right) \tag{2}
$$

其中每个输入通道同样仅有一个偏置 $c$。$H$ 标识潜在特征图组；$\tilde{\mathbf{W}}$ 标识权重在两个维度上的翻转操作。公式 (1) 和 (2) 中的二维卷积由上下文确定。一个 $m \times m$ 的矩阵与一个 $n \times n$ 的矩阵进行卷积，可能产生一个 $(m + n - 1) \times (m + n - 1)$ 的矩阵（全卷积），或一个 $(m - n + 1) \times (m - n + 1)$ 的矩阵（有效卷积）。要最小化的代价函数是均方误差（MSE）：

$$
E(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (\mathbf{x}_i - \mathbf{y}_i)^2. \tag{3}
$$

与标准网络一样，反向传播算法被应用于计算误差函数关于参数的梯度。这可以通过卷积运算使用以下公式轻松获得：

$$
\frac{\partial E(\theta)}{\partial \mathbf{W}^k} = \mathbf{x} * \delta \mathbf{h}^k + \tilde{\mathbf{h}}^k * \delta \mathbf{y}. \tag{4}
$$

$\delta \mathbf{h}$ 和 $\delta \mathbf{y}$ 分别是隐藏状态和重建的delta。然后使用随机梯度下降更新权重。

### 3.1 最大池化 (Max-Pooling)

对于一般的层次化网络，尤其是CNN，通常引入最大池化层[22]以获得平移不变表示。最大池化以常数因子对潜在表示进行下采样，通常取非重叠子区域的最大值。这有助于提高滤波器的选择性，因为潜在表示中每个神经元的激活由特征与感受野输入区域之间的"匹配"程度决定。最大池化最初仅用于全监督的前馈架构。

在这里，我们引入了一个最大池化层，它通过擦除非重叠子区域中的所有非最大值来在隐藏表示上引入稀疏性。这迫使特征检测器变得更具广泛适用性，避免了诸如仅有一个权重为"开"（恒等函数）之类的平凡解。在重建阶段，这种稀疏的潜在编码减少了参与每个像素解码的滤波器平均数量，迫使滤波器更加通用。因此，有了最大池化层，显然就不再需要对隐藏单元和/或权重进行 $L_1$ 和/或 $L_2$ 正则化。

### 3.2 堆叠卷积自编码器 (Stacked Convolutional Auto-Encoders, CAES)

多个AE可以堆叠形成深层层次结构，例如[27]。每一层接受其下层潜在表示的输出作为输入。与深度信念网络一样，无监督预训练可以以贪婪、逐层的方式进行。之后，可以使用反向传播对权重进行微调，或者将顶层的激活用作SVM或其他分类器的特征向量。类似地，CAE堆叠（CAES）可用于在有监督训练阶段之前初始化具有相同拓扑结构的CNN。

---

## 4 实验 (Experiments)

我们首先通过可视化各种CAE在不同设置下训练的滤波器来开始实验，这些CAE在数字数据集（MNIST [14]）和自然图像（CIFAR10 [13]）上进行训练。在图1中，我们比较了四个具有相同拓扑结构但训练方式不同的CAE的20个 $7 \times 7$ 滤波器（在MNIST上学习）。第一个在原始数字图像上训练（a），第二个在添加了50%二项噪声的含噪输入上训练（b），第三个具有额外的 $2 \times 2$ 最大池化层（c），第四个在含噪输入（30%二项噪声）上训练并具有 $2 \times 2$ 最大池化层（d）。我们在结合最大池化层时添加30%的噪声，以避免丢失过多相关信息。没有任何额外约束的CAE（a）学习到平凡解。只有使用最大池化层训练的CAE才会涌现出有趣且具有生物学合理性的滤波器。加入额外的噪声后，滤波器变得更加局部化。对于这个特定示例，最大池化产生了视觉上最漂亮的滤波器；其他方法的滤波器没有良好定义的形状。最大池化层是一种优雅的方式，可以强制执行处理卷积架构过完备表示所需的稀疏编码。

*[图1. 在MNIST上学习到的第一层滤波器的随机子集，用于比较噪声和池化的效果。（a）无最大池化，0%噪声；（b）无最大池化，50%噪声；（c）$2 \times 2$ 最大池化；（d）$2 \times 2$ 最大池化，30%噪声。]*

*[图2. 在CIFAR10上学习到的第一层滤波器的随机子集，用于比较噪声和池化的效果（彩色最佳）。（a）无池化，0%噪声；（b）无池化，50%噪声；（c）$2 \times 2$ 池化，0%噪声；（d）$2 \times 2$ 池化，50%噪声。]*

在处理自然彩色图像时，将高斯噪声而非二项噪声添加到去噪CAE的输入中。我们在CIFAR10上重复上述实验。对应的滤波器如图2所示。最大池化层的影响是显著的（c），而添加噪声（b）除权重幅度外几乎没有视觉效果（d）。与MNIST一样，只有最大池化层能保证令人信服的解，这表明最大池化是必不可少的。它似乎至少部分解决了通常在使用梯度下降训练自编码器时出现的问题。我们方法的另一个值得欢迎的方面是，除了最大池化核大小之外，无需通过反复尝试或耗时的交叉验证来设置额外的参数。

### 4.1 用训练好的CAES权重初始化CNN (Initializing a CNN with Trained CAES Weights)

上一节中发现的滤波器不仅本身很有趣，而且在生物学上也是合理的。我们现在训练一个CAES，并用它来初始化具有相同拓扑结构的CNN，以针对分类任务进行微调。这已经被证明可以缓解训练深度标准MLP时的常见问题[6]。我们通过与随机初始化的CNN进行比较来研究无监督预训练的好处。

我们从公认的MNIST基准[14]开始，以展示预训练对于不同大小子集的效果。表1中的分类结果基于完整的测试集和指定数量的训练样本。该网络有6个隐藏层：1）每个输入通道具有100个 $5 \times 5$ 滤波器的卷积层；2）$2 \times 2$ 最大池化层；3）每个特征图具有150个 $5 \times 5$ 滤波器的卷积层；4）$2 \times 2$ 最大池化层；5）具有200个 $3 \times 3$ 大小特征图的卷积层；6）具有300个隐藏神经元的全连接层。输出层具有softmax激活函数，每个类别一个神经元。学习率在训练过程中退火。没有对MNIST应用变形来增加"虚拟"训练样本数量，否则会减弱无监督预训练对这个已经被认为几乎已解决的问题的影响。我们还在CIFAR10上测试了我们的模型。该数据集具有挑战性，因为其 $32 \times 32$ 像素的输入模式传达的信息很少。许多方法在此数据集上进行了测试。最成功的方法使用归一化技术来消除像素间的二阶信息[5, 12]，或使用深度CNN [3]。我们的方法即使在仅使用"原始"像素信息进行训练时也能提供良好的识别率。

| 训练样本数 | 1k | 10k | 50k |
|---|---|---|---|
| CAE [%] | 7.23 | 1.88 | 0.71 |
| CNN [%] | 7.63 | 2.21 | 0.79 |
| K-means (4k 特征) [5]$^a$ | - | - | 0.88 |

> $^a$ 我们使用作者提供的代码进行了此实验。

**表1.** 使用完整数据集的各种子集的MNIST分类结果。

| 训练样本数 | 1k | 10k | 50k |
|---|---|---|---|
| CAE [%] | 52.30 | 34.35 | 21.80 |
| CNN [%] | 55.52 | 35.23 | 22.50 |
| Mean-cov. RBM [21] | - | - | 29.00 |
| Conv. RBM [12] | - | - | 21.10 |
| K-means (4k 特征) [5] | - | - | 20.40 |

**表2.** 使用完整数据集的各种子集的CIFAR10分类结果；与其他无监督方法的比较。

我们仅在有监督微调时添加5%的平移增强，并复用MNIST CNN架构，不同之处在于输入层有三个图，对应于每个颜色通道。结果如表2所示。据我们所知，在CIFAR10上，我们在任何使用未白化数据训练的无监督架构中取得了迄今为止最好的结果。使用原始数据使系统完全在线，而且无需收集整个训练集的统计信息。相对于随机初始化CNN的性能提升比MNIST更大，因为该问题要困难得多，网络从无监督预训练中获益更多。

---

## 5 结论 (Conclusion)

我们引入了卷积自编码器，一种用于分层特征提取的无监督方法。它学习具有生物学合理性的滤波器。CNN可以通过CAE堆叠进行初始化。虽然CAE的过完备隐藏表示使得学习比标准自编码器更加困难，但如果我们使用最大池化层——一种强制执行稀疏编码的优雅方式，且无需任何需要反复尝试设置的正则化参数——就能涌现出良好的滤波器。预训练CNN的性能倾向于略微但一致地优于随机初始化的网络。我们的CIFAR10结果是在原始数据上训练的任何无监督方法中的最佳结果，并且接近该基准测试上已发表的最佳结果。

---

## 参考文献 (References)

1. Behnke, S.: Hierarchical Neural Networks for Image Interpretation. LNCS, vol. 2766, pp. 1–13. Springer, Heidelberg (2003)

2. Bengio, Y., Lamblin, P., Popovici, D., Larochelle, H.: Greedy layer-wise training of deep networks. In: Neural Information Processing Systems, NIPS (2007)

3. Cireşan, D.C., Meier, U., Masci, J., Gambardella, L.M., Schmidhuber, J.: High-Performance Neural Networks for Visual Object Classification. ArXiv e-prints, arXiv:1102.0183v1 (cs.AI) (February 2011)

4. Ciresan, D.C., Meier, U., Masci, J., Schmidhuber, J.: Flexible, high performance convolutional neural networks for image classification. In: International Joint Conference on Artificial Intelligence, IJCAI (to appear 2011)

5. Coates, A., Lee, H., Ng, A.: An analysis of single-layer networks in unsupervised feature learning. Advances in Neural Information Processing Systems (2010)

6. Erhan, D., Bengio, Y., Courville, A., Manzagol, P.A., Vincent, P.: Why Does Unsupervised Pre-training Help Deep Learning? Journal of Machine Learning Research 11, 625–660 (2010)

7. Fukushima, K.: Neocognitron: A self-organizing neural network for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics 36(4), 193–202 (1980)

8. Hinton, G.E.: Training products of experts by minimizing contrastive divergence. Neural Comp. 14(8), 1771–1800 (2002)

9. Hinton, G.E., Osindero, S., Teh, Y.W.: A fast learning algorithm for deep belief nets. Neural Computation (2006)

10. Hochreiter, S., Schmidhuber, J.: Feature extraction through LOCOCODE. Neural Computation 11(3), 679–714 (1999)

11. Hubel, D.H., Wiesel, T.N.: Receptive fields and functional architecture of monkey striate cortex. The Journal of Physiology 195(1), 215–243 (1968)

12. Krizhevsky, A.: Convolutional deep belief networks on CIFAR-2010 (2010)

13. Krizhevsky, A.: Learning multiple layers of features from tiny images. Master's thesis, Computer Science Department, University of Toronto (2009)

14. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11), 2278–2324 (1998)

15. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F.: A tutorial on energy-based learning. In: Bakir, G., Hofman, T., Schölkopf, B., Smola, A., Taskar, B. (eds.) Predicting Structured Data. MIT Press, Cambridge (2006)

16. Lee, H., Grosse, R., Ranganath, R., Ng, A.Y.: Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In: Proceedings of the 26th International Conference on Machine Learning, pp. 609–616 (2009)

17. Lowe, D.: Object recognition from local scale-invariant features. In: The Proceedings of the Seventh IEEE International Conference on Computer Vision, vol. 2, pp. 1150–1157 (1999)

18. Norouzi, M., Ranjbar, M., Mori, G.: Stacks of convolutional Restricted Boltzmann Machines for shift-invariant feature learning. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2735–2742 (June 2009)

19. Ranzato, M., Boureau, Y., LeCun, Y.: Sparse feature learning for deep belief networks. In: Advances in Neural Information Processing Systems, NIPS 2007 (2007)

20. Ranzato, M., Fu Jie Huang, Y.L.B., LeCun, Y.: Unsupervised learning of invariant feature hierarchies with applications to object recognition. In: Proc. of Computer Vision and Pattern Recognition Conference (2007)

21. Ranzato, M., Hinton, G.E.: Modeling pixel means and covariances using factorized third-order boltzmann machines. In: Proc. of Computer Vision and Pattern Recognition Conference, CVPR 2010 (2010)

22. Scherer, D., Müller, A., Behnke, S.: Evaluation of pooling operations in convolutional architectures for object recognition. In: International Conference on Artificial Neural Networks (2010)

23. Schmidhuber, J.: Learning factorial codes by predictability minimization. Neural Computation 4(6), 863–879 (1992)

24. Schmidhuber, J., Eldracher, M., Foltin, B.: Semilinear predictability minimization produces well-known feature detectors. Neural Computation 8(4), 773–786 (1996)

25. Serre, T., Wolf, L., Poggio, T.: Object recognition with features inspired by visual cortex. In: Proc. of Computer Vision and Pattern Recognition Conference (2007)

26. Simard, P., Steinkraus, D., Platt, J.: Best practices for convolutional neural networks applied to visual document analysis. In: Seventh International Conference on Document Analysis and Recognition, pp. 958–963 (2003)

27. Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.A.: Extracting and Composing Robust Features with Denoising Autoencoders. In: Neural Information Processing Systems, NIPS (2008)

28. Zeiler, M.D., Krishnan, D., Taylor, G.W., Fergus, R.: Deconvolutional Networks. In: Proc. Computer Vision and Pattern Recognition Conference, CVPR 2010 (2010)
