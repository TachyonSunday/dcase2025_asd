"""
DANN (Domain Adversarial Neural Network) 域对抗自编码器。
在 ConvAE 瓶颈层附加域分类器, 通过梯度反转层实现对抗训练,
迫使编码器学习域不变的特征表示, 提升跨域泛化能力。
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import yaml

from src.models.conv_ae import ConvAE, ConvAEEncoder, ConvAEDecoder
from src.models.grad_reverse import GradientReversalLayer


class DomainClassifier(nn.Module):
    """
    域分类器 — 从隐向量预测样本所属的域 (机器工况/环境)。

    参数
    ----
    input_dim : int
        隐向量维度。
    hidden_dim : int
        隐藏层维度。
    num_domains : int
        域的数量。
    dropout : float
        Dropout 概率, 防止域分类器过拟合。
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_domains: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        z : torch.Tensor
            隐向量, shape=(B, latent_dim)。

        返回
        ----
        torch.Tensor
            域分类 logits, shape=(B, num_domains)。
        """
        return self.classifier(z)


class DANNAutoEncoder(nn.Module):
    """
    DANN 域对抗自编码器 — ConvAE + 梯度反转层 + 域分类器。

    架构::

        输入 x ──→ [Encoder] ──→ z ──→ [Decoder] ──→ x_recon (重构损失)
                              │
                              ├──→ [GRL] ──→ [DomainClassifier] ──→ domain_logits (域对抗损失)

    GRL 在前向时恒等, 反向时将域分类损失对 z 的梯度取反,
    使 encoder 学习"欺骗"域分类器, 从而产出域不变的特征。

    使用方式::

        model = DANNAutoEncoder.from_config("config.yaml")
        x = torch.randn(16, 1, 128, 64)
        domain_ids = torch.randint(0, 3, (16,))
        (x_recon, z), domain_logits = model(x)
        # 联合优化: recon_loss + λ * domain_loss

    参数
    ----
    conv_ae : ConvAE
        预训练或随机初始化的 ConvAE 基础模型。
    num_domains : int
        域的数量。
    domain_hidden : int
        域分类器隐藏层维度。
    lambda_init : float
        对抗损失初始权重。
    """

    def __init__(
        self,
        conv_ae: ConvAE,
        num_domains: int = 3,
        domain_hidden: int = 64,
        lambda_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = conv_ae.encoder
        self.decoder = conv_ae.decoder

        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_init=lambda_init)

        # 域分类器
        latent_dim = conv_ae.encoder._latent_dim
        self.domain_classifier = DomainClassifier(
            input_dim=latent_dim,
            hidden_dim=domain_hidden,
            num_domains=num_domains,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入频谱图, shape=(B, 1, n_mels, window_size)。

        返回
        ----
        tuple[tuple[Tensor, Tensor], Tensor]
            ((x_recon, z), domain_logits)
            - x_recon: 重建频谱图
            - z: 隐向量
            - domain_logits: 域分类 logits
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)

        # 域分类分支 (经过 GRL 反转梯度)
        z_reversed = self.grl(z)
        domain_logits = self.domain_classifier(z_reversed)

        return (x_recon, z), domain_logits

    def set_lambda(self, value: float) -> None:
        """动态设置对抗损失权重 λ。"""
        self.grl.set_lambda(value)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码, 返回隐向量。"""
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """仅重建。"""
        z = self.encoder(x)
        return self.decoder(z)

    def bind(self, sample_input: torch.Tensor) -> None:
        """
        使用样本输入完成惰性初始化 (确保编解码器已绑定)。
        必须在首次训练前调用一次。
        """
        with torch.no_grad():
            self.encoder._init_fc(sample_input)
            self.decoder._init_fc(sample_input, self.encoder.conv)
            self.decoder._latent_dim_val = self.encoder._latent_dim

    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        conv_ae: Optional[ConvAE] = None,
    ) -> "DANNAutoEncoder":
        """
        从 YAML 配置文件创建 DANNAutoEncoder。

        参数
        ----
        config_path : str
            YAML 配置文件路径。
        conv_ae : ConvAE, 可选
            已有的 ConvAE 模型 (可用预训练权重初始化), 若为 None 则随机初始化。

        返回
        ----
        DANNAutoEncoder
            已初始化的 DANN 模型。
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if conv_ae is None:
            conv_ae = ConvAE.from_config(config_path)

        dann_cfg = config["dann"]
        return cls(
            conv_ae=conv_ae,
            num_domains=dann_cfg["num_domains"],
            domain_hidden=dann_cfg.get("domain_hidden", 64),
            lambda_init=dann_cfg.get("lambda_init", 0.0),
        )

    @classmethod
    def from_conv_ae_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "config.yaml",
    ) -> "DANNAutoEncoder":
        """
        从 ConvAE 检查点加载预训练权重并构建 DANN 模型。

        参数
        ----
        checkpoint_path : str
            ConvAE 检查点文件路径。
        config_path : str
            YAML 配置文件路径。

        返回
        ----
        DANNAutoEncoder
            使用预训练编码器/解码器初始化的 DANN 模型。
        """
        conv_ae = ConvAE.from_config(config_path)
        # 先用一个 dummy 输入完成绑定
        dummy = torch.randn(1, 1, 128, 64)
        conv_ae.bind(dummy)
        # 加载 ConvAE 权重
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        conv_ae.load_state_dict(ckpt["model_state_dict"])
        return cls.from_config(config_path, conv_ae=conv_ae)
