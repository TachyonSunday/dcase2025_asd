"""
卷积自编码器 (ConvAE) —— 通过重构 Log-Mel 频谱图进行无监督异常声音检测。
编码器将频谱图压缩到低维隐空间, 解码器从中重建原始输入,
异常片段因与正常分布不符而导致更高的重构误差。
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import yaml


class ConvAEEncoder(nn.Module):
    """
    ConvAE 编码器 — 逐层压缩 Log-Mel 频谱图的空间维度, 最终映射到隐向量。

    参数
    ----
    in_channels : int
        输入通道数 (默认 1, 即单通道频谱图)。
    channels : list[int]
        每层卷积的输出通道数, 长度决定编码器的卷积层数。
    kernel_size : int
        卷积核大小。
    padding : int
        Padding 大小。
    latent_dim : int
        瓶颈隐向量维度。
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        padding: int = 1,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        # 构建卷积编码器层
        layers: List[nn.Module] = []
        in_ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # 计算展平后的维度 (依赖输入尺寸, 此处假设 (1, n_mels, window_size))
        self._latent_dim = latent_dim
        self._flatten_dim: Optional[int] = None  # 首次 forward 后自动推断

        # 隐向量投影层 (惰性初始化)
        self.fc_mu: Optional[nn.Linear] = None

    def _init_fc(self, x: torch.Tensor) -> None:
        """根据输入张量的形状惰性初始化全连接层。"""
        if self.fc_mu is not None:
            return
        conv_out = self.conv(x)
        self._flatten_dim = int(conv_out.view(conv_out.size(0), -1).size(1))
        self.fc_mu = nn.Linear(self._flatten_dim, self._latent_dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入频谱图, shape=(B, 1, n_mels, window_size)。

        返回
        ----
        torch.Tensor
            隐向量 z, shape=(B, latent_dim)。
        """
        self._init_fc(x)
        h = self.conv(x)                    # (B, out_ch, H', W')
        h = h.view(h.size(0), -1)           # (B, flatten_dim)
        z = self.fc_mu(h)                   # (B, latent_dim)
        return z


class ConvAEDecoder(nn.Module):
    """
    ConvAE 解码器 — 从隐向量重建 Log-Mel 频谱图, 结构与编码器对称。

    参数
    ----
    latent_dim : int
        瓶颈隐向量维度。
    channels : list[int]
        编码器使用的通道数序列 (逆序用于解码器)。
    kernel_size : int
        卷积核大小。
    padding : int
        Padding 大小。
    input_shape : tuple[int, int, int]
        原始输入形状 (C, H, W), 用于重建目标尺寸。
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        padding: int = 1,
        input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        if input_shape is None:
            input_shape = (1, 128, 64)

        self._latent_dim_val: int = latent_dim
        self.input_shape = input_shape
        self.channels = channels

        # 逆序通道列表, 用于解码器上采样
        reversed_channels = list(reversed(channels))

        # 展平维度 (通过编码器的逆运算计算, 惰性初始化)
        self._flatten_dim: Optional[int] = None
        self._init_shape: Optional[Tuple[int, int, int]] = None

        # 全连接扩展层 (惰性初始化)
        self.fc_expand: Optional[nn.Linear] = None

        # 构建转置卷积解码器层
        layers: List[nn.Module] = []
        # 循环创建上采样层: 每层通道数从 reversed_channels[i] → reversed_channels[i+1]
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            is_last = (i == len(reversed_channels) - 2)  # 最后一层不加 BN/ReLU
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                                   stride=2, padding=padding, output_padding=1)
            )
            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def _init_fc(self, dummy_input: torch.Tensor, encoder_conv: nn.Module) -> None:
        """通过编码器前向计算展平维度与中间形状。"""
        if self.fc_expand is not None:
            return
        with torch.no_grad():
            conv_out = encoder_conv(dummy_input)
            self._init_shape = conv_out.shape[1:]  # (C, H, W)
            self._flatten_dim = int(conv_out.view(conv_out.size(0), -1).size(1))
        self.fc_expand = nn.Linear(self.latent_dim, self._flatten_dim).to(dummy_input.device)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim_val

    @latent_dim.setter
    def latent_dim(self, val: int) -> None:
        self._latent_dim_val = val

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
            重建的频谱图, shape=(B, 1, n_mels, window_size)。
        """
        h = self.fc_expand(z)               # (B, flatten_dim)
        h = h.view(h.size(0), *self._init_shape)  # (B, C, H, W)
        x_recon = self.deconv(h)            # (B, 1, n_mels, window_size)
        return x_recon


class ConvAE(nn.Module):
    """
    卷积自编码器 (ConvAE) — 端到端的异常声音检测模型。

    使用方式::

        model = ConvAE.from_config("config.yaml")
        x = torch.randn(16, 1, 128, 64)
        x_recon, z = model(x)

    参数
    ----
    encoder : ConvAEEncoder
        编码器。
    decoder : ConvAEDecoder
        解码器。
    """

    def __init__(self, encoder: ConvAEEncoder, decoder: ConvAEDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        返回
        ----
        tuple[torch.Tensor, torch.Tensor]
            (重建频谱图, 隐向量) — 重建 shape 同输入, 隐向量 shape=(B, latent_dim)。
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码, 返回隐向量。"""
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """仅解码重建。"""
        z = self.encoder(x)
        return self.decoder(z)

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "ConvAE":
        """
        从 YAML 配置文件创建 ConvAE 模型。

        参数
        ----
        config_path : str
            YAML 配置文件的路径。

        返回
        ----
        ConvAE
            已初始化的 ConvAE 模型, 编解码器尚未绑定 (需首次 forward 完成惰性初始化)。
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ae_cfg = config["conv_ae"]
        encoder = ConvAEEncoder(
            in_channels=ae_cfg["channels"][0],
            channels=ae_cfg["channels"][1:],
            kernel_size=ae_cfg.get("kernel_size", 3),
            padding=ae_cfg.get("padding", 1),
            latent_dim=ae_cfg["latent_dim"],
        )
        # decoder 的 input_shape 设为 None, 首次 forward 后自动绑定
        decoder = ConvAEDecoder(
            latent_dim=ae_cfg["latent_dim"],
            channels=ae_cfg["channels"],
            kernel_size=ae_cfg.get("kernel_size", 3),
            padding=ae_cfg.get("padding", 1),
        )
        return cls(encoder, decoder)

    def bind(self, sample_input: torch.Tensor) -> None:
        """
        使用一个样本输入完成编解码器的惰性初始化 (形状推断)。

        必须在首次训练/推理前调用一次。
        """
        with torch.no_grad():
            self.encoder._init_fc(sample_input)
            self.decoder._init_fc(sample_input, self.encoder.conv)
            # 确保编码器 fc_mu 已经存在于 decoder.latent_dim 所需的设备
            self.decoder.latent_dim = self.encoder._latent_dim
