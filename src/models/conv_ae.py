"""
卷积自编码器 (ConvAE) —— 通过重构 Log-Mel 频谱图进行无监督异常声音检测。
v3: 动态层数卷积瓶颈, 支持任意数量的下采样层。
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class ConvAEEncoder(nn.Module):
    """
    ConvAE 编码器 — N 层 stride-2 空间压缩 + 1 层 stride-1 通道瓶颈。
    层数由 channels 列表长度决定。

    参数
    ----
    in_channels : int
        输入通道数。
    channels : list[int]
        每层下采样的输出通道数, 如 [64, 128] 表示 2 层下采样。
    bottleneck_dim : int
        瓶颈层通道数。
    kernel_size : int
        卷积核大小 (默认 3)。
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        bottleneck_dim: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        # 动态创建下采样层
        self.down_layers = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channels:
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # stride-1 通道瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (瓶颈特征图, GAP 隐向量)。"""
        h = x
        for down in self.down_layers:
            h = down(h)
        fmap = self.bottleneck(h)
        z = fmap.mean(dim=[2, 3])
        return fmap, z


class ConvAEDecoder(nn.Module):
    """
    ConvAE 解码器 — 镜像编码器结构。

    参数
    ----
    bottleneck_dim : int
        瓶颈层通道数。
    channels : list[int]
        解码器上采样通道数 (逆序), 如 [128, 64] (对应编码器 [64, 128])。
    out_channels : int
        输出通道数。
    kernel_size : int
        转置卷积核大小。
    """

    def __init__(
        self,
        bottleneck_dim: int = 32,
        channels: Optional[List[int]] = None,
        out_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [128, 64, 32]

        # stride-1 通道扩展
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck_dim, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # 动态创建上采样层
        self.up_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernel_size,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ))
        # 最后一层: 恢复到输入通道, 不加 BN/ReLU
        self.final = nn.ConvTranspose2d(channels[-1], out_channels, kernel_size=kernel_size,
                                        stride=2, padding=1, output_padding=1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        """从瓶颈特征图重建频谱图。"""
        h = self.expand(fmap)
        for up in self.up_layers:
            h = up(h)
        return self.final(h)


class ConvAE(nn.Module):
    """卷积自编码器 (ConvAE) — v3 动态层数。"""

    def __init__(self, encoder: ConvAEEncoder, decoder: ConvAEDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap, z = self.encoder(x)
        x_recon = self.decoder(fmap)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, z = self.encoder(x)
        return z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        fmap, _ = self.encoder(x)
        return self.decoder(fmap)

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "ConvAE":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ae_cfg = config["conv_ae"]
        channels = ae_cfg.get("channels", [1, 32, 64, 128])
        encoder_channels = channels[1:]  # 去掉输入通道
        decoder_channels = list(reversed(encoder_channels))
        encoder = ConvAEEncoder(
            in_channels=channels[0],
            channels=encoder_channels,
            bottleneck_dim=ae_cfg["latent_dim"],
            kernel_size=ae_cfg.get("kernel_size", 3),
        )
        decoder = ConvAEDecoder(
            bottleneck_dim=ae_cfg["latent_dim"],
            channels=decoder_channels,
            out_channels=channels[0],
            kernel_size=ae_cfg.get("kernel_size", 3),
        )
        return cls(encoder, decoder)

    def bind(self, sample_input: torch.Tensor) -> None:
        pass
