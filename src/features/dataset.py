"""
数据集模块 —— 基于 torch.utils.data.Dataset 的机器声音数据集, 支持帧级加载。
"""

import os
import glob
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset


class MachineSoundDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    从预处理后的 .pt 张量文件加载 Log-Mel 频谱片段。

    每个 .pt 文件包含一个形状为 ``(1, n_mels, total_frames)`` 的张量,
    Dataset 负责按滑动窗口切分为固定长度的时间帧块。

    参数
    ----
    data_dir : str
        存放 .pt 文件的目录路径。
    window_size : int
        每帧块包含的频谱时间步数。
    hop_size : int
        相邻帧块之间的滑动步长。
    label : int
        该目录下所有样本的标签 (0=正常, 1=异常), 默认为 0。
    file_pattern : str
        文件搜索匹配模式, 默认 ``"*.pt"``。
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 64,
        hop_size: int = 32,
        label: int = 0,
        file_pattern: str = "*.pt",
    ) -> None:
        self.data_dir = data_dir
        self.window_size = window_size
        self.hop_size = hop_size
        self.label = label

        # 收集所有 .pt 文件路径
        self.file_paths: List[str] = sorted(
            glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True)
        )

        # 预计算每个文件中可切分出的帧块索引, [(file_path, start_frame), ...]
        self._indices: List[Tuple[str, int]] = self._build_index()

    def _build_index(self) -> List[Tuple[str, int]]:
        """预扫描所有文件, 构建 (文件路径, 起始帧) 索引映射表。"""
        indices: List[Tuple[str, int]] = []
        for fpath in self.file_paths:
            try:
                spec = torch.load(fpath, weights_only=True)
                # spec shape: (1, n_mels, total_frames)
                total_frames = spec.shape[-1]
                if total_frames >= self.window_size:
                    for start in range(0, total_frames - self.window_size + 1, self.hop_size):
                        indices.append((fpath, start))
            except Exception:
                continue
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个帧块及其标签。

        返回
        ----
        tuple[torch.Tensor, int]
            (频谱帧块, 标签) — 帧块 shape=(1, n_mels, window_size)。
        """
        fpath, start_frame = self._indices[idx]
        spec = torch.load(fpath, weights_only=True)
        chunk = spec[:, :, start_frame : start_frame + self.window_size]
        return chunk, self.label

    @property
    def num_files(self) -> int:
        """返回数据集包含的 .pt 文件数量。"""
        return len(self.file_paths)

    @property
    def num_frames(self) -> int:
        """返回数据集包含的总帧块数。"""
        return len(self._indices)
