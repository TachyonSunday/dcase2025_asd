"""
dataset.py
============================================================
DCASE 2025 Task 2 - 音频数据集类 (Domain-Generalized)
============================================================

功能说明:
    1. 扫描指定目录下的 .wav 音频文件
    2. 将音频加载为 16kHz 单声道原始波形 (配合 CED-Tiny 输入)
    3. 区分训练集（仅正常样本）和测试集（正常 + 异常样本）
    4. 通过文件名关键词自动判定:
       - anomaly_label:  0=正常, 1=异常
       - domain_label:   0=source, 1=target
    5. 支持 Domain-Generalized 设定（训练集区分 source/target 域）

DCASE 2025 数据集命名约定:
    - source 正常:  section_00_source_train_normal_0000_n_B.wav
    - target 正常:  section_00_target_train_normal_0000_n_B.wav
    - source 异常:  section_00_source_test_anomaly_0000_n_B.wav
    - target 异常:  section_00_target_test_anomaly_0000_n_B.wav

注意:
    - 本数据集直接返回原始波形，由 CED-Tiny 特征提取器内部处理为频谱图
    - 采样率固定为 16kHz（CED-Tiny 要求）
    - Log-Mel 计算已移除，不再在 Dataset 层进行频谱转换
"""

import os
import glob
import re
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict


class AudioAnomalyDataset(Dataset):
    """
    音频异常检测数据集 (Domain-Generalized)

    参数:
        audio_dir (str):       音频文件所在目录
        config (dict):         配置文件加载后的字典（来自 config.yaml）
        mode (str):            "train" 或 "test"
                               - train: 仅加载正常样本（忽略异常文件）
                               - test:  加载所有样本，并标注正常/异常
        augment (bool):        是否启用数据增强（仅在训练时使用）
    """

    # ---- 域标签常量 ----
    DOMAIN_SOURCE = 0
    DOMAIN_TARGET = 1

    def __init__(
        self,
        audio_dir: str,
        config: dict,
        mode: str = "train",
        augment: bool = False,
    ):
        super().__init__()

        self.audio_dir = audio_dir
        self.config = config
        self.mode = mode
        self.augment = augment

        # ---- 从配置中提取音频参数 ----
        audio_cfg = config["audio"]
        self.sr = audio_cfg["sr"]
        self.duration = audio_cfg.get("duration", 0)

        # ---- 异常关键词 ----
        dataset_cfg = config.get("dataset", {})
        self.anomaly_keywords = dataset_cfg.get(
            "anomaly_keywords", ["anomaly", "abnormal", "Anomaly", "Abnormal"]
        )

        # ---- 文件扩展名 ----
        file_ext = dataset_cfg.get("file_extension", ".wav")

        # ---- 扫描音频文件 ----
        self.file_paths = self._scan_audio_files(audio_dir, file_ext)
        self.anomaly_labels = self._assign_anomaly_labels()
        self.domain_labels = self._assign_domain_labels()

        # ---- 训练模式下只保留正常样本 ----
        if self.mode == "train":
            normal_indices = [
                i for i, label in enumerate(self.anomaly_labels) if label == 0
            ]
            self.file_paths = [self.file_paths[i] for i in normal_indices]
            self.anomaly_labels = [0] * len(normal_indices)
            self.domain_labels = [self.domain_labels[i] for i in normal_indices]

        # ---- 统计信息 ----
        n_source = self.domain_labels.count(self.DOMAIN_SOURCE)
        n_target = self.domain_labels.count(self.DOMAIN_TARGET)
        n_normal = self.anomaly_labels.count(0)
        n_anomaly = self.anomaly_labels.count(1)

        print(
            f"[Dataset] mode={self.mode} | 目录: {audio_dir} | "
            f"样本数: {len(self.file_paths)} | "
            f"正常: {n_normal} | 异常: {n_anomaly} | "
            f"source: {n_source} | target: {n_target}"
        )

    def _scan_audio_files(self, directory: str, ext: str) -> List[str]:
        """
        递归扫描目录下所有音频文件，返回排序后的文件路径列表
        根据 mode 过滤: train 模式只包含路径/文件名中含 "train" 的文件，
                      test 模式只包含路径/文件名中含 "test" 的文件

        Args:
            directory: 音频目录路径
            ext: 文件扩展名，如 ".wav"

        Returns:
            排序后的文件路径列表
        """
        pattern = os.path.join(directory, "**", f"*{ext}")
        files = sorted(glob.glob(pattern, recursive=True))

        if not files:
            # 如果递归搜索无结果，尝试直接搜索
            pattern = os.path.join(directory, f"*{ext}")
            files = sorted(glob.glob(pattern))

        if not files:
            print(f"[Warning] 目录 {directory} 下未找到任何 {ext} 文件")
            return files

        # ---- 根据 mode 过滤文件 ----
        # train 模式: 只保留路径或文件名中包含 "train" 的文件
        # test 模式: 只保留路径或文件名中包含 "test" 的文件
        filtered_files = []
        for fp in files:
            # 检查完整路径（包括目录名和文件名）
            path_lower = fp.lower()
            if self.mode == "train" and "train" in path_lower:
                filtered_files.append(fp)
            elif self.mode == "test" and "test" in path_lower:
                filtered_files.append(fp)

        if len(filtered_files) < len(files):
            print(f"[Dataset] 按 mode={self.mode} 过滤: {len(files)} → {len(filtered_files)} 个文件")

        return filtered_files

    def _assign_anomaly_labels(self) -> List[int]:
        """
        根据文件名判定异常标签:
            0 = 正常 (normal)
            1 = 异常 (anomaly)

        Returns:
            与 self.file_paths 对应的标签列表
        """
        labels = []
        for fp in self.file_paths:
            filename = os.path.basename(fp)
            # 检查文件名中是否包含异常关键词
            is_anomaly = any(kw in filename for kw in self.anomaly_keywords)
            labels.append(1 if is_anomaly else 0)
        return labels

    def _assign_domain_labels(self) -> List[int]:
        """
        根据文件名判定域标签 (Domain-Generalized):
            0 = source 域
            1 = target 域

        DCASE 2025 文件名约定:
            - section_00_source_... -> source
            - section_00_target_... -> target

        使用正则表达式或字符串匹配进行解析。

        Returns:
            与 self.file_paths 对应的域标签列表
        """
        labels = []
        # 正则表达式: 匹配 _source_ 或 _target_ (前后有下划线分隔)
        domain_pattern = re.compile(r"[_\-.](source|target)[_\-.]")

        for fp in self.file_paths:
            filename = os.path.basename(fp)

            # 优先使用正则匹配
            match = domain_pattern.search(filename.lower())
            if match:
                domain_str = match.group(1)
                if domain_str == "source":
                    labels.append(self.DOMAIN_SOURCE)
                else:
                    labels.append(self.DOMAIN_TARGET)
            else:
                # fallback: 简单字符串包含检查
                if "source" in filename.lower():
                    labels.append(self.DOMAIN_SOURCE)
                elif "target" in filename.lower():
                    labels.append(self.DOMAIN_TARGET)
                else:
                    # 默认标记为 source (未明确域名的样本)
                    labels.append(self.DOMAIN_SOURCE)

        return labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        加载单个音频样本并返回原始波形

        Returns:
            dict:
                - "waveform" (Tensor):  16kHz 单声道原始波形, shape = (num_samples,)
                - "anomaly_label" (int): 0=正常, 1=异常
                - "domain_label" (int):  0=source, 1=target
                - "file_path" (str):    文件路径
                - "filename" (str):     文件名
        """
        file_path = self.file_paths[idx]
        anomaly_label = self.anomaly_labels[idx]
        domain_label = self.domain_labels[idx]

        # ---- 加载音频波形 ----
        waveform = self._load_audio(file_path)

        # ---- 转换为 Tensor, shape: (num_samples,) ----
        waveform_tensor = torch.from_numpy(waveform).float()

        return {
            "waveform": waveform_tensor,
            "anomaly_label": anomaly_label,
            "domain_label": domain_label,
            "file_path": file_path,
            "filename": os.path.basename(file_path),
        }

    def _load_audio(self, file_path: str) -> np.ndarray:
        """
        使用 librosa 加载音频文件，转换为 16kHz 单声道

        Args:
            file_path: 音频文件路径

        Returns:
            一维 numpy 数组，shape = (num_samples,), dtype = float32
        """
        try:
            waveform, sr = librosa.load(file_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"[Error] 无法加载音频: {file_path} -> {e}")
            # 出错时返回静音
            num_samples = int(self.sr * (self.duration if self.duration > 0 else 1.0))
            waveform = np.zeros(num_samples, dtype=np.float32)

        # ---- 时长处理：截断或填充 ----
        if self.duration > 0:
            target_samples = int(self.sr * self.duration)
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]  # 截断
            elif len(waveform) < target_samples:
                # 零填充
                waveform = np.pad(
                    waveform, (0, target_samples - len(waveform)), mode="constant"
                )

        # ---- 数据增强（仅在训练 + augment=True 时生效） ----
        if self.mode == "train" and self.augment:
            waveform = self._apply_augmentation(waveform)

        return waveform

    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        """
        简易数据增强策略（仅在训练阶段使用）:
            1. 随机增益调整
            2. 随机时间偏移
            3. 添加高斯噪声

        Args:
            waveform: 原始波形

        Returns:
            增强后的波形
        """
        # 1. 随机增益 (0.8 ~ 1.2)
        gain = np.random.uniform(0.8, 1.2)
        waveform = waveform * gain

        # 2. 随机时间偏移（最多偏移 5% 的总长度）
        max_shift = int(len(waveform) * 0.05)
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift)
            waveform = np.roll(waveform, shift)

        # 3. 添加微弱高斯噪声
        noise = np.random.normal(0, 0.001, len(waveform)).astype(np.float32)
        waveform = waveform + noise

        return waveform.astype(np.float32)

    # ---- 辅助方法 ----

    def get_file_paths(self) -> List[str]:
        """返回所有文件路径列表"""
        return self.file_paths

    def get_anomaly_labels(self) -> List[int]:
        """返回所有异常标签列表"""
        return self.anomaly_labels

    def get_domain_labels(self) -> List[int]:
        """返回所有域标签列表"""
        return self.domain_labels

    def get_normal_count(self) -> int:
        """返回正常样本数"""
        return self.anomaly_labels.count(0)

    def get_anomaly_count(self) -> int:
        """返回异常样本数"""
        return self.anomaly_labels.count(1)

    def get_source_count(self) -> int:
        """返回 source 域样本数"""
        return self.domain_labels.count(self.DOMAIN_SOURCE)

    def get_target_count(self) -> int:
        """返回 target 域样本数"""
        return self.domain_labels.count(self.DOMAIN_TARGET)

    def get_domain_name(self, domain_label: int) -> str:
        """将域标签转换为可读名称"""
        return "source" if domain_label == self.DOMAIN_SOURCE else "target"


def create_dataloader(
    audio_dir: str,
    config: dict,
    mode: str = "train",
    shuffle: bool = None,
) -> torch.utils.data.DataLoader:
    """
    便捷工厂函数：创建 Dataset + DataLoader

    Args:
        audio_dir: 音频目录
        config: 配置字典
        mode: "train" 或 "test"
        shuffle: 是否打乱顺序（训练默认 True，测试默认 False）

    Returns:
        torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = (mode == "train")

    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)

    dataset = AudioAnomalyDataset(
        audio_dir=audio_dir,
        config=config,
        mode=mode,
        augment=(mode == "train"),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader


# ---- 调试入口 ----
if __name__ == "__main__":
    import yaml

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 测试训练集加载 ----
    print("=" * 60)
    print("测试训练集加载 (Domain-Generalized)")
    print("=" * 60)
    train_dir = config["dataset"]["train_dir"]
    if os.path.exists(train_dir):
        train_ds = AudioAnomalyDataset(train_dir, config, mode="train")
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"\n样本 0:")
            print(f"  Waveform shape:  {sample['waveform'].shape}")
            print(f"  Anomaly label:   {sample['anomaly_label']}")
            print(f"  Domain label:    {sample['domain_label']} ({train_ds.get_domain_name(sample['domain_label'])})")
            print(f"  File:            {sample['filename']}")

            # 统计
            print(f"\n训练集统计:")
            print(f"  总样本数:   {len(train_ds)}")
            print(f"  正常样本:   {train_ds.get_normal_count()}")
            print(f"  Source 域:  {train_ds.get_source_count()}")
            print(f"  Target 域:  {train_ds.get_target_count()}")
    else:
        print(f"[Skip] 训练目录不存在: {train_dir}")

    # ---- 测试测试集加载 ----
    print("\n" + "=" * 60)
    print("测试测试集加载 (Domain-Generalized)")
    print("=" * 60)
    test_dir = config["dataset"]["test_dir"]
    if os.path.exists(test_dir):
        test_ds = AudioAnomalyDataset(test_dir, config, mode="test")
        if len(test_ds) > 0:
            sample = test_ds[0]
            print(f"\n样本 0:")
            print(f"  Waveform shape:  {sample['waveform'].shape}")
            print(f"  Anomaly label:   {sample['anomaly_label']}")
            print(f"  Domain label:    {sample['domain_label']} ({test_ds.get_domain_name(sample['domain_label'])})")
            print(f"  File:            {sample['filename']}")

            # 统计
            print(f"\n测试集统计:")
            print(f"  总样本数:   {len(test_ds)}")
            print(f"  正常样本:   {test_ds.get_normal_count()}")
            print(f"  异常样本:   {test_ds.get_anomaly_count()}")
            print(f"  Source 域:  {test_ds.get_source_count()}")
            print(f"  Target 域:  {test_ds.get_target_count()}")
    else:
        print(f"[Skip] 测试目录不存在: {test_dir}")
