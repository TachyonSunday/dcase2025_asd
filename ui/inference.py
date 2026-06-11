"""
推理桥接模块 —— 将训练好的模型与前端音频上传管线连接。
负责模型加载、特征提取、逐帧推理、异常分数聚合。
"""

import os
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import yaml

from src.features.pipeline import FeaturePipeline
from src.models.conv_ae import ConvAE
from src.models.dann import DANNAutoEncoder


class MLPAE(torch.nn.Module):
    """MLP 自编码器 (与 train_all_baseline.py BaselineAE 一致)。"""
    def __init__(self):
        super().__init__()
        bn = {"momentum": 0.01, "eps": 1e-3}
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(640, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 8), torch.nn.BatchNorm1d(8, **bn), torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128, **bn), torch.nn.ReLU(),
            torch.nn.Linear(128, 640),
        )
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z


class InferenceEngine:
    """
    推理引擎 —— 封装模型加载与推理流程, 为 Streamlit 前端提供简洁 API。

    使用方式::

        engine = InferenceEngine("config.yaml")
        engine.load_model("checkpoints/best_model.pt", model_type="conv_ae")
        result = engine.predict("uploaded_audio.wav")

    参数
    ----
    config_path : str
        YAML 配置文件路径。
    device : str
        推理设备, 默认 ``"cuda"``。
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device or self.config.get("device", "cuda"))
        self.pipeline = FeaturePipeline(config_path)

        # 帧切片参数
        frame_cfg = self.config["frame"]
        self.window_size = frame_cfg["window_size"]
        self.hop_size = frame_cfg["hop_size"]

        # 模型
        self.model: Optional[torch.nn.Module] = None
        self.model_type: Optional[str] = None
        self.threshold: Optional[float] = None

    def load_model(
        self,
        checkpoint_path: str,
        model_type: str = "conv_ae",
        threshold: Optional[float] = None,
    ) -> None:
        """
        加载模型检查点。

        参数
        ----
        checkpoint_path : str
            .pt 检查点文件路径。
        model_type : str
            ``"conv_ae"`` 或 ``"dann"``。
        threshold : float, 可选
            异常判定阈值, 若为 None 则使用配置文件中的值。
        """
        self.model_type = model_type

        if model_type == "mlp":
            self.model = MLPAE().to(self.device)
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt)
        elif model_type == "conv_ae":
            self.model = ConvAE.from_config()
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            dummy = torch.randn(1, 1, self.config["mel"]["n_mels"], self.window_size).to(self.device)
            self.model.bind(dummy)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif model_type == "dann":
            self.model = DANNAutoEncoder.from_config()
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            dummy = torch.randn(1, 1, self.config["mel"]["n_mels"], self.window_size).to(self.device)
            self.model.bind(dummy)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        if self.threshold is None:
            manual = self.config["anomaly"].get("decision_threshold")
            if manual is not None:
                self.threshold = float(manual)

    def _sliding_window_frames(self, spec: torch.Tensor) -> torch.Tensor:
        """
        将长频谱图按滑动窗口切分为帧块批次。

        参数
        ----
        spec : torch.Tensor
            Log-Mel 频谱图, shape=(1, n_mels, total_frames)。

        返回
        ----
        torch.Tensor
            帧块批次, shape=(num_windows, 1, n_mels, window_size)。
        """
        total_frames = spec.shape[-1]
        if total_frames < self.window_size:
            # 太短则填充
            pad = self.window_size - total_frames
            spec = torch.nn.functional.pad(spec, (0, pad))
            total_frames = self.window_size

        frames: List[torch.Tensor] = []
        for start in range(0, total_frames - self.window_size + 1, self.hop_size):
            chunk = spec[:, :, start : start + self.window_size]
            frames.append(chunk)

        if not frames:
            # 至少取最后一个窗口
            frames.append(spec[:, :, -self.window_size:])

        return torch.stack(frames, dim=0)  # (N, 1, n_mels, window_size)

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        """
        对音频文件执行完整的异常检测推理。

        参数
        ----
        audio_path : str
            音频文件路径 (wav/mp3/flac)。

        返回
        ----
        dict
            包含以下键的字典:
            - ``"waveform"``: 原始波形 (np.ndarray)
            - ``"sample_rate"``: 采样率 (int)
            - ``"log_mel"``: Log-Mel 频谱图 (np.ndarray, shape=(n_mels, T))
            - ``"frame_scores"``: 逐帧异常分数 (np.ndarray, shape=(N,))
            - ``"file_score"``: 文件级异常分数 (float)
            - ``"is_anomaly"``: 是否判定为异常 (bool)
            - ``"recon_error_map"``: 重建误差热力图 (np.ndarray, shape=(n_mels, T))
        """
        if self.model is None:
            raise RuntimeError("请先调用 load_model() 加载模型")

        # 步骤1: 特征提取
        waveform = self.pipeline.load_audio(audio_path)
        waveform_denoised = self.pipeline.denoiser.denoise(
            waveform, enabled=self.pipeline.denoise_enabled
        )
        log_mel = self.pipeline.extractor.extract(waveform_denoised)  # (n_mels, T)

        # 转为张量
        spec_tensor = torch.from_numpy(log_mel).unsqueeze(0).to(self.device)  # (1, n_mels, T)

        # 步骤2: 推理 (分支: MLP用5帧堆叠, ConvAE用2D卷积)
        frame_scores: List[float] = []
        recon_errors_full: Optional[torch.Tensor] = None

        if self.model_type == "mlp":
            # MLP: 5 帧堆叠 → 640-dim 向量 → 逐帧 MSE
            n_frames = 5
            T = spec_tensor.shape[-1]
            n_vecs = max(0, T - n_frames + 1)
            vecs = []
            for t in range(n_vecs):
                vecs.append(spec_tensor[0, :, t:t+n_frames].flatten())  # (640,)
            if not vecs:
                vecs.append(torch.zeros(640))
            all_vecs = torch.stack(vecs, dim=0).to(self.device)  # (N, 640)
            # 逐 batch 推理
            bs = 2048
            for i in range(0, len(all_vecs), bs):
                batch = all_vecs[i:i+bs]
                x_recon, _ = self.model(batch)
                mse = ((x_recon - batch) ** 2).mean(dim=1)  # (B,)
                frame_scores.extend(mse.cpu().tolist())
            # MLP 不产生 2D 重建误差热力图
        else:
            # ConvAE/DANN: 2D 滑动窗口
            frames = self._sliding_window_frames(spec_tensor)
            batch_size = self.config["train"]["batch_size"]
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size].to(self.device)
                if self.model_type == "conv_ae":
                    x_recon, _ = self.model(batch)
                else:
                    (x_recon, _), _ = self.model(batch)
                sq_error = (x_recon - batch) ** 2
                scores = sq_error.mean(dim=[1, 2, 3])
                frame_scores.extend(scores.cpu().tolist())
                sq_error_mean = sq_error.mean(dim=0)
                if recon_errors_full is None:
                    recon_errors_full = sq_error_mean.cpu()
                else:
                    recon_errors_full = torch.cat([recon_errors_full, sq_error_mean.cpu()], dim=-1)

        frame_scores_arr = np.array(frame_scores)

        # 步骤4: 聚合为文件级分数 (均值 + top-10% 均值)
        file_score = float(np.mean(frame_scores_arr))
        k = max(1, int(len(frame_scores_arr) * 0.1))  # top 10%
        topk_score = float(np.mean(np.sort(frame_scores_arr)[-k:]))
        max_score = float(np.max(frame_scores_arr))

        # 步骤5: 判定 (topk 均值对局部异常更敏感)
        if self.threshold is not None:
            is_anomaly = bool(topk_score > self.threshold)
        else:
            is_anomaly = bool(topk_score > file_score * 1.5)

        return {
            "waveform": waveform,
            "waveform_denoised": waveform_denoised,
            "sample_rate": self.pipeline.sample_rate,
            "log_mel": log_mel,
            "frame_scores": frame_scores_arr,
            "file_score": file_score,
            "topk_score": topk_score,
            "max_score": max_score,
            "is_anomaly": is_anomaly,
            "recon_error_map": recon_errors_full.squeeze(0).numpy() if recon_errors_full is not None else None,
        }
