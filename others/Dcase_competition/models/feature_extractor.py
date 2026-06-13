"""
models/feature_extractor.py
============================================================
DCASE 2025 Task 2 - BEATs 特征提取器 (GenRep 对齐版本)
============================================================

功能说明:
    基于 BEATs (Audio Pre-Training with Acoustic Tokenizers) 预训练模型，
    按照 GenRep baseline 的方式提取多层注意力特征并执行分块时序池化。

    核心流程 (对齐 GenRep run_genrep_dcase2023.py):
        1. 加载本地 BEATs 源码模块 (beats/BEATs.py)
        2. 从检查点恢复预训练权重 (BEATs_iter3_plus_AS2M.pt)
        3. 冻结所有参数，执行推理
        4. 调用 model.extract_features(waveform, layer=11, need_weights=True)
        5. 对每层注意力输出执行 GenRep 分块时序池化:
           reshape(bs, T', num_chunks, -1).mean(dim=1)
        6. 跨所有层拼接，得到最终 Embedding

    BEATs 架构 (从源码验证):
        - 12 层 Transformer Encoder
        - encoder_embed_dim = 768
        - encoder_attention_heads = 12
        - input_patch_size = 16 (Conv2d kernel/stride)
        - 内部 fbank: 128 mel bins, 25ms frame, 10ms shift
        - Patch 嵌入后时间维度: T' = T_fbank // 16

    GenRep 时序池化 (从源码验证):
        - BEATs 输出序列长度: T' * freq_groups = 62 * 8 = 496 tokens
          (T'=62 时间步, freq_groups=8 频率组, 来自 Conv2d patch embedding)
        - 每层隐藏状态: [496, bs, 768]
        - .transpose(0,1) → [bs, 496, 768]
        - .reshape(bs, 62, 8, -1) → [bs, 62, 8, 768]  (62*8=496, -1=D=768)
        - .mean(1) 对 62 个时间步求均值 → [bs, 8, 768]
        - 每层展平: 8 * 768 * 2 = 12288 (Hybrid Mean+Max Pooling)
        - 11 层 (encoder layers 0-10): [11, bs, 12288]
        - 评分器逐层评估, 选取最优层 (通常为 Layer 5)

    所有网络参数彻底冻结，不参与梯度更新。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional


class BEATsFeatureExtractor(nn.Module):
    """
    基于 BEATs 的音频特征提取器 (GenRep 分块时序池化版本)

    参数:
        config (dict): 配置字典，需包含:
            model.beats_dir:    BEATs 源码目录路径
            model.checkpoint:   预训练检查点路径
            model.layer_idx:    提取到哪一层 (0-indexed, 默认 11)
            model.num_chunks:   GenRep 时序分块数 (默认 8)
            audio.sr:           采样率 (默认 16000)
            audio.duration:     音频时长 (秒, 默认 10.0)

    输出:
        Per-layer Embedding, shape = (num_layers, batch_size, per_layer_dim)
        per_layer_dim = num_chunks * D * 2 = 8 * 768 * 2 = 12288 (Hybrid Mean+Max)
        默认返回 11 层 (encoder layers 0-10): [11, bs, 12288]
        评分器逐层评估, 选取最优层
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config.get("model", {})
        audio_cfg = config.get("audio", {})

        self.sr = audio_cfg.get("sr", 16000)
        self.duration = audio_cfg.get("duration", 10.0)

        # ---- BEATs 配置参数 ----
        self.checkpoint_path = model_cfg.get(
            "checkpoint", "./checkpoints/BEATs_iter3_plus_AS2M.pt"
        )
        self.beats_dir = model_cfg.get("beats_dir", "./beats")
        self.layer_idx = model_cfg.get("layer_idx", 11)
        self.num_chunks = model_cfg.get("num_chunks", 8)

        # ---- 将项目根目录和 models/ 加入 sys.path ----
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        from models.beats.BEATs import BEATs as BEATsModel, BEATsConfig

        # ---- 加载检查点 ----
        print(f"[BEATs] 加载检查点: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # ---- 解析检查点结构 ----
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # 标准格式: {"cfg": BEATsConfig.__dict__, "model": state_dict}
            print("[BEATs] 检查点格式: 标准 dict (cfg + model)")
            cfg_dict = checkpoint.get("cfg", {})
            state_dict = checkpoint["model"]

            if isinstance(cfg_dict, dict):
                beats_cfg = BEATsConfig(cfg_dict)
            else:
                # cfg_dict 可能本身就是 BEATsConfig 对象
                beats_cfg = cfg_dict
            pretrained = True

        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:5]
        ):
            # 纯 state_dict
            print("[BEATs] 检查点格式: 纯 state_dict (使用默认 BEATsConfig)")
            beats_cfg = BEATsConfig()
            state_dict = checkpoint
            pretrained = True

        else:
            print("[BEATs] 检查点格式: 未知, 尝试作为 BEATsConfig + state_dict")
            beats_cfg = BEATsConfig()
            state_dict = checkpoint
            pretrained = True

        # ---- 创建 BEATs 模型 ----
        self.model = BEATsModel(beats_cfg)

        # ---- 加载预训练权重 ----
        if pretrained:
            # 处理可能的 DDP 前缀 (module.)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    cleaned_state_dict[key[len("module."):]] = value
                else:
                    cleaned_state_dict[key] = value

            missing, unexpected = self.model.load_state_dict(
                cleaned_state_dict, strict=False
            )
            print(
                f"[BEATs] 权重加载完成 | "
                f"Missing keys: {len(missing)} | "
                f"Unexpected keys: {len(unexpected)}"
            )
            if missing:
                print(f"  Missing (前 5 个): {missing[:5]}")
            if unexpected:
                print(f"  Unexpected (前 5 个): {unexpected[:5]}")

        # ---- 从模型配置中提取关键维度 ----
        self.encoder_embed_dim = self.model.cfg.encoder_embed_dim  # D = 768
        self.num_encoder_layers = self.model.cfg.encoder_layers  # L = 12
        self.num_heads = self.model.cfg.encoder_attention_heads  # H = 12
        self.patch_size = self.model.cfg.input_patch_size  # P = 16

        # ---- 冻结所有参数 ----
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # ---- 干运行: 计算 T' (时间维度) 和输出维度 ----
        self._compute_temporal_dims(audio_cfg)

        print(
            f"[BEATs] 加载完成 | "
            f"D={self.encoder_embed_dim} | "
            f"L={self.num_encoder_layers} | "
            f"H={self.num_heads} | "
            f"patch_size={self.patch_size} | "
            f"T'={self.T_prime} | "
            f"chunks={self.num_chunks} | "
            f"layer_idx={self.layer_idx} | "
            f"output_dim={self.embedding_dim}"
        )

    def _compute_temporal_dims(self, audio_cfg: dict):
        """
        通过干运行计算 BEATs 的时间维度和最终输出维度。

        BEATs 内部流程:
            waveform → fbank [B, T_fbank, 128]
                     → Conv2d(1, 512, kernel=P, stride=P)
                       → [B, 512, T_temporal, freq_groups]
                       其中 T_temporal = (T_fbank - P) // P + 1
                            freq_groups = 128 // P
                     → reshape → [B, 512, T_temporal * freq_groups]
                     → transpose → [B, seq_len, 512]
                     → post_extract_proj → [B, seq_len, D]
            其中 seq_len = T_temporal * freq_groups

        GenRep 池化维度推导:
            layer_output: [seq_len, B, D]  (T-first)
            .transpose(0,1) → [B, seq_len, D]
            .reshape(bs, T_temporal, freq_groups, -1)
                因为 T_temporal * freq_groups = seq_len, -1 = D
                → [B, T_temporal, freq_groups, D]
            .mean(1) 对 T_temporal (时间步) 求均值
                → [B, freq_groups, D]
            flatten → freq_groups * D * 2 = num_chunks * D * 2 = 12288
        """
        num_samples = int(self.sr * self.duration)

        # ---- 使用 BEATs 内部 preprocess 计算 fbank 维度 ----
        with torch.no_grad():
            dummy = torch.zeros(1, num_samples)
            fbank = self.model.preprocess(dummy)  # [1, T_fbank, 128]
            T_fbank = fbank.shape[1]

        # ---- Patch embedding 后的空间维度 ----
        self.T_fbank = T_fbank
        self.T_prime = (T_fbank - self.patch_size) // self.patch_size + 1  # T_temporal
        self.freq_groups = 128 // self.patch_size  # freq_groups

        # ---- 序列总长度 ----
        self.seq_len = self.T_prime * self.freq_groups  # 62 * 8 = 496

        # ---- GenRep 池化后每层维度 ----
        # Hybrid Pooling (Mean + Max): 2x 原始维度
        # .mean(1) → [B, freq_groups, D]
        # .max(1)  → [B, freq_groups, D]
        # cat → [B, freq_groups, D*2]
        # flatten → freq_groups * D * 2 = 12288
        self.per_layer_dim = self.freq_groups * self.encoder_embed_dim * 2

        # ---- 总共使用的层数 ----
        # layer_idx 控制 BEATs extract_features 计算到哪一层
        # attns[1:] 跳过 pre-encoder, 取 encoder layers 0..layer_idx
        self.num_used_layers = self.layer_idx  # layer_idx=11 → 11 layers (0-10)

        # ---- 单层的 Embedding 维度 (评分器逐层评估) ----
        self.embedding_dim = self.per_layer_dim  # 12288 (Hybrid Mean+Max)

        print(
            f"[BEATs] 维度计算: "
            f"T_fbank={self.T_fbank} | "
            f"T_temporal={self.T_prime} | "
            f"freq_groups={self.freq_groups} | "
            f"seq_len={self.seq_len} | "
            f"per_layer_dim={self.per_layer_dim} | "
            f"num_used_layers={self.num_used_layers}"
        )

    def _genrep_temporal_pool(
        self, attns: list, bs: int
    ) -> torch.Tensor:
        """
        GenRep 分块时序池化 (严格对齐 run_genrep_dcase2023.py)

        原始 GenRep 代码:
            out_layers = [
                f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0)
                for f_layer in attns
            ][1:]

        维度推导 (10s @ 16kHz):
            f_layer[0] = x: [496, B, 768]      (T-first, seq_len=496)
            .transpose(0, 1) → [B, 496, 768]     (B-first)
            .reshape(bs, 62, 8, -1)               (62*8=496, -1=D=768)
                → [B, 62, 8, 768]
            .mean(1) 对 62 个时间步求均值 + .max(1) 取最大值 (Hybrid Pooling)
                → mean: [B, 8, 768], max: [B, 8, 768]
                → cat:  [B, 8, 1536]
            flatten → 8 * 1536 = 12288
            .cpu().unsqueeze(0) → [1, B, 12288]
            [1:] 去掉 pre-transformer 条目 (index 0)

        Args:
            attns: layer_results 列表, 每个元素为 (x, z) 元组
                - x: [seq_len, B, D] — 隐藏状态 (T-first 格式)
                - z: attention weights or None
            bs: 当前 batch 大小

        Returns:
            Tensor, shape = (num_used_layers, bs, per_layer_dim)
            即 (11, bs, 12288) 默认参数 (Hybrid Mean+Max Pooling)
        """
        out_layers = []
        for f_layer in attns:
            # f_layer[0] = x: [seq_len, B, D]  (seq_len = T_prime * freq_groups)
            # .transpose(0, 1) → [B, seq_len, D]
            x_transposed = f_layer[0].transpose(0, 1)  # [B, 496, 768]

            # .reshape(bs, T_prime, freq_groups, -1)
            # T_prime * freq_groups = seq_len, so -1 = D = 768
            # → [B, 62, 8, 768]
            reshaped = x_transposed.reshape(
                bs, self.T_prime, self.freq_groups, -1
            )

            # Hybrid Pooling (Mean + Max) — 捕获瞬态异常信号
            # Mean: 捕获持续性特征 (背景声学模式)
            # Max: 捕获瞬态异常 (短促的齿轮咔嗒声、突发噪声)
            # [B, 62, 8, 768] → [B, 8, 768] each
            mean_pool = reshaped.mean(1)        # [B, 8, 768]
            max_pool = reshaped.max(1).values   # [B, 8, 768]
            pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 8, 1536]

            # flatten → [B, freq_groups * D * 2] = [B, 12288]
            flat = pooled.reshape(bs, -1)

            # .cpu().unsqueeze(0) → [1, B, 12288]
            out_layers.append(flat.cpu().unsqueeze(0))

        # [1:] — 去掉 index 0 (pre-transformer 输入, 非 encoder 层输出)
        # 取 encoder layers 0..(layer_idx-1), 共 layer_idx 层
        out_layers = out_layers[1 : self.layer_idx + 1]

        # 堆叠: num_layers 个 [1, bs, 12288]
        # → [num_layers, bs, 12288]
        return torch.cat(out_layers, dim=0)

    @torch.no_grad()
    def extract_single_batch(
        self, waveform: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        """
        对单个 batch 的波形执行 BEATs 特征提取 + GenRep 池化

        Args:
            waveform: shape = (B, num_samples), 16kHz 单声道
            device: 计算设备

        Returns:
            features: shape = (num_layers, B, per_layer_dim)
                      即 (11, B, 12288), CPU 上的 float32 tensor
                      评分器逐层评估, 选取最优层
        """
        self.model.eval()
        waveform = waveform.to(device)
        bs = waveform.shape[0]

        # ---- BEATs forward pass ----
        # extract_features 返回: (x, padding_mask, layer_results)
        # layer_results: list of (x, z) tuples
        #   x: [seq_len, B, D] — hidden states (T-first)
        #   z: attention weights or None
        _, _, attns = self.model.extract_features(
            waveform,
            padding_mask=None,
            need_weights=True,
            layer=self.layer_idx,
        )

        # ---- GenRep 分块时序池化 ----
        # attns 包含 layer_idx + 1 个条目 (layer=10 时):
        #   [0]: pre-transformer input
        #   [1..layer_idx]: encoder layer outputs 0..(layer_idx-1)
        # _genrep_temporal_pool 内部执行 [1:] 跳过 pre-transformer
        features = self._genrep_temporal_pool(attns, bs)
        # features shape: [num_layers, bs, per_layer_dim] = [11, bs, 12288]

        return features

    def get_embedding_dim(self) -> int:
        """返回输出 Embedding 维度"""
        return self.embedding_dim

    def get_embeddings(
        self,
        dataloader,
        device: str = "cuda",
        layer_idx: Optional[int] = None,
        pooling: str = "temporal",
    ) -> tuple:
        """
        遍历 DataLoader 批量提取 BEATs 特征 (GenRep 分块时序池化)

        Args:
            dataloader: AudioAnomalyDataset 的 DataLoader
                batch dict 需包含 "waveform" 或 "input" 键
            device: 计算设备 ("cuda" / "cpu")
            layer_idx: 提取到哪一层 (默认使用 self.layer_idx)
            pooling: 池化模式 (当前仅支持 "temporal")

        Returns:
            (embeddings, anomaly_labels, domain_labels, file_paths):
                embeddings:     np.ndarray, shape = (num_layers, N, per_layer_dim)
                                即 (11, N, 12288) — 评分器逐层评估
                anomaly_labels: list of int (0=正常, 1=异常)
                domain_labels:  list of int (0=source, 1=target)
                file_paths:     list of str
        """
        if pooling != "temporal":
            raise ValueError(
                f"[BEATs] 当前仅支持 'temporal' 池化模式, 收到: '{pooling}'"
            )

        # 临时覆盖 layer_idx
        if layer_idx is not None:
            original_layer_idx = self.layer_idx
            self.layer_idx = layer_idx
            self.num_used_layers = layer_idx

        self.eval()
        self.to(device)

        all_embeddings = []
        all_anomaly_labels = []
        all_domain_labels = []
        all_file_paths = []

        pbar = tqdm(dataloader, desc="[BEATs] 提取特征", leave=True)
        for batch in pbar:
            # ---- 兼容不同的 batch key ----
            if "input" in batch:
                waveform = batch["input"]
            elif "waveform" in batch:
                waveform = batch["waveform"]
            else:
                available_keys = list(batch.keys())
                raise KeyError(
                    f"[BEATs] batch 中未找到 'input' 或 'waveform' 键. "
                    f"可用键: {available_keys}"
                )

            # ---- 提取特征 → [num_layers, bs, 12288] ----
            features = self.extract_single_batch(waveform, device=device)
            all_embeddings.append(features)

            # ---- 收集标签和路径 ----
            if "anomaly_label" in batch:
                labels = batch["anomaly_label"]
                if isinstance(labels, torch.Tensor):
                    all_anomaly_labels.extend(labels.tolist())
                else:
                    all_anomaly_labels.extend(labels)

            if "domain_label" in batch:
                dlabels = batch["domain_label"]
                if isinstance(dlabels, torch.Tensor):
                    all_domain_labels.extend(dlabels.tolist())
                else:
                    all_domain_labels.extend(dlabels)

            if "file_path" in batch:
                all_file_paths.extend(batch["file_path"])

            # ---- GPU 显存管理 ----
            torch.cuda.empty_cache()

            pbar.set_postfix(
                {
                    "batch_shape": tuple(features.shape),
                    "mem": f"{torch.cuda.memory_allocated(device) / 1e9:.2f}GB"
                    if device == "cuda" and torch.cuda.is_available()
                    else "N/A",
                }
            )

        # ---- 拼接所有 batch 的特征 (沿 batch 维度) ----
        # 每个 batch: [num_layers, bs, 12288]
        # 拼接后: [num_layers, N, 12288]
        all_embeddings = torch.cat(all_embeddings, dim=1)
        embeddings_np = all_embeddings.numpy()  # 已在 CPU 上

        # ---- 恢复 layer_idx ----
        if layer_idx is not None:
            self.layer_idx = original_layer_idx
            self.num_used_layers = original_layer_idx

        print(
            f"[BEATs] 特征提取完成 | "
            f"层数: {embeddings_np.shape[0]} | "
            f"样本数: {embeddings_np.shape[1]} | "
            f"每层维度: {embeddings_np.shape[2]} | "
            f"数据类型: {embeddings_np.dtype}"
        )

        return embeddings_np, all_anomaly_labels, all_domain_labels, all_file_paths


def get_feature_extractor(config: dict) -> BEATsFeatureExtractor:
    """
    工厂函数：根据配置创建 BEATs 特征提取器

    Args:
        config: 配置字典 (来自 config.yaml)

    Returns:
        BEATsFeatureExtractor 实例
    """
    model = BEATsFeatureExtractor(config)
    embedding_dim = model.get_embedding_dim()

    print(
        f"[Factory] BEATs 特征提取器 | "
        f"Embedding 维度: {embedding_dim} | "
        f"Checkpoint: {config.get('model', {}).get('checkpoint', 'N/A')}"
    )

    return model


# ---- 调试入口 ----
if __name__ == "__main__":
    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 创建模型 ----
    model = get_feature_extractor(config)
    print(f"\n模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- 测试前向传播 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 模拟输入: batch_size=2, 16kHz, 10 秒
    batch_size = 2
    sr = config["audio"]["sr"]
    duration = config["audio"]["duration"]
    dummy_input = torch.randn(batch_size, int(sr * duration)).to(device)

    with torch.no_grad():
        output = model.extract_single_batch(dummy_input, device=device)

    print(f"\n输入 shape:       {dummy_input.shape}")
    print(f"输出 shape:       {output.shape}")
    print(f"预期 shape:       ({model.num_used_layers}, {batch_size}, {model.per_layer_dim})")
    print(f"输出样例 (L0):    {output[0, 0, :5]}")
    print(f"\nGenRep 分块时序池化:")
    print(f"  音频时长:       {duration}s")
    print(f"  T_fbank:        {model.T_fbank}")
    print(f"  T_temporal:     {model.T_prime}")
    print(f"  freq_groups:    {model.freq_groups}")
    print(f"  seq_len:        {model.seq_len} ({model.T_prime} * {model.freq_groups})")
    print(f"  reshape:        (bs, {model.T_prime}, {model.freq_groups}, {model.encoder_embed_dim})")
    print(f"  mean(1):        (bs, {model.freq_groups}, {model.encoder_embed_dim})")
    print(f"  per_layer_dim:  {model.freq_groups} * {model.encoder_embed_dim} = {model.per_layer_dim}")
    print(f"  num_layers:     {model.num_used_layers} (encoder layers 0-{model.num_used_layers - 1})")
    print(f"  output shape:   [{model.num_used_layers}, bs, {model.per_layer_dim}]")
