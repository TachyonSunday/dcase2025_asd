"""
训练器模块 —— 封装 ConvAE 的训练循环、Early Stopping、Checkpoint 管理。
"""

import os
import math
from typing import Optional, Dict, Any, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.conv_ae import ConvAE


class EarlyStopping:
    """
    Early Stopping — 验证损失在 patience 轮内未改善则停止训练。

    参数
    ----
    patience : int
        容忍轮数。
    min_delta : float
        视为改善的最小损失变化量。
    mode : str
        ``"min"`` 监控损失下降, ``"max"`` 监控分数上升。
    """

    def __init__(
        self, patience: int = 15, min_delta: float = 1e-5, mode: str = "min"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best = state["best"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]


class Trainer:
    """
    ConvAE 训练器 — 管理训练循环、学习率调度、检查点保存/恢复。

    使用方式::

        trainer = Trainer(model, config_path="config.yaml")
        trainer.train(train_loader, val_loader=val_loader)

    参数
    ----
    model : ConvAE
        待训练的卷积自编码器。
    config_path : str
        YAML 配置文件的路径。
    device : str
        训练设备, ``"cuda"`` 或 ``"cpu"``。
    """

    def __init__(
        self,
        model: ConvAE,
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        train_cfg = self.config["train"]
        self.device = torch.device(
            device or self.config.get("device", "cuda")
        )
        self.model = model.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # 早停
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 15),
            mode="min",
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 路径与超参
        self.epochs = train_cfg["epochs"]
        self.checkpoint_dir = self.config["paths"]["checkpoints"]
        self.logs_dir = self.config["paths"]["logs"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 训练历史
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.current_epoch: int = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        执行一个训练轮次。

        参数
        ----
        train_loader : DataLoader
            训练数据加载器。

        返回
        ----
        float
            当前轮次的平均训练损失。
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            x_recon, _ = self.model(batch_x)
            loss = self.criterion(x_recon, batch_x)
            loss.backward()
            self.optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / total_samples

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        执行一个验证轮次。

        参数
        ----
        val_loader : DataLoader
            验证数据加载器。

        返回
        ----
        float
            当前轮次的平均验证损失。
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)
            x_recon, _ = self.model(batch_x)
            loss = self.criterion(x_recon, batch_x)
            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / total_samples

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        执行完整的训练循环。

        参数
        ----
        train_loader : DataLoader
            训练数据加载器。
        val_loader : DataLoader, 可选
            验证数据加载器, 用于 Early Stopping 监控。

        返回
        ----
        dict[str, list[float]]
            训练历史 (train_loss, val_loss)。
        """
        # 模型绑定 (惰性初始化)
        sample_batch = next(iter(train_loader))[0].to(self.device)
        self.model.bind(sample_batch)

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # 验证
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.history["val_loss"].append(val_loss)
                monitor_loss = val_loss
            else:
                val_loss = float("nan")
                monitor_loss = train_loss

            # 学习率调度
            self.scheduler.step(monitor_loss)

            # 日志
            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

            # Early Stopping
            if self.early_stopping(monitor_loss):
                print(f"Early Stopping 触发于 Epoch {epoch + 1} (best val_loss: {best_val_loss:.6f})")
                break

        # 保存最后一轮的检查点
        self.save_checkpoint("last_model.pt")
        return self.history

    def save_checkpoint(self, filename: str) -> str:
        """
        保存模型检查点。

        参数
        ----
        filename : str
            检查点文件名。

        返回
        ----
        str
            检查点完整路径。
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "early_stopping": self.early_stopping.state_dict(),
                "history": self.history,
            },
            path,
        )
        return path

    def load_checkpoint(self, filename: str) -> int:
        """
        从检查点恢复模型与训练状态。

        参数
        ----
        filename : str
            检查点文件名。

        返回
        ----
        int
            恢复后应开始的轮次编号。
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        self.current_epoch = checkpoint.get("epoch", -1) + 1
        return self.current_epoch
