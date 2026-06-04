"""
DANN 训练器 —— 联合优化重构损失与域对抗损失, 含 λ 退火调度。
"""

import os
import math
from typing import Optional, Dict, Any, List

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dann import DANNAutoEncoder
from src.utils.trainer import EarlyStopping


class LambdaScheduler:
    """
    λ 退火调度器 — 控制域对抗损失的权重随训练进程逐步增大。

    使用指数增长策略::

        λ(p) = λ_final * (1 - exp(-γ * p))
               ────────────────────────────
               where p = current_step / total_steps

    这使编码器先专注于重构学习, 再逐步引入域对抗压力,
    避免训练初期域分类器支配梯度, 导致不稳定的表示。

    参数
    ----
    lambda_init : float
        初始 λ (通常为 0)。
    lambda_final : float
        最终 λ (域对抗损失的最大权重)。
    gamma : float
        指数增长因子, 越大则 λ 增长越快。
    """

    def __init__(
        self,
        lambda_init: float = 0.0,
        lambda_final: float = 0.1,
        gamma: float = 10.0,
    ) -> None:
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.gamma = gamma

    def get_lambda(self, progress: float) -> float:
        """
        根据训练进度计算当前 λ 值。

        参数
        ----
        progress : float
            训练进度, 取值范围 [0, 1] (0=开始, 1=结束)。

        返回
        ----
        float
            当前 λ 值。
        """
        progress = max(0.0, min(1.0, progress))
        # λ(p) = λ_final * (2 / (1 + exp(-γ * p)) - 1)
        # 使用 sigmoid 平滑增长
        scaled = (2.0 / (1.0 + math.exp(-self.gamma * progress))) - 1.0
        return self.lambda_init + (self.lambda_final - self.lambda_init) * scaled


class DANNTrainer:
    """
    DANN 训练器 — 联合优化重构损失 (MSE) 和域对抗损失 (CrossEntropy)。

    使用方式::

        trainer = DANNTrainer(model, config_path="config.yaml")
        trainer.train(train_loader, val_loader)

    参数
    ----
    model : DANNAutoEncoder
        DANN 域对抗自编码器。
    config_path : str
        YAML 配置文件路径。
    device : str
        训练设备。
    """

    def __init__(
        self,
        model: DANNAutoEncoder,
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        train_cfg = self.config["train"]
        dann_cfg = self.config["dann"]
        self.device = torch.device(device or self.config.get("device", "cuda"))
        self.model = model.to(self.device)

        # 分离参数组: 编码器+解码器 vs 域分类器
        ae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        domain_params = list(self.model.domain_classifier.parameters())

        self.optimizer_ae = optim.Adam(
            ae_params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )
        self.optimizer_domain = optim.Adam(
            domain_params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )

        # 学习率调度器
        self.scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_ae, mode="min", factor=0.5, patience=5
        )
        self.scheduler_domain = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_domain, mode="min", factor=0.5, patience=5
        )

        # 早停
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 15),
            mode="min",
        )

        # λ 退火调度
        self.lambda_scheduler = LambdaScheduler(
            lambda_init=dann_cfg.get("lambda_init", 0.0),
            lambda_final=dann_cfg.get("lambda_final", 0.1),
            gamma=dann_cfg.get("lambda_gamma", 10.0),
        )

        # 损失函数
        self.recon_criterion = nn.MSELoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        # 路径与超参
        self.epochs = train_cfg["epochs"]
        self.checkpoint_dir = self.config["paths"]["checkpoints"]
        self.logs_dir = self.config["paths"]["logs"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 训练历史
        self.history: Dict[str, List[float]] = {
            "train_recon_loss": [],
            "train_domain_loss": [],
            "val_recon_loss": [],
            "val_domain_accuracy": [],
            "lambda_values": [],
        }
        self.current_epoch: int = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        执行一个 DANN 训练轮次。

        每步同时更新:
        1. 域分类器 (最小化域分类损失, 提升域判别能力)
        2. 编码器+解码器 (最小化重构损失 + λ * 域对抗损失, 学习域不变特征)

        返回
        ----
        dict[str, float]
            当前轮次的平均损失统计。
        """
        self.model.train()
        total_recon = 0.0
        total_domain = 0.0
        total_samples = 0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [DANN Train]")
        for batch_idx, (batch_x, batch_domain) in enumerate(pbar):
            batch_x = batch_x.to(self.device)
            batch_domain = batch_domain.to(self.device)
            bs = batch_x.size(0)
            total_samples += bs

            # 计算训练进度 (用于 λ 退火)
            global_progress = (
                self.current_epoch * len(train_loader) + batch_idx
            ) / (self.epochs * len(train_loader))
            current_lambda = self.lambda_scheduler.get_lambda(global_progress)
            self.model.set_lambda(current_lambda)

            # ---- 步骤1: 更新域分类器 (最大化域判别准确率) ----
            self.optimizer_domain.zero_grad()
            (x_recon, z), domain_logits = self.model(batch_x)
            domain_loss = self.domain_criterion(domain_logits, batch_domain)
            domain_loss.backward(retain_graph=True)
            self.optimizer_domain.step()

            # ---- 步骤2: 更新编码器+解码器 (最小化重构 + 对抗) ----
            self.optimizer_ae.zero_grad()
            recon_loss = self.recon_criterion(x_recon, batch_x)
            # 域对抗损失: 编码器希望域分类器犯错 (最大化域分类熵)
            _, domain_logits_adv = self.model(batch_x)
            domain_adv_loss = self.domain_criterion(domain_logits_adv, batch_domain)
            # 联合损失: 重构 + λ * 域对抗
            # 注意: 经过 GRL 后, domain_adv_loss 的梯度已经反转,
            # 所以最小化此联合损失 = 最小化重构 + 最大化域混淆
            joint_loss = recon_loss + current_lambda * domain_adv_loss
            joint_loss.backward()
            self.optimizer_ae.step()

            total_recon += recon_loss.item() * bs
            total_domain += domain_loss.item() * bs

            pbar.set_postfix({
                "recon": f"{recon_loss.item():.2f}",
                "domain": f"{domain_loss.item():.2f}",
                "λ": f"{current_lambda:.4f}",
            })

        return {
            "recon_loss": total_recon / total_samples,
            "domain_loss": total_domain / total_samples,
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        执行一个验证轮次。

        返回
        ----
        dict[str, float]
            验证集的平均损失与域分类准确率。
        """
        self.model.eval()
        total_recon = 0.0
        total_domain = 0.0
        correct_domain = 0
        total_samples = 0

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [DANN Val]")
        for batch_x, batch_domain in pbar:
            batch_x = batch_x.to(self.device)
            batch_domain = batch_domain.to(self.device)
            bs = batch_x.size(0)
            total_samples += bs

            (x_recon, _), domain_logits = self.model(batch_x)

            recon_loss = self.recon_criterion(x_recon, batch_x)
            domain_loss = self.domain_criterion(domain_logits, batch_domain)

            total_recon += recon_loss.item() * bs
            total_domain += domain_loss.item() * bs

            # 域分类准确率
            pred = domain_logits.argmax(dim=1)
            correct_domain += (pred == batch_domain).sum().item()

            pbar.set_postfix({
                "recon": f"{recon_loss.item():.2f}",
                "domain_acc": f"{(pred == batch_domain).float().mean().item():.2f}",
            })

        return {
            "recon_loss": total_recon / total_samples,
            "domain_loss": total_domain / total_samples,
            "domain_accuracy": correct_domain / total_samples,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        执行完整的 DANN 训练循环。

        参数
        ----
        train_loader : DataLoader
            训练数据加载器 (需包含 domain 标签)。
        val_loader : DataLoader, 可选
            验证数据加载器 (需包含 domain 标签)。

        返回
        ----
        dict[str, list[float]]
            训练历史。
        """
        # 模型绑定 (惰性初始化)
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(self.device)
        self.model.bind(sample_batch)

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 训练
            train_stats = self.train_epoch(train_loader)
            self.history["train_recon_loss"].append(train_stats["recon_loss"])
            self.history["train_domain_loss"].append(train_stats["domain_loss"])

            # 验证
            if val_loader is not None:
                val_stats = self.validate_epoch(val_loader)
                self.history["val_recon_loss"].append(val_stats["recon_loss"])
                self.history["val_domain_accuracy"].append(val_stats["domain_accuracy"])
                monitor_loss = val_stats["recon_loss"]
                domain_acc_str = f"{val_stats['domain_accuracy']:.3f}"
            else:
                val_stats = {"recon_loss": float("nan"), "domain_accuracy": float("nan")}
                monitor_loss = train_stats["recon_loss"]
                domain_acc_str = "N/A"

            # 学习率调度
            self.scheduler_ae.step(monitor_loss)
            self.scheduler_domain.step(monitor_loss)

            # 记录当前 λ
            avg_lambda = self.model.grl.lambda_.item()
            self.history["lambda_values"].append(avg_lambda)

            # 日志
            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"recon: {train_stats['recon_loss']:.4f} → {val_stats['recon_loss']:.4f} | "
                f"domain_loss: {train_stats['domain_loss']:.4f} | "
                f"domain_acc: {domain_acc_str} | "
                f"λ: {avg_lambda:.4f} | "
                f"lr: {self.optimizer_ae.param_groups[0]['lr']:.2e}"
            )

            # 保存最佳模型
            if val_stats['recon_loss'] < best_val_loss:
                best_val_loss = val_stats['recon_loss']
                self.save_checkpoint("best_dann_model.pt")

            # Early Stopping
            if self.early_stopping(monitor_loss):
                print(f"Early Stopping 触发于 Epoch {epoch + 1} (best val_recon: {best_val_loss:.4f})")
                break

        # 保存最终检查点
        self.save_checkpoint("last_dann_model.pt")
        return self.history

    def save_checkpoint(self, filename: str) -> str:
        """保存模型检查点。"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_ae_state_dict": self.optimizer_ae.state_dict(),
                "optimizer_domain_state_dict": self.optimizer_domain.state_dict(),
                "scheduler_ae_state_dict": self.scheduler_ae.state_dict(),
                "scheduler_domain_state_dict": self.scheduler_domain.state_dict(),
                "early_stopping": self.early_stopping.state_dict(),
                "history": self.history,
            },
            path,
        )
        return path

    def load_checkpoint(self, filename: str) -> int:
        """从检查点恢复训练状态。"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_ae.load_state_dict(checkpoint["optimizer_ae_state_dict"])
        self.optimizer_domain.load_state_dict(checkpoint["optimizer_domain_state_dict"])
        if "scheduler_ae_state_dict" in checkpoint:
            self.scheduler_ae.load_state_dict(checkpoint["scheduler_ae_state_dict"])
            self.scheduler_domain.load_state_dict(checkpoint["scheduler_domain_state_dict"])
        if "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        self.current_epoch = checkpoint.get("epoch", -1) + 1
        return self.current_epoch
