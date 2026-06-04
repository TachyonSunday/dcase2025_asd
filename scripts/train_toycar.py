"""
ToyCar ConvAE 训练脚本 — 在 DCASE 2025 Task 2 开发集上训练和评估。
"""

import os
import sys
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.features.dataset import MachineSoundDataset
from src.models.conv_ae import ConvAE
from src.utils.trainer import Trainer
from src.utils.evaluator import Evaluator


def main():
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # 使用更合理的训练配置
    cfg["train"]["epochs"] = 30
    cfg["train"]["batch_size"] = 128

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        train_config = tf.name

    frame_cfg = cfg["frame"]
    window_size = frame_cfg["window_size"]
    hop_size = frame_cfg["hop_size"]

    processed_dir = os.path.join(PROJECT_ROOT, "data/processed/dcase2025/ToyCar")

    # ---- 创建数据集 ----
    train_ds = MachineSoundDataset(
        os.path.join(processed_dir, "train"),
        window_size=window_size, hop_size=hop_size, label=0,
    )
    test_normal_ds = MachineSoundDataset(
        os.path.join(processed_dir, "test"),
        window_size=window_size, hop_size=hop_size, label=0,
        file_pattern="*normal*.pt",
    )
    test_anomaly_ds = MachineSoundDataset(
        os.path.join(processed_dir, "test"),
        window_size=window_size, hop_size=hop_size, label=1,
        file_pattern="*anomaly*.pt",
    )
    test_ds = ConcatDataset([test_normal_ds, test_anomaly_ds])

    print(f"训练帧块: {train_ds.num_frames}")
    print(f"测试正常帧块: {test_normal_ds.num_frames}")
    print(f"测试异常帧块: {test_anomaly_ds.num_frames}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )
    # 验证集: 取 20% 训练数据
    val_size = max(1, int(len(train_ds) * 0.2))
    val_indices = torch.randperm(len(train_ds))[:val_size].tolist()
    val_subset = torch.utils.data.Subset(train_ds, val_indices)
    val_loader = DataLoader(
        val_subset, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    # ---- 训练 ConvAE ----
    print("\n" + "=" * 50)
    print("训练 ConvAE on ToyCar")
    print("=" * 50)

    model = ConvAE.from_config(config_path)
    trainer = Trainer(model, config_path=train_config)
    history = trainer.train(train_loader, val_loader=val_loader)

    # ---- 评估 ----
    print("\n" + "=" * 50)
    print("评估")
    print("=" * 50)

    evaluator = Evaluator(model, config_path=train_config)
    evaluator.load_checkpoint(os.path.join(PROJECT_ROOT, "checkpoints/best_model.pt"))

    scores, labels = evaluator.predict(test_loader)
    auc = evaluator.compute_auc(scores, labels)
    threshold = evaluator.find_threshold(scores, labels, method="best_f1")

    # 分类统计
    normal_scores = [s for s, l in zip(scores, labels) if l == 0]
    anomaly_scores = [s for s, l in zip(scores, labels) if l == 1]

    tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s > threshold)
    tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s <= threshold)
    fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s > threshold)
    fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s <= threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"📊 评估结果")
    print(f"{'='*50}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  阈值:      {threshold:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 分数分布
    axes[0].hist(normal_scores, bins=30, alpha=0.6, label=f"正常 (n={len(normal_scores)})")
    axes[0].hist(anomaly_scores, bins=30, alpha=0.6, label=f"异常 (n={len(anomaly_scores)})")
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"阈值={threshold:.4f}")
    axes[0].set_xlabel("异常分数 (MSE)")
    axes[0].set_ylabel("频次")
    axes[0].set_title(f"ToyCar 异常分数分布\nAUC={auc:.3f}, F1={f1:.3f}")
    axes[0].legend()

    # 训练曲线
    axes[1].plot(history["train_loss"], "o-", markersize=3, label="训练损失")
    axes[1].plot(history["val_loss"], "s-", markersize=3, label="验证损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("训练曲线")
    axes[1].legend()

    # 分数排序图
    all_scores_sorted = sorted(zip(scores, labels), key=lambda x: x[0])
    x_axis = list(range(len(all_scores_sorted)))
    colors = ["blue" if l == 0 else "red" for _, l in all_scores_sorted]
    axes[2].scatter(x_axis, [s for s, _ in all_scores_sorted], c=colors, s=5, alpha=0.5)
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("样本 (按分数排序)")
    axes[2].set_ylabel("异常分数")
    axes[2].set_title("排序异常分数")

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    plt.savefig(os.path.join(PROJECT_ROOT, "logs/toycar_results.png"), dpi=120)
    print(f"\n📈 图表已保存至 logs/toycar_results.png")

    # 清理临时配置
    os.unlink(train_config)


if __name__ == "__main__":
    main()
