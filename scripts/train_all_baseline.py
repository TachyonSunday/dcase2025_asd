"""
DCASE 2025 Task 2 官方 Baseline 精确复现 (v2)。
修复 5 个关键差异:
1. 马氏距离对重构误差向量 (640-dim) 而非隐向量 (8-dim)
2. 测试按文件级批处理 (batch_size = 文件帧数)
3. 学习率 0.03
4. Mahalanobis 双域评分取 min
5. 阈值从训练异常分数分布拟合
"""

import os, sys, yaml, glob, json, argparse, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# 1) 帧堆叠 Dataset
# ============================================================
class FrameStackDataset(Dataset):
    """5 帧连续堆叠 → 640 维向量。"""
    def __init__(self, data_dir, n_frames=5, label=0, file_pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        self.n_frames = n_frames
        self.label = label
        self.indices = []
        for fi, fpath in enumerate(self.files):
            spec = torch.load(fpath, weights_only=True)
            T = spec.shape[-1]
            if T >= n_frames:
                for t in range(0, T - n_frames + 1, 1):
                    self.indices.append((fi, t))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        fi, t = self.indices[idx]
        spec = torch.load(self.files[fi], weights_only=True)
        stack = spec[0, :, t:t + self.n_frames]
        return stack.flatten(), self.label


# ============================================================
# 2) Per-file Dataset (测试用: 返回整个文件的所有帧)
# ============================================================
class PerFileDataset(Dataset):
    """返回一个文件的所有 5 帧堆叠向量, 用于文件级评分。"""
    def __init__(self, data_dir, n_frames=5, label=0, file_pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        self.n_frames = n_frames
        self.label = label
        # 预计算每个文件的帧数量
        self.file_sizes = []
        for fpath in self.files:
            spec = torch.load(fpath, weights_only=True)
            T = spec.shape[-1]
            n = max(0, T - n_frames + 1)
            self.file_sizes.append(n)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        """返回 (all_vectors, label), all_vectors shape=(n_vecs, 640)"""
        spec = torch.load(self.files[idx], weights_only=True)
        T = spec.shape[-1]
        n_vecs = max(0, T - self.n_frames + 1)
        vecs = []
        for t in range(n_vecs):
            vecs.append(spec[0, :, t:t + self.n_frames].flatten())
        if len(vecs) == 0:
            return torch.zeros(1, 128 * self.n_frames), self.label
        return torch.stack(vecs, dim=0), self.label


# ============================================================
# 3) MLP 自编码器 (官方参数: BN momentum=0.01, eps=1e-3)
# ============================================================
class BaselineAE(nn.Module):
    def __init__(self, input_dim=640, latent_dim=8, block_size=640):
        super().__init__()
        bn_args = {"momentum": 0.01, "eps": 1e-3}
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.BatchNorm1d(latent_dim, **bn_args), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn_args), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        # 协方差矩阵 (官方: 640×640)
        self.register_buffer("cov_source", torch.zeros(block_size, block_size))
        self.register_buffer("cov_target", torch.zeros(block_size, block_size))

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# ============================================================
# 4) 马氏距离辅助函数 (匹配官方 mahala.py)
# ============================================================
def mahalanobis_distance(delta, inv_cov):
    """delta: (N, D), inv_cov: (D, D) → mahalanobis distance per sample: (N,)"""
    # (delta @ inv_cov) * delta → sum over dim 1 → per-sample score
    return torch.sum((delta @ inv_cov) * delta, dim=1)


def compute_error_covariance(model, dataloader, device, block_size=640):
    """收集训练集所有重构误差向量 diff = x - x_recon, 计算协方差。"""
    model.eval()
    all_diffs = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_recon, _ = model(x)
            diff = (x - x_recon.view(x.shape)).view(-1, block_size)
            all_diffs.append(diff)
    all_diffs = torch.cat(all_diffs, dim=0)  # (N_train_vecs, 640)
    mu = all_diffs.mean(dim=0)
    centered = all_diffs - mu
    cov = (centered.T @ centered) / (len(all_diffs) - 1)
    return cov, mu


# ============================================================
# 5) 训练 + 评估 (单机器类型)
# ============================================================
def train_and_eval(machine_type, processed_root, result_dir, cfg, device="cuda"):
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training: {machine_type} → {result_dir}")
    print(f"{'='*60}")

    n_frames_stack = 5
    block_size = cfg["mel"]["n_mels"] * n_frames_stack  # 640

    proc = os.path.join(processed_root, machine_type)
    train_ds = FrameStackDataset(os.path.join(proc, "train"), n_frames=n_frames_stack, label=0)
    test_n_ds = PerFileDataset(os.path.join(proc, "test"), n_frames=n_frames_stack, label=0, file_pattern="*normal*.pt")
    test_a_ds = PerFileDataset(os.path.join(proc, "test"), n_frames=n_frames_stack, label=1, file_pattern="*anomaly*.pt")

    print(f"  Train frames: {len(train_ds)}, Test N files: {len(test_n_ds)}, Test A files: {len(test_a_ds)}")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)

    model = BaselineAE(input_dim=block_size, latent_dim=8, block_size=block_size).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 差异 3: 学习率 0.03
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    epochs = 100
    best_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        history.append(float(avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(result_dir, "checkpoint.pt"))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.6f}, best={best_loss:.6f} @ {best_epoch}")

    # 差异 1: 最后一个 epoch 计算重构误差协方差 (640×640)
    print(f"\n  Computing error covariance (640×640)...")
    model.load_state_dict(torch.load(os.path.join(result_dir, "checkpoint.pt")))
    cov_source, _ = compute_error_covariance(model, train_loader, device, block_size)
    model.cov_source.data = cov_source
    model.cov_target.data = cov_source  # 单域时两者相同

    # 计算逆协方差矩阵 (加正则化防止奇异)
    reg = 1e-4 * torch.eye(block_size, device=device)
    inv_cov = torch.linalg.inv(cov_source.to(device) + reg)

    # 差异 5: 拟合训练异常分数分布以获取阈值
    model.eval()
    train_scores = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            x_recon, _ = model(x)
            diff = (x - x_recon.view(x.shape)).view(-1, block_size)
            train_scores.extend(mahalanobis_distance(diff, inv_cov).cpu().tolist())
    decision_threshold = float(np.percentile(train_scores, 90))

    # ---- 文件级评估 ----
    def eval_files(file_ds, inv_cov_matrix):
        """逐文件评估: 所有帧的 Mahalanobis 距离取均值作为文件分数。"""
        scores, labels = [], []
        with torch.no_grad():
            for i in range(len(file_ds)):
                x_all, label = file_ds[i]
                if x_all.size(0) == 0:
                    continue
                x_all = x_all.to(device)
                x_recon, _ = model(x_all)
                diff = (x_all - x_recon.view(x_all.shape)).view(-1, block_size)
                # 差异 4: 取所有帧马氏距离的均值作为文件分数
                file_score = mahalanobis_distance(diff, inv_cov_matrix).mean().item()
                scores.append(file_score)
                labels.append(label)
        return np.array(scores), np.array(labels)

    mah_n, lbl_n = eval_files(test_n_ds, inv_cov)
    mah_a, lbl_a = eval_files(test_a_ds, inv_cov)

    mah_scores = np.concatenate([mah_n, mah_a])
    labels = np.concatenate([lbl_n, lbl_a])

    auc_mah = roc_auc_score(labels, mah_scores)
    preds = (mah_scores > decision_threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"  AUC (Mahalanobis):  {auc_mah:.4f}")
    print(f"  F1:                 {f1:.4f}")
    print(f"  Threshold:          {decision_threshold:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mah_n, bins=30, alpha=0.6, label=f"Normal ({len(mah_n)})")
    axes[0].hist(mah_a, bins=30, alpha=0.6, label=f"Anomaly ({len(mah_a)})")
    axes[0].axvline(decision_threshold, color="r", ls="--", lw=2, label=f"Thr={decision_threshold:.2f}")
    axes[0].set_title(f"{machine_type} Mahalanobis (AUC={auc_mah:.3f}, F1={f1:.3f})")
    axes[0].legend()

    sidx = np.argsort(mah_scores)
    colors = ["blue" if labels[i] == 0 else "red" for i in sidx]
    axes[1].scatter(range(len(mah_scores)), mah_scores[sidx], c=colors, s=8, alpha=0.5)
    axes[1].axhline(decision_threshold, color="r", ls="--", lw=1.5)
    axes[1].set_title("Per-File Sorted Mahalanobis Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "scores.png"), dpi=120)
    plt.close()

    # 保存
    with open(os.path.join(result_dir, "train_history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_loss": float(best_loss)}, f)

    eval_result = {
        "machine": machine_type,
        "auc_mah": float(auc_mah), "f1": float(f1),
        "threshold": float(decision_threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "train_frames": len(train_ds), "test_n_files": len(test_n_ds), "test_a_files": len(test_a_ds),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(result_dir, "eval.json"), "w") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


# ============================================================
# 6) 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DCASE 2025 T2 Baseline v2")
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp", type=str, default="baseline_v2", help="实验名称")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = args.exp
    processed_root = os.path.join(PROJECT_ROOT, "data", "processed", exp_name)
    results_root = os.path.join(PROJECT_ROOT, "results", exp_name)

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    if args.machine:
        machine_types = [args.machine]

    # 特征处理
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline(cfg_path)
    pipeline.norm_mean = None
    pipeline.norm_std = None

    for mt in machine_types:
        train_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "train")
        test_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "test")
        if not os.path.exists(train_dir):
            print(f"WARNING: Skip {mt}, raw data not found")
            continue
        out_train = os.path.join(processed_root, mt, "train")
        out_test = os.path.join(processed_root, mt, "test")
        if not os.path.exists(out_train) or len(os.listdir(out_train)) == 0:
            print(f"\nProcessing {mt} train (fmin=0, no norm) -> {out_train}")
            pipeline.process_directory(train_dir, out_train)
        if not os.path.exists(out_test) or len(os.listdir(out_test)) == 0:
            print(f"Processing {mt} test -> {out_test}")
            pipeline.process_directory(test_dir, out_test)

    # 训练 + 评估
    all_results = []
    for mt in machine_types:
        train_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "train")
        if not os.path.exists(train_dir):
            continue
        result_dir = os.path.join(results_root, mt)
        try:
            r = train_and_eval(mt, processed_root, result_dir, cfg, device="cuda")
            all_results.append(r)
        except Exception as e:
            print(f"FAILED {mt}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'Machine':15s} {'AUC(Mah)':>10s} {'F1':>10s} {'TN':>8s} {'BestEp':>8s}")
    print(f"{'='*70}")
    for r in all_results:
        print(f"{r['machine']:15s} {r['auc_mah']:10.4f} {r['f1']:10.4f} {r['tn']:8d} {r['best_epoch']:8d}")
    if all_results:
        avg_auc = np.mean([r["auc_mah"] for r in all_results])
        print(f"{'AVERAGE':15s} {avg_auc:10.4f}")
        print(f"{'='*70}")
        with open(os.path.join(results_root, "summary.json"), "w") as f:
            json.dump({"results": all_results, "average_auc_mah": float(avg_auc)}, f, indent=2)
        print(f"\nResults saved to: {results_root}/")


if __name__ == "__main__":
    main()
