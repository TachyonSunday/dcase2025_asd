"""
DCASE 2025 Task 2 官方 Baseline 复现脚本。
MLP 自编码器 (640-dim 输入, 8-dim 瓶颈) + 无归一化 + 马氏距离评分 + 全 7 机器类型。

每次训练结果独立保存到 results/<exp_name>/<machine_type>/ 下，永不覆盖旧数据。
"""

import os, sys, yaml, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis


# ============================================================
# 1) 帧堆叠 Dataset (无归一化, 读原始 dB 值)
# ============================================================
class FrameStackDataset(Dataset):
    """5 帧连续堆叠 → 640 维向量。"""
    def __init__(self, data_dir, n_frames=5, label=0, file_pattern="*.pt", max_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        if max_files:
            self.files = self.files[:max_files]
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
# 2) MLP 自编码器 (精确匹配官方: 5 层 Linear128, BN(0.01, 1e-3), 瓶颈 8)
# ============================================================
class BaselineAE(nn.Module):
    def __init__(self, input_dim=640, latent_dim=8):
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

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# ============================================================
# 3) 训练 + 马氏距离评估 (单机器类型)
# ============================================================
def train_and_eval(machine_type, processed_root, result_dir, cfg, device="cuda"):
    """训练并评估单个机器类型, 所有产物写入 result_dir/。"""
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training: {machine_type} → {result_dir}")
    print(f"{'='*60}")

    n_frames_stack = 5
    input_dim = cfg["mel"]["n_mels"] * n_frames_stack  # 640

    proc = os.path.join(processed_root, machine_type)
    train_ds = FrameStackDataset(os.path.join(proc, "train"), n_frames=n_frames_stack, label=0)
    test_n_ds = FrameStackDataset(os.path.join(proc, "test"), n_frames=n_frames_stack, label=0, file_pattern="*normal*.pt")
    test_a_ds = FrameStackDataset(os.path.join(proc, "test"), n_frames=n_frames_stack, label=1, file_pattern="*anomaly*.pt")

    print(f"  Train frames: {len(train_ds)}, Test N: {len(test_n_ds)}, Test A: {len(test_a_ds)}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_n_loader = DataLoader(test_n_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_a_loader = DataLoader(test_a_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = BaselineAE(input_dim=input_dim, latent_dim=8).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    # 保存训练历史
    with open(os.path.join(result_dir, "train_history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_loss": float(best_loss)}, f)

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(result_dir, "checkpoint.pt")))
    model.eval()

    # ---- 马氏距离统计量 ----
    all_z = []
    with torch.no_grad():
        for x, _ in train_loader:
            _, z = model(x.to(device))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    mu = all_z.mean(axis=0)
    cov = np.cov(all_z.T)
    cov_inv = np.linalg.inv(cov + 1e-4 * np.eye(cov.shape[0]))

    # ---- 评估: MSE + 马氏距离 ----
    def compute_scores(dataloader):
        mse_scores, mah_scores, all_z_list = [], [], []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                x_recon, z = model(x)
                mse = ((x_recon - x) ** 2).mean(dim=1).cpu().numpy()
                mse_scores.extend(mse)
                all_z_list.append(z.cpu().numpy())
        all_z_arr = np.concatenate(all_z_list, axis=0)
        for zi in all_z_arr:
            mah_scores.append(mahalanobis(zi, mu, cov_inv))
        return np.array(mse_scores), np.array(mah_scores)

    mse_n, mah_n = compute_scores(test_n_loader)
    mse_a, mah_a = compute_scores(test_a_loader)

    # ---- 文件级聚合 ----
    def file_scores(dataloader, mse_all, mah_all):
        mse_file, mah_file, labels = [], [], []
        offset = 0
        for batch_x, _ in dataloader:
            bs = batch_x.size(0)
            mse_file.append(float(mse_all[offset:offset+bs].mean()))
            mah_file.append(float(mah_all[offset:offset+bs].mean()))
            labels.append(dataloader.dataset.label)
            offset += bs
        return np.array(mse_file), np.array(mah_file), np.array(labels)

    mse_nf, mah_nf, lbl_n = file_scores(test_n_loader, mse_n, mah_n)
    mse_af, mah_af, lbl_a = file_scores(test_a_loader, mse_a, mah_a)

    mse_scores = np.concatenate([mse_nf, mse_af])
    mah_scores = np.concatenate([mah_nf, mah_af])
    labels = np.concatenate([lbl_n, lbl_a])

    auc_mse = roc_auc_score(labels, mse_scores)
    auc_mah = roc_auc_score(labels, mah_scores)

    p, r, t = precision_recall_curve(labels, mah_scores)
    f1_vals = 2 * p * r / (p + r + 1e-8)
    best_idx = np.argmax(f1_vals)
    best_t = float(t[best_idx]) if best_idx < len(t) else float(t[-1])
    best_f1 = float(f1_vals[best_idx])

    preds = (mah_scores > best_t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    print(f"  AUC (MSE):          {auc_mse:.4f}")
    print(f"  AUC (Mahalanobis):  {auc_mah:.4f}")
    print(f"  F1  (Mahalanobis):  {best_f1:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mah_nf, bins=30, alpha=0.6, label=f"Normal ({len(mah_nf)})")
    axes[0].hist(mah_af, bins=30, alpha=0.6, label=f"Anomaly ({len(mah_af)})")
    axes[0].axvline(best_t, color="r", ls="--", lw=2, label=f"Thr={best_t:.2f}")
    axes[0].set_title(f"{machine_type} Mahalanobis (AUC={auc_mah:.3f}, F1={best_f1:.3f})")
    axes[0].legend()

    sidx = np.argsort(mah_scores)
    colors = ["blue" if labels[i] == 0 else "red" for i in sidx]
    axes[1].scatter(range(len(mah_scores)), mah_scores[sidx], c=colors, s=5, alpha=0.5)
    axes[1].axhline(best_t, color="r", ls="--", lw=1.5)
    axes[1].set_title("Sorted Mahalanobis Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "scores.png"), dpi=120)
    plt.close()

    # ---- 保存评估结果 JSON ----
    eval_result = {
        "machine": machine_type,
        "auc_mse": float(auc_mse), "auc_mah": float(auc_mah), "f1_mah": best_f1,
        "threshold": best_t, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "train_frames": len(train_ds), "test_n_files": len(mah_nf), "test_a_files": len(mah_af),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(result_dir, "eval.json"), "w") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


# ============================================================
# 4) 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DCASE 2025 T2 Baseline Training")
    parser.add_argument("--machine", type=str, default=None, help="单机器测试 (如 ToyCar)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp", type=str, default="baseline_v1", help="实验名称 (用于结果文件夹命名)")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = args.epochs

    exp_name = args.exp
    processed_root = os.path.join(PROJECT_ROOT, "data", "processed", exp_name)
    results_root = os.path.join(PROJECT_ROOT, "results", exp_name)

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    if args.machine:
        machine_types = [args.machine]

    # ---- 特征处理 (每个实验独立存储, 不覆盖) ----
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline(cfg_path)
    pipeline.norm_mean = None  # 强制无归一化
    pipeline.norm_std = None

    for mt in machine_types:
        train_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "train")
        test_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "test")
        if not os.path.exists(train_dir):
            print(f"⚠️ 跳过 {mt}: 原始数据不存在")
            continue
        out_train = os.path.join(processed_root, mt, "train")
        out_test = os.path.join(processed_root, mt, "test")
        if not os.path.exists(out_train) or len(os.listdir(out_train)) == 0:
            print(f"\n处理 {mt} 训练集 (fmin=0, 无归一化) → {out_train}")
            pipeline.process_directory(train_dir, out_train)
        if not os.path.exists(out_test) or len(os.listdir(out_test)) == 0:
            print(f"处理 {mt} 测试集 → {out_test}")
            pipeline.process_directory(test_dir, out_test)

    # ---- 训练 + 评估 ----
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
            print(f"❌ {mt} 失败: {e}")
            import traceback
            traceback.print_exc()

    # ---- 汇总 ----
    print(f"\n{'='*70}")
    print(f"{'Machine':15s} {'AUC(MSE)':>10s} {'AUC(Mah)':>10s} {'F1(Mah)':>10s} {'Train':>8s} {'BestEp':>8s}")
    print(f"{'='*70}")
    for r in all_results:
        print(f"{r['machine']:15s} {r['auc_mse']:10.4f} {r['auc_mah']:10.4f} {r['f1_mah']:10.4f} {r['train_frames']:8d} {r['best_epoch']:8d}")
    if all_results:
        avg_auc = np.mean([r["auc_mah"] for r in all_results])
        print(f"{'---':70s}")
        print(f"{'AVERAGE':15s} {'':10s} {avg_auc:10.4f}")
        print(f"{'='*70}")
        # 保存汇总
        with open(os.path.join(results_root, "summary.json"), "w") as f:
            json.dump({"results": all_results, "average_auc_mah": float(avg_auc),
                        "exp_name": exp_name, "config": cfg}, f, indent=2)
        print(f"\n结果已保存至: {results_root}/")


if __name__ == "__main__":
    main()
