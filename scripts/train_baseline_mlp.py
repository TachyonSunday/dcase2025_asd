"""
DCASE 2025 Task 2 Baseline: 帧堆叠 MLP 自编码器。
参考 nttcslab/dcase2023_task2_baseline_ae, 使用 5 帧堆叠作为输入。
"""

import os, sys, yaml, tempfile, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# 1) 帧堆叠 Dataset: 连续 5 帧 → 1 个输入向量
# ============================================================
class FrameStackDataset(Dataset):
    def __init__(self, data_dir, n_frames=5, label=0, file_pattern="*.pt",
                 max_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        if max_files:
            self.files = self.files[:max_files]
        self.n_frames = n_frames
        self.label = label

        # 预建索引
        self.indices = []
        for fi, fpath in enumerate(self.files):
            spec = torch.load(fpath, weights_only=True)  # (1, n_mels, T)
            T = spec.shape[-1]
            if T >= n_frames:
                for t in range(0, T - n_frames + 1, 1):
                    self.indices.append((fi, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fi, t = self.indices[idx]
        spec = torch.load(self.files[fi], weights_only=True)
        stack = spec[0, ::4, t:t + self.n_frames]  # 降采样: 128→32 mel 频带
        return stack.flatten(), self.label  # (32 * n_frames,)


# ============================================================
# 2) MLP 自编码器
# ============================================================
class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128, 128), latent_dim=8):
        super().__init__()
        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.BatchNorm1d(hidden_dims[-1]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1, 0, -1):
            dec_layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i-1]), nn.BatchNorm1d(hidden_dims[i-1]), nn.ReLU()])
        dec_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_mu(h)
        x_recon = self.decoder(z)
        return x_recon, z


# ============================================================
# 3) 训练 + 评估
# ============================================================
def main():
    # 读取配置
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # mel 频带数 — 降采样到 32 频带 (匹配 DCASE baseline)
    n_mels = cfg["mel"]["n_mels"] // 4  # 128 → 32
    n_frames_stack = 5
    input_dim = n_mels * n_frames_stack  # 32 * 5 = 160

    processed = os.path.join(PROJECT_ROOT, "data/processed/dcase2025/ToyCar")

    # 数据集
    train_ds = FrameStackDataset(os.path.join(processed, "train"), n_frames=n_frames_stack, label=0)
    test_n_ds = FrameStackDataset(os.path.join(processed, "test"), n_frames=n_frames_stack, label=0, file_pattern="*normal*.pt")
    test_a_ds = FrameStackDataset(os.path.join(processed, "test"), n_frames=n_frames_stack, label=1, file_pattern="*anomaly*.pt")

    print(f"训练: {len(train_ds)} 帧堆叠")
    print(f"测试正常: {len(test_n_ds)}, 测试异常: {len(test_a_ds)}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader_n = DataLoader(test_n_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader_a = DataLoader(test_a_ds, batch_size=256, shuffle=False, num_workers=2)

    # 模型
    model = MLPAutoEncoder(input_dim=input_dim, hidden_dims=(128, 128, 128), latent_dim=8)
    device = torch.device("cuda")
    model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    epochs = 50
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "checkpoints/best_mlp_model.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # ---- 评估 ----
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "checkpoints/best_mlp_model.pt")))
    model.eval()

    def get_scores(dataloader):
        scores = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                x_recon, _ = model(x)
                mse = ((x_recon - x) ** 2).mean(dim=1)
                scores.extend(mse.cpu().tolist())
        return np.array(scores)

    normal_scores = get_scores(test_loader_n)
    anomaly_scores = get_scores(test_loader_a)

    # ---- 文件级聚合 ----
    # 按文件分组帧分数
    def file_scores(dataset, n_frames):
        file_score_list = []
        labels = []
        current_file = -1
        buf = []
        for idx in range(len(dataset)):
            fi, _ = dataset.indices[idx]
            if fi != current_file and buf:
                file_score_list.append(np.mean(buf))
                labels.append(dataset.label)
                buf = []
            current_file = fi
            # 获取分数
            spec = torch.load(dataset.files[fi], weights_only=True)[0]  # (n_mels, T)
            t = dataset.indices[idx][1]
            stack = spec[::4, t:t+n_frames].flatten().unsqueeze(0).to(device)
            with torch.no_grad():
                xr, _ = model(stack)
                mse = ((xr - stack) ** 2).mean().item()
            buf.append(mse)
        if buf:
            file_score_list.append(np.mean(buf))
            labels.append(dataset.label)
        return np.array(file_score_list), np.array(labels)

    file_scores_n, lbls_n = file_scores(test_n_ds, n_frames_stack)
    file_scores_a, lbls_a = file_scores(test_a_ds, n_frames_stack)

    all_file_scores = np.concatenate([file_scores_n, file_scores_a])
    all_file_labels = np.concatenate([lbls_n, lbls_a])

    auc = roc_auc_score(all_file_labels, all_file_scores)

    # 最佳 F1 阈值
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(all_file_labels, all_file_scores)
    f1 = 2 * p * r / (p + r + 1e-8)
    best_idx = np.argmax(f1)
    threshold = t[best_idx] if best_idx < len(t) else t[-1]

    preds = (all_file_scores > threshold).astype(int)
    tp = ((preds == 1) & (all_file_labels == 1)).sum()
    tn = ((preds == 0) & (all_file_labels == 0)).sum()
    fp = ((preds == 1) & (all_file_labels == 0)).sum()
    fn = ((preds == 0) & (all_file_labels == 1)).sum()

    print(f"\n{'='*50}")
    print(f"MLP Baseline 评估结果")
    print(f"{'='*50}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1:        {f1[best_idx]:.4f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(file_scores_n, bins=30, alpha=0.6, label="Normal")
    ax[0].hist(file_scores_a, bins=30, alpha=0.6, label="Anomaly")
    ax[0].axvline(threshold, color="red", ls="--", lw=2, label=f"Thr={threshold:.4f}")
    ax[0].set_title(f"ToyCar MLP Baseline (AUC={auc:.3f}, F1={f1[best_idx]:.3f})")
    ax[0].legend()

    sorted_idx = np.argsort(all_file_scores)
    colors = ["blue" if all_file_labels[i] == 0 else "red" for i in sorted_idx]
    ax[1].scatter(range(len(all_file_scores)), all_file_scores[sorted_idx], c=colors, s=8, alpha=0.5)
    ax[1].axhline(threshold, color="red", ls="--", lw=1.5)
    ax[1].set_title("Sorted Scores")

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "logs/toycar_mlp_results.png"), dpi=120)
    print(f"Chart saved to logs/toycar_mlp_results.png")


if __name__ == "__main__":
    main()
