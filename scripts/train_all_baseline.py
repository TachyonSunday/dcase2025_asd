"""
DCASE 2025 Task 2 Baseline v4.
改进: 实时进度显示 + 每10 epoch保存 + 断点续训 + ETA预估。
"""

import os, sys, yaml, glob, json, argparse, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================
class FrameStackDataset(Dataset):
    def __init__(self, data_dir, n_frames=5, file_pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        self.n_frames = n_frames
        self.indices = []
        for fi, fpath in enumerate(self.files):
            spec = torch.load(fpath, weights_only=True)
            T = spec.shape[-1]
            for t in range(0, max(0, T - n_frames + 1), 1):
                self.indices.append((fi, t))
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        fi, t = self.indices[idx]
        spec = torch.load(self.files[fi], weights_only=True)
        return spec[0, :, t:t + self.n_frames].flatten(), 0

class PerFileTestDataset(Dataset):
    def __init__(self, data_dir, n_frames=5, label=0, file_pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        self.n_frames, self.label = n_frames, label
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        spec = torch.load(self.files[idx], weights_only=True)
        T = spec.shape[-1]; n = max(0, T - self.n_frames + 1)
        vecs = [spec[0, :, t:t + self.n_frames].flatten() for t in range(n)] or [torch.zeros(640)]
        domain = "target" if "_target_" in os.path.basename(self.files[idx]) else "source"
        return torch.stack(vecs, dim=0), self.label, domain


# ============================================================
# Model
# ============================================================
class BaselineAE(nn.Module):
    def __init__(self, input_dim=640, latent_dim=8):
        super().__init__()
        bn = {"momentum": 0.01, "eps": 1e-3}
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.BatchNorm1d(latent_dim, **bn), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, **bn), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x): z = self.encoder(x); return self.decoder(z), z


# ============================================================
# Training (with periodic save + live progress + resume)
# ============================================================
def train_and_eval(machine_type, processed_root, result_dir, epochs=100, device="cuda"):
    os.makedirs(result_dir, exist_ok=True)
    n_frames, input_dim = 5, 640
    proc = os.path.join(processed_root, machine_type)

    # Datasets
    train_ds = FrameStackDataset(os.path.join(proc, "train"), n_frames=n_frames)
    test_src_n = PerFileTestDataset(os.path.join(proc, "test"), label=0, file_pattern="*source*normal*.pt")
    test_src_a = PerFileTestDataset(os.path.join(proc, "test"), label=1, file_pattern="*source*anomaly*.pt")
    test_tgt_n = PerFileTestDataset(os.path.join(proc, "test"), label=0, file_pattern="*target*normal*.pt")
    test_tgt_a = PerFileTestDataset(os.path.join(proc, "test"), label=1, file_pattern="*target*anomaly*.pt")

    print(f"\n{'='*60}")
    print(f"[{machine_type}] Train: {len(train_ds):,} frames | "
          f"Test: src_N={len(test_src_n)} src_A={len(test_src_a)} tgt_N={len(test_tgt_n)} tgt_A={len(test_tgt_a)}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    # Model + optimizer
    model = BaselineAE(input_dim=input_dim, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # ---- Resume support ----
    ckpt_path = os.path.join(result_dir, "checkpoint.pt")
    start_epoch, best_loss, best_epoch, history = 0, float("inf"), 0, []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"[{machine_type}] Resumed from epoch {start_epoch} (best loss={best_loss:.4f} @ {best_epoch})")

    # ---- Training loop ----
    eta_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        total_loss, n_batches = 0.0, len(train_loader)
        pbar = tqdm(train_loader, desc=f"[{machine_type}] Epoch {epoch+1:3d}/{epochs}", leave=False)

        for x, _ in pbar:
            x = x.to(device)
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        history.append(float(avg_loss))
        epoch_time = time.time() - epoch_start

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)

        # Periodic save (every 10 epochs, prevents loss on crash)
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss, "best_epoch": best_epoch, "history": history,
            }, ckpt_path.replace(".pt", f"_epoch{epoch+1}.pt"))

        # ETA
        eta_per_epoch = (time.time() - eta_start) / max(1, epoch + 1 - start_epoch)
        eta_remaining = eta_per_epoch * (epochs - epoch - 1)
        print(f"[{machine_type}] Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} best={best_loss:.4f}@{best_epoch} | "
              f"{epoch_time:.0f}s/ep | ETA: {eta_remaining/60:.0f}min")

    # ---- Evaluation ----
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    def score_files(dataset):
        scores, labels = [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                x_all, label, _ = dataset[i]
                if x_all.size(0) == 0: continue
                x_recon, _ = model(x_all.to(device))
                scores.append(((x_recon - x_all.to(device))**2).mean().item())
                labels.append(label)
        return np.array(scores), np.array(labels)

    sn, ln = score_files(test_src_n); sa, la = score_files(test_src_a)
    tn, ltn = score_files(test_tgt_n); ta, lta = score_files(test_tgt_a)

    y_src = np.concatenate([ln, la, lta]); s_src = np.concatenate([sn, sa, ta])
    y_tgt = np.concatenate([ltn, la, lta]); s_tgt = np.concatenate([tn, sa, ta])
    y_all = np.concatenate([ln, ltn, la, lta]); s_all = np.concatenate([sn, tn, sa, ta])

    auc_src = roc_auc_score(y_src, s_src) if len(np.unique(y_src)) > 1 else np.nan
    auc_tgt = roc_auc_score(y_tgt, s_tgt) if len(np.unique(y_tgt)) > 1 else np.nan
    auc_all = roc_auc_score(y_all, s_all)

    print(f"[{machine_type}] AUC src={auc_src:.4f} tgt={auc_tgt:.4f} all={auc_all:.4f}")
    print(f"[{machine_type}] Normal: src={np.mean(sn):.2f}±{np.std(sn):.2f} tgt={np.mean(tn):.2f}±{np.std(tn):.2f} | Anomaly: {np.mean(sa):.2f}±{np.std(sa):.2f}")

    result = {"machine": machine_type, "auc_source": float(auc_src), "auc_target": float(auc_tgt),
              "auc_overall": float(auc_all), "best_epoch": best_epoch, "best_loss": float(best_loss)}
    with open(os.path.join(result_dir, "eval.json"), "w") as f: json.dump(result, f, indent=2)
    with open(os.path.join(result_dir, "train_history.json"), "w") as f: json.dump({"history": history}, f)
    return result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp", type=str, default="baseline_v4")
    args = parser.parse_args()

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    if args.machine: machine_types = [args.machine]

    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    processed_root = os.path.join(PROJECT_ROOT, "data", "processed", args.exp)
    results_root = os.path.join(PROJECT_ROOT, "results", args.exp)

    # Feature processing
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline(cfg_path)
    pipeline.norm_mean = None; pipeline.norm_std = None

    for mt in machine_types:
        train_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "train")
        test_dir = os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "test")
        if not os.path.exists(train_dir):
            print(f"WARNING: Skip {mt}, not found"); continue
        for dir_name, src_dir in [("train", train_dir), ("test", test_dir)]:
            out_dir = os.path.join(processed_root, mt, dir_name)
            if not os.path.exists(out_dir) or len(os.listdir(out_dir)) == 0:
                print(f"Processing {mt}/{dir_name}...")
                pipeline.process_directory(src_dir, out_dir)

    # Train & eval
    all_results = []
    for mt in machine_types:
        if not os.path.exists(os.path.join(PROJECT_ROOT, "data", "raw", mt, mt, "train")): continue
        result_dir = os.path.join(results_root, mt)
        try:
            r = train_and_eval(mt, processed_root, result_dir, epochs=args.epochs)
            all_results.append(r)
        except Exception as e:
            print(f"FAILED {mt}: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}\nExperiment: {args.exp}\n{'Machine':15s} {'AUC(src)':>10s} {'AUC(tgt)':>10s} {'AUC(all)':>10s}\n{'='*70}")
    for r in all_results:
        print(f"{r['machine']:15s} {r['auc_source']:10.4f} {r['auc_target']:10.4f} {r['auc_overall']:10.4f}")
    if all_results:
        with open(os.path.join(results_root, "summary.json"), "w") as f:
            json.dump({"results": all_results}, f, indent=2)
        print(f"Saved to {results_root}/")


if __name__ == "__main__":
    main()
