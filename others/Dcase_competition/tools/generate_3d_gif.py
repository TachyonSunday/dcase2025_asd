#!/usr/bin/env python3
"""
tools/generate_3d_gif.py
============================================================
DCASE 2025 Task 2 - Sci-Fi 3D UMAP Rotation GIF
============================================================

Generates a professional animated GIF showing ToyCar feature embeddings
(Layer 2, 12288-D) projected into 3D space via UMAP, with a slow 360deg
rotation on a dark background.

Usage:
    python tools/generate_3d_gif.py

Output:
    ./results/figures/ToyCar_3D_Space.gif

Dependencies:
    pip install umap-learn matplotlib Pillow
"""

import os
import sys
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import umap

# ── 项目根目录 (tools/ 的上一级) ─────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ────────────────────────────────────────────
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "features")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ToyCar_3D_Space.gif")

BEST_LAYER = 2
MACHINE_TYPE = "ToyCar"

# Animation parameters
N_FRAMES = 72          # 72 frames = 5deg per step = smooth 360deg rotation
FPS = 15               # ~4.8 second loop
DPI = 100              # reasonable resolution for GIF

# UMAP parameters
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
UMAP_RANDOM_STATE = 42

# Style
BG_COLOR = "black"
GRID_COLOR = "#1a1a2e"
LABEL_COLOR = "#e0e0e0"

# Category colors (neon / sci-fi palette)
COLOR_SRC_NORMAL = "#00BFFF"   # cyan / deep sky blue
COLOR_TGT_NORMAL = "#39FF14"   # neon green
COLOR_ANOMALY = "#FF073A"      # neon red / pink

MARKER_SRC = "o"
MARKER_TGT = "^"
MARKER_ANOMALY = "X"

SIZE_NORMAL = 8
SIZE_ANOMALY = 40


# ── Data Loading ─────────────────────────────────────────────
def load_layer2_toycar():
    """Load Layer 2 features for ToyCar (train + test combined)."""
    print(f"[1/4] Loading Layer {BEST_LAYER} features from {FEATURES_DIR}...")

    # Load 3D tensors: (11, N, 12288)
    train_emb_all = np.load(os.path.join(FEATURES_DIR, "train_embeddings.npy"))
    test_emb_all = np.load(os.path.join(FEATURES_DIR, "test_embeddings.npy"))

    # Slice Layer 2: shape (N, 12288)
    train_emb = train_emb_all[BEST_LAYER]
    test_emb = test_emb_all[BEST_LAYER]

    # Free memory
    del train_emb_all, test_emb_all

    # Load metadata
    with open(os.path.join(FEATURES_DIR, "train_metadata.pkl"), "rb") as f:
        train_meta = pickle.load(f)
    with open(os.path.join(FEATURES_DIR, "test_metadata.pkl"), "rb") as f:
        test_meta = pickle.load(f)

    print(f"  Full dataset: train={train_emb.shape}, test={test_emb.shape}")

    # Filter ToyCar
    def filter_machine(embeddings, meta, machine):
        idx = [i for i, p in enumerate(meta["file_paths"]) if machine in p]
        sub_emb = embeddings[idx]
        sub_meta = {}
        for k, v in meta.items():
            if isinstance(v, list):
                sub_meta[k] = [v[i] for i in idx]
            else:
                sub_meta[k] = v
        return sub_emb, sub_meta

    tr_emb, tr_meta = filter_machine(train_emb, train_meta, MACHINE_TYPE)
    te_emb, te_meta = filter_machine(test_emb, test_meta, MACHINE_TYPE)

    print(f"  {MACHINE_TYPE}: train={tr_emb.shape[0]}, test={te_emb.shape[0]}")

    # Combine train + test
    all_emb = np.concatenate([tr_emb, te_emb], axis=0)

    # Build combined labels
    n_train = tr_emb.shape[0]
    n_test = te_emb.shape[0]

    anomaly_labels = np.array(
        tr_meta["anomaly_labels"] + te_meta["anomaly_labels"]
    )
    domain_labels = np.array(
        tr_meta["domain_labels"] + te_meta["domain_labels"]
    )
    # Source: train domain_labels, test domain_labels
    # Anomaly: 0=normal, 1=anomaly
    # Domain: 0=source, 1=target

    # Category masks
    src_normal = (domain_labels == 0) & (anomaly_labels == 0)
    tgt_normal = (domain_labels == 1) & (anomaly_labels == 0)
    anomaly_mask = anomaly_labels == 1

    n_src = src_normal.sum()
    n_tgt = tgt_normal.sum()
    n_anom = anomaly_mask.sum()
    print(f"  Categories: Source Normal={n_src}, Target Normal={n_tgt}, Anomaly={n_anom}")

    return all_emb, src_normal, tgt_normal, anomaly_mask


# ── UMAP 3D Projection ──────────────────────────────────────
def compute_umap_3d(embeddings):
    """Fit UMAP with n_components=3."""
    print(f"[2/4] Computing 3D UMAP projection (n_neighbors={UMAP_N_NEIGHBORS})...")
    print(f"  Input shape: {embeddings.shape}")
    print(f"  This may take 1-3 minutes for ~{embeddings.shape[0]} samples in 12288-D...")

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)

    print(f"  UMAP output shape: {coords.shape}")
    return coords


# ── 3D Scatter Plot + Animation ──────────────────────────────
def create_3d_animation(coords, src_normal, tgt_normal, anomaly_mask):
    """Create rotating 3D scatter animation and save as GIF."""
    print(f"[3/4] Building 3D animation ({N_FRAMES} frames @ {FPS} fps)...")

    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)

    # ── Static scatter plot ──────────────────────────────────
    # Source Normal (cyan, small dots)
    sc_src = ax.scatter(
        coords[src_normal, 0], coords[src_normal, 1], coords[src_normal, 2],
        c=COLOR_SRC_NORMAL, marker=MARKER_SRC, s=SIZE_NORMAL,
        alpha=0.6, depthshade=True, label="Source Normal",
    )
    # Target Normal (neon green, small triangles)
    sc_tgt = ax.scatter(
        coords[tgt_normal, 0], coords[tgt_normal, 1], coords[tgt_normal, 2],
        c=COLOR_TGT_NORMAL, marker=MARKER_TGT, s=SIZE_NORMAL,
        alpha=0.6, depthshade=True, label="Target Normal",
    )
    # Anomalies (neon red/pink, large X markers)
    sc_anom = ax.scatter(
        coords[anomaly_mask, 0], coords[anomaly_mask, 1], coords[anomaly_mask, 2],
        c=COLOR_ANOMALY, marker=MARKER_ANOMALY, s=SIZE_ANOMALY,
        alpha=0.9, depthshade=True, label="Anomaly",
        edgecolors="white", linewidths=0.5,
    )

    # ── Axis styling (dark sci-fi) ───────────────────────────
    ax.set_xlabel("UMAP-1", color=LABEL_COLOR, fontsize=10, labelpad=8)
    ax.set_ylabel("UMAP-2", color=LABEL_COLOR, fontsize=10, labelpad=8)
    ax.set_zlabel("UMAP-3", color=LABEL_COLOR, fontsize=10, labelpad=8)

    # Title with machine info
    ax.set_title(
        f"{MACHINE_TYPE} Feature Space  |  Layer {BEST_LAYER}  |  12288-D → 3-D UMAP",
        color="white", fontsize=12, fontweight="bold", pad=15,
    )

    # Dark grid styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.xaxis._axinfo["grid"]["color"] = GRID_COLOR
    ax.yaxis._axinfo["grid"]["color"] = GRID_COLOR
    ax.zaxis._axinfo["grid"]["color"] = GRID_COLOR
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.3
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.3
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.3

    # Tick styling
    ax.tick_params(colors=LABEL_COLOR, labelsize=7, pad=3)

    # Legend (positioned outside the main plot area)
    legend = ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.3,
        facecolor="#111111",
        edgecolor="#333333",
        labelcolor=LABEL_COLOR,
    )

    # Initial view
    ax.view_init(elev=20.0, azim=0)

    # ── Animation update function ────────────────────────────
    def update(frame_idx):
        azim_angle = (360.0 / N_FRAMES) * frame_idx
        ax.view_init(elev=20.0, azim=azim_angle)
        return ax

    print(f"  Rendering {N_FRAMES} frames...")
    anim = FuncAnimation(
        fig, update,
        frames=N_FRAMES,
        interval=1000 // FPS,   # ms per frame
        blit=False,
        repeat=True,
    )

    return anim, fig


# ── Save ──────────────────────────────────────────────────────
def save_gif(anim, fig):
    """Save animation as GIF using Pillow writer."""
    print(f"[4/4] Saving GIF to {OUTPUT_PATH}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    writer = PillowWriter(fps=FPS)
    anim.save(OUTPUT_PATH, writer=writer)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  Saved: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" DCASE 2025 — Sci-Fi 3D UMAP Rotation GIF Generator")
    print("=" * 60)
    print()

    # Step 1: Load data
    all_emb, src_normal, tgt_normal, anomaly_mask = load_layer2_toycar()

    # Step 2: UMAP 3D projection
    coords = compute_umap_3d(all_emb)

    # Free embeddings to save memory
    del all_emb

    # Step 3: Build animation
    anim, fig = create_3d_animation(coords, src_normal, tgt_normal, anomaly_mask)

    # Step 4: Save GIF
    save_gif(anim, fig)

    print()
    print("=" * 60)
    print(f" Done! GIF ready at: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
