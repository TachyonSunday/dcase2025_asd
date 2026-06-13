#!/usr/bin/env python3
"""
tools/generate_sci_plots.py
============================================================
DCASE 2025 Task 2 - SCI-Level Visualization Suite
============================================================

Generates 4 publication-quality figures for the final report:
  1. UMAP Feature Space (Representation Proof)
  2. KDE Score Distribution (Discriminability Proof)
  3. Hyperparameter Sensitivity Heatmap (Robustness Proof)
  4. ROC & PR Curves (Performance Envelope)

All plots use Layer 2 features (12288-D, Hybrid Mean+Max Pooling).

Usage:
    python tools/generate_sci_plots.py

Output:
    ./results/figures/fig1_umap_toycar.png
    ./results/figures/fig2_kde_toycar.png
    ./results/figures/fig3_sensitivity_heatmap.png
    ./results/figures/fig4_roc_pr_curves.png
"""

import os
import sys
import pickle
import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import umap

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from utils.scoring import DomainWiseDensityScorer, compute_auc, compute_pauc


# ============================================================
# Global styling
# ============================================================
sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")
BEST_LAYER = 2
MACHINE_TYPES = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]


# ============================================================
# Data loading utilities
# ============================================================
def load_data(features_dir=None, layer_idx=BEST_LAYER):
    """Load Layer 2 features and metadata."""
    if features_dir is None:
        features_dir = os.path.join(_PROJECT_ROOT, "features")
    train_emb = np.load(os.path.join(features_dir, "train_embeddings.npy"))[layer_idx]
    test_emb = np.load(os.path.join(features_dir, "test_embeddings.npy"))[layer_idx]

    with open(os.path.join(features_dir, "train_metadata.pkl"), "rb") as f:
        train_meta = pickle.load(f)
    with open(os.path.join(features_dir, "test_metadata.pkl"), "rb") as f:
        test_meta = pickle.load(f)

    return train_emb, test_emb, train_meta, test_meta


def filter_machine(embeddings, meta, machine_type):
    """Filter by machine type."""
    idx = [i for i, p in enumerate(meta["file_paths"]) if machine_type in p]
    sub_emb = embeddings[idx]
    sub_meta = {}
    for k, v in meta.items():
        if isinstance(v, list):
            sub_meta[k] = [v[i] for i in idx]
        else:
            sub_meta[k] = v
    return sub_emb, sub_meta


def build_scorer(config):
    """Build DomainWiseDensityScorer from config."""
    knn = config.get("knn", {})
    return DomainWiseDensityScorer(
        k_source=knn.get("k_source", 16),
        k_target=knn.get("k_target", 9),
        k_score=knn.get("k_score", 5),
        metric=knn.get("metric", "euclidean"),
        n_mix_support=knn.get("n_mix_support", None),
        alpha=knn.get("alpha", 0.9),
        n_jobs=knn.get("n_jobs", -1),
        score_normalization=knn.get("score_normalization", "local_density"),
    )


def harmonic_mean(values):
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


# ============================================================
# Plot 1: UMAP Feature Space
# ============================================================
def plot_umap(test_emb, test_meta, machine="ToyCar"):
    """UMAP projection of Layer 2 test features."""
    print("[Plot 1] UMAP Feature Space...")

    emb, meta = filter_machine(test_emb, test_meta, machine)
    anomaly = np.array(meta["anomaly_labels"])
    domain = np.array(meta["domain_labels"])

    # Categories
    src_normal = (domain == 0) & (anomaly == 0)
    tgt_normal = (domain == 1) & (anomaly == 0)
    anomaly_mask = anomaly == 1

    # UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(coords[src_normal, 0], coords[src_normal, 1],
               c="#2196F3", marker="o", s=40, alpha=0.7, label="Source Normal",
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(coords[tgt_normal, 0], coords[tgt_normal, 1],
               c="#4CAF50", marker="^", s=50, alpha=0.7, label="Target Normal",
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(coords[anomaly_mask, 0], coords[anomaly_mask, 1],
               c="#F44336", marker="x", s=50, alpha=0.8, label="Anomaly",
               linewidths=2, zorder=4)

    ax.set_xlabel("UMAP-1", fontsize=13)
    ax.set_ylabel("UMAP-2", fontsize=13)
    ax.set_title(f"{machine} Feature Space (Layer {BEST_LAYER}, UMAP)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=11)
    sns.despine()

    path = os.path.join(OUTPUT_DIR, "fig1_umap_toycar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 2: KDE Score Distribution
# ============================================================
def plot_kde(train_emb, test_emb, train_meta, test_meta, config, machine="ToyCar"):
    """KDE of anomaly scores: Normal vs Anomaly."""
    print("[Plot 2] KDE Score Distribution...")

    tr_emb, tr_meta = filter_machine(train_emb, train_meta, machine)
    te_emb, te_meta = filter_machine(test_emb, test_meta, machine)

    tr_domain = np.array(tr_meta["domain_labels"])
    te_anomaly = np.array(te_meta["anomaly_labels"])

    scorer = build_scorer(config)
    scorer.fit(tr_emb, tr_domain)
    scores, _, _ = scorer.score_with_details(te_emb)

    normal_scores = scores[te_anomaly == 0]
    anomaly_scores = scores[te_anomaly == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.kdeplot(normal_scores, ax=ax, label=f"Normal (n={len(normal_scores)})",
                color="#2196F3", fill=True, alpha=0.3, linewidth=2.5)
    sns.kdeplot(anomaly_scores, ax=ax, label=f"Anomaly (n={len(anomaly_scores)})",
                color="#F44336", fill=True, alpha=0.3, linewidth=2.5)

    # Find intersection threshold (approximate via KDE evaluation)
    from scipy.stats import gaussian_kde
    kde_n = gaussian_kde(normal_scores)
    kde_a = gaussian_kde(anomaly_scores)
    x_grid = np.linspace(scores.min(), scores.max(), 500)
    diff = kde_n(x_grid) - kde_a(x_grid)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        threshold = x_grid[sign_changes[0]]
        ax.axvline(threshold, color="#FF9800", linestyle="--", linewidth=2,
                   label=f"Threshold ≈ {threshold:.4f}")

    # AUC annotation
    test_auc = compute_auc(te_anomaly, scores)
    ax.annotate(f"AUC = {test_auc:.3f}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                fontsize=13, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Anomaly Score (density-normalized)", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title(f"{machine} Score Distribution (Layer {BEST_LAYER})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    sns.despine()

    path = os.path.join(OUTPUT_DIR, "fig2_kde_toycar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 3: Hyperparameter Sensitivity Heatmap
# ============================================================
def plot_sensitivity(train_emb, test_emb, train_meta, test_meta, config):
    """Heatmap of Omega across K_s x K_t grid (Layer 2, all machines)."""
    print("[Plot 3] Hyperparameter Sensitivity Heatmap...")

    ks_values = [10, 16, 20, 26]
    kt_values = [3, 6, 9, 12]
    grid = np.zeros((len(ks_values), len(kt_values)))

    for i, ks in enumerate(ks_values):
        for j, kt in enumerate(kt_values):
            omegas = []
            for machine in MACHINE_TYPES:
                tr_emb, tr_meta = filter_machine(train_emb, train_meta, machine)
                te_emb, te_meta = filter_machine(test_emb, test_meta, machine)

                if len(tr_emb) == 0 or len(te_emb) == 0:
                    continue

                tr_domain = np.array(tr_meta["domain_labels"])
                te_anomaly = np.array(te_meta["anomaly_labels"])
                te_domain = np.array(te_meta["domain_labels"])

                scorer = DomainWiseDensityScorer(
                    k_source=ks, k_target=kt, k_score=5,
                    metric="euclidean",
                    n_mix_support=None, alpha=0.9,
                    n_jobs=-1,
                    score_normalization="local_density",
                )
                scorer.fit(tr_emb, tr_domain)
                scores, _, _ = scorer.score_with_details(te_emb)

                # Compute Omega
                src_mask = te_domain == 0
                tgt_mask = te_domain == 1
                auc_s = compute_auc(te_anomaly[src_mask], scores[src_mask]) if len(np.unique(te_anomaly[src_mask])) > 1 else 0
                auc_t = compute_auc(te_anomaly[tgt_mask], scores[tgt_mask]) if len(np.unique(te_anomaly[tgt_mask])) > 1 else 0
                pauc = compute_pauc(te_anomaly, scores, max_fpr=0.1) if len(np.unique(te_anomaly)) > 1 else 0
                omega = harmonic_mean([auc_s, auc_t, pauc])
                omegas.append(omega)

            grid[i, j] = np.mean(omegas) if omegas else 0
            print(f"  K_s={ks:2d}, K_t={kt:2d} -> avg Omega = {grid[i, j]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        grid,
        xticklabels=[str(k) for k in kt_values],
        yticklabels=[str(k) for k in ks_values],
        annot=True, fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Avg Official Score (Omega)"},
        linewidths=1, linecolor="white",
        vmin=grid.min() - 0.005, vmax=grid.max() + 0.005,
        ax=ax,
    )
    ax.set_xlabel("$K_t$ (Target Neighbors)", fontsize=13)
    ax.set_ylabel("$K_s$ (Source Neighbors)", fontsize=13)
    ax.set_title(f"Hyperparameter Sensitivity (Layer {BEST_LAYER})", fontsize=14, fontweight="bold")

    # Highlight best cell
    best_i, best_j = np.unravel_index(np.argmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((best_j, best_i), 1, 1,
                                fill=False, edgecolor="#F44336", linewidth=3))

    path = os.path.join(OUTPUT_DIR, "fig3_sensitivity_heatmap.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 4: ROC & PR Curves
# ============================================================
def plot_roc_pr(test_emb, test_meta, train_emb, train_meta, config,
                machines=("ToyCar", "ToyTrain")):
    """ROC and PR curves for two machines."""
    print("[Plot 4] ROC & PR Curves...")

    colors = {"ToyCar": "#2196F3", "ToyTrain": "#FF9800"}
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))

    for machine in machines:
        tr_emb, tr_meta = filter_machine(train_emb, train_meta, machine)
        te_emb, te_meta = filter_machine(test_emb, test_meta, machine)

        tr_domain = np.array(tr_meta["domain_labels"])
        te_anomaly = np.array(te_meta["anomaly_labels"])

        scorer = build_scorer(config)
        scorer.fit(tr_emb, tr_domain)
        scores, _, _ = scorer.score_with_details(te_emb)

        # ROC
        fpr, tpr, _ = roc_curve(te_anomaly, scores)
        roc_auc = auc(fpr, tpr)
        p_auc = compute_pauc(te_anomaly, scores, max_fpr=0.1)

        ax_roc.plot(fpr, tpr, color=colors[machine], linewidth=2.5,
                    label=f"{machine} (AUC={roc_auc:.3f}, pAUC={p_auc:.3f})")

        # PR
        precision, recall, _ = precision_recall_curve(te_anomaly, scores)
        ap = average_precision_score(te_anomaly, scores)

        ax_pr.plot(recall, precision, color=colors[machine], linewidth=2.5,
                   label=f"{machine} (AP={ap:.3f})")

    # ROC styling
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate", fontsize=13)
    ax_roc.set_ylabel("True Positive Rate", fontsize=13)
    ax_roc.set_title(f"ROC Curve (Layer {BEST_LAYER})", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)

    # PR styling
    ax_pr.set_xlabel("Recall", fontsize=13)
    ax_pr.set_ylabel("Precision", fontsize=13)
    ax_pr.set_title(f"Precision-Recall Curve (Layer {BEST_LAYER})", fontsize=14, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)

    sns.despine()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig4_roc_pr_curves.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print(" DCASE 2025 SCI Visualization Suite")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    config_path = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load data
    print(f"\n[Data] Loading Layer {BEST_LAYER} features...")
    train_emb, test_emb, train_meta, test_meta = load_data(layer_idx=BEST_LAYER)
    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

    # Generate all plots
    print()
    plot_umap(test_emb, test_meta, machine="ToyCar")
    print()
    plot_kde(train_emb, test_emb, train_meta, test_meta, config, machine="ToyCar")
    print()
    plot_sensitivity(train_emb, test_emb, train_meta, test_meta, config)
    print()
    plot_roc_pr(test_emb, test_meta, train_emb, train_meta, config)

    print(f"\n{'=' * 60}")
    print(f" All 4 figures saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
