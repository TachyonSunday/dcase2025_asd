#!/usr/bin/env python3
"""
tools/plot_ablations.py
============================================================
DCASE 2025 Task 2 - Ablation Study Visualizations
============================================================

Generates 3 ablation study charts for the final presentation:
  1. Feature Engine Ablation (SSL vs Fine-Tuned, line chart)
  2. Pooling Strategy Ablation (Mean vs Mean+Max, grouped bar)
  3. Architecture Evolution (Waterfall chart)

Usage:
    python tools/plot_ablations.py

Output:
    ./results/figures/fig5_ablation_engine.png
    ./results/figures/fig6_ablation_pooling.png
    ./results/figures/fig7_ablation_waterfall.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── 项目根目录 (tools/ 的上一级) ─────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Global styling
# ============================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")


# ============================================================
# Plot 1: Feature Engine Ablation (Line Chart)
# ============================================================
def plot_feature_engine_ablation():
    """Compare SSL vs Fine-Tuned checkpoint across layers."""
    print("[Plot 5] Feature Engine Ablation...")

    layers = list(range(11))
    ssl_scores = [0.5720, 0.5651, 0.5635, 0.5633, 0.5624,
                  0.5642, 0.5593, 0.5576, 0.5471, 0.5418, 0.5409]
    ft_scores  = [0.5737, 0.5739, 0.5730, 0.5692, 0.5695,
                  0.5741, 0.5758, 0.5727, 0.5653, 0.5503, 0.5407]

    fig, ax = plt.subplots(figsize=(10, 6))

    # SSL line
    ax.plot(layers, ssl_scores, 'o-', color="#78909C", linewidth=2.5,
            markersize=8, label="SSL Checkpoint (BEATs_iter3_plus_AS2M)",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    # Fine-tuned line
    ax.plot(layers, ft_scores, 's-', color="#1565C0", linewidth=2.5,
            markersize=8, label="Fine-Tuned Checkpoint (beats_ft1)",
            markeredgecolor="white", markeredgewidth=1.5, zorder=4)

    # Highlight Layer 6 peak on fine-tuned
    ax.annotate(f"Peak: {ft_scores[6]:.4f}",
                xy=(6, ft_scores[6]), xytext=(7.5, ft_scores[6] + 0.006),
                fontsize=12, fontweight="bold", color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.9))

    # Highlight SSL Layer 0 peak
    ax.annotate(f"SSL Best: {ssl_scores[0]:.4f}",
                xy=(0, ssl_scores[0]), xytext=(2.5, ssl_scores[0] + 0.008),
                fontsize=11, color="#78909C",
                arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", alpha=0.9))

    # Shallow-layer collapse annotation
    ax.axvspan(-0.5, 1.5, alpha=0.08, color="#F44336", zorder=0)
    ax.text(0.5, min(ssl_scores) - 0.003, "Shallow\nCollapse",
            ha="center", fontsize=10, color="#F44336", style="italic", alpha=0.7)

    ax.set_xlabel("Transformer Layer Index", fontsize=13)
    ax.set_ylabel("Official Score (Omega)", fontsize=13)
    ax.set_title("Feature Engine Ablation: SSL vs Fine-Tuned Checkpoint",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    ax.set_ylim(0.535, 0.585)

    sns.despine()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig5_ablation_engine.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 2: Pooling Strategy Ablation (Grouped Bar Chart)
# ============================================================
def plot_pooling_ablation():
    """Compare Mean vs Mean+Max pooling per machine type."""
    print("[Plot 6] Pooling Strategy Ablation...")

    machines = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    phase8_mean = [0.6561, 0.5577, 0.6196, 0.5645, 0.5212, 0.5096, 0.6018]
    phase9_hybrid = [0.6549, 0.5532, 0.6160, 0.5576, 0.5986, 0.5926, 0.5922]

    x = np.arange(len(machines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, phase8_mean, width,
                   color="#78909C", label="Phase 8: Mean Pool (6144-D)",
                   edgecolor="white", linewidth=1.2, zorder=3)
    bars2 = ax.bar(x + width/2, phase9_hybrid, width,
                   color="#1565C0", label="Phase 9: Mean+Max Pool (12288-D)",
                   edgecolor="white", linewidth=1.2, zorder=3)

    # Annotate ToyCar (+7.7%) and ToyTrain (+8.3%)
    highlight_indices = {4: "+7.7%", 5: "+8.3%"}
    for idx, label in highlight_indices.items():
        delta = phase9_hybrid[idx] - phase8_mean[idx]
        y_pos = max(phase8_mean[idx], phase9_hybrid[idx]) + 0.012
        ax.annotate(label,
                    xy=(x[idx] + width/2, phase9_hybrid[idx]),
                    xytext=(x[idx], y_pos),
                    fontsize=13, fontweight="bold", color="#C62828",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#C62828", lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.95))

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, color="#546E7A")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, color="#0D47A1")

    ax.set_xlabel("Machine Type", fontsize=13)
    ax.set_ylabel("Official Score (Omega)", fontsize=13)
    ax.set_title("Pooling Strategy Ablation: Mean vs Hybrid Mean+Max",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(machines, fontsize=11)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.set_ylim(0.45, 0.72)

    # Reference line at 0.60
    ax.axhline(y=0.60, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(len(machines) - 0.5, 0.605, "0.60 target", fontsize=9,
            color="#FF9800", ha="right", alpha=0.7)

    sns.despine()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig6_ablation_pooling.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 3: Architecture Evolution (Waterfall Chart)
# ============================================================
def plot_waterfall():
    """Waterfall chart showing cumulative score evolution."""
    print("[Plot 7] Architecture Evolution Waterfall...")

    milestones = ['Baseline\n(Phase 4)',
                  'Remove\nMemMixup',
                  'Sq-Euclidean\n+ SUM',
                  'Fine-Tuned\nCheckpoint',
                  'Hybrid\nPooling']
    bases  = [0.0,    0.5568, 0.5585, 0.5720, 0.5758]
    steps  = [0.5568, 0.0017, 0.0135, 0.0038, 0.0192]
    cumulative = [b + s for b, s in zip(bases, steps)]  # running totals

    # Colors: first bar is blue (base), rest are green (improvements)
    colors = ["#1565C0"] + ["#2E7D32"] * (len(milestones) - 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(milestones))
    bar_width = 0.55

    # Draw bars
    for i in range(len(milestones)):
        if i == 0:
            # Base bar starts from 0
            bar = ax.bar(x[i], steps[i], bar_width, bottom=0,
                         color=colors[i], edgecolor="white", linewidth=1.5, zorder=3)
        else:
            # Step bars float on top of previous cumulative
            bar = ax.bar(x[i], steps[i], bar_width, bottom=bases[i],
                         color=colors[i], edgecolor="white", linewidth=1.5, zorder=3)

            # Connector line from previous bar
            ax.plot([x[i-1] + bar_width/2, x[i] - bar_width/2],
                    [bases[i], bases[i]],
                    color="#90A4AE", linewidth=1.5, linestyle=":", zorder=2)

        # Value annotation on bar
        if i == 0:
            ax.text(x[i], steps[i]/2, f"{steps[i]:.4f}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        else:
            ax.text(x[i], bases[i] + steps[i]/2, f"+{steps[i]:.4f}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # Final cumulative annotation
    final_score = cumulative[-1]
    ax.annotate(f"Final: {final_score:.4f}",
                xy=(x[-1], cumulative[-1]),
                xytext=(x[-1] + 0.8, cumulative[-1] + 0.005),
                fontsize=14, fontweight="bold", color="#C62828",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=2.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                          edgecolor="#C62828", alpha=0.95))

    # 0.60 target line
    ax.axhline(y=0.60, color="#FF9800", linestyle="--", linewidth=2, alpha=0.5)
    ax.text(-0.3, 0.602, "0.60 target", fontsize=10, color="#FF9800", alpha=0.7)

    ax.set_xlabel("Optimization Phase", fontsize=13)
    ax.set_ylabel("Official Score (Omega)", fontsize=13)
    ax.set_title("Architecture Evolution: Cumulative Score Waterfall",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(milestones, fontsize=10)
    ax.set_ylim(0, 0.65)
    ax.set_xlim(-0.6, len(milestones) - 0.4 + 1.2)

    sns.despine()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "fig7_ablation_waterfall.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print(" DCASE 2025 Ablation Study Visualizations")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    plot_feature_engine_ablation()
    print()
    plot_pooling_ablation()
    print()
    plot_waterfall()

    print(f"\n{'=' * 60}")
    print(f" All 3 ablation figures saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
