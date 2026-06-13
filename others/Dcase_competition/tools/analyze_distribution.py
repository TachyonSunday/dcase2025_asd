#!/usr/bin/env python3
"""
analyze_distribution.py
============================================================
DCASE 2025 Task 2 - ToyCar Score Distribution Diagnostic
============================================================

Generates a high-quality visualization comparing Normal vs. Anomaly
score distributions for the ToyCar machine type (Source domain).

Uses the best layer features and DomainWiseDensityScorer with
local_density normalization (squared Euclidean, no L2, no MemMixup).

Output: results/ToyCar_Score_Distribution.png
"""

import os
import sys
import yaml
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# 将项目根目录加入 sys.path (tools/ 的上一级)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from utils.scoring import DomainWiseDensityScorer


def load_layer_features(features_dir, mode, layer_idx):
    """Load features for a specific layer from 3D tensor."""
    embeddings_path = os.path.join(features_dir, f"{mode}_embeddings.npy")
    metadata_path = os.path.join(features_dir, f"{mode}_metadata.pkl")

    embeddings_3d = np.load(embeddings_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # Slice the requested layer
    embeddings = embeddings_3d[layer_idx]

    return embeddings, metadata


def filter_by_machine(embeddings, metadata, machine_type):
    """Filter samples by machine type."""
    indices = [i for i, p in enumerate(metadata['file_paths'])
               if machine_type in p]
    if not indices:
        return None, None

    filtered_emb = embeddings[indices]
    filtered_meta = {}
    for key, val in metadata.items():
        if isinstance(val, list):
            filtered_meta[key] = [val[i] for i in indices]
        else:
            filtered_meta[key] = val

    return filtered_emb, filtered_meta


def main():
    print("=" * 60)
    print(" ToyCar Score Distribution Diagnostic")
    print("=" * 60)

    # ---- Load config ----
    config_path = os.path.join(_PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    features_dir = os.path.join(_PROJECT_ROOT, config['dataset']['features_dir'])
    knn_cfg = config.get('knn', {})
    best_layer = 6  # Phase 8 best layer

    # ---- Load Layer 6 features ----
    print(f"\n[Load] Layer {best_layer} features from {features_dir}")
    train_emb, train_meta = load_layer_features(features_dir, 'train', best_layer)
    test_emb, test_meta = load_layer_features(features_dir, 'test', best_layer)
    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

    # ---- Filter ToyCar ----
    print("[Filter] ToyCar samples")
    train_tc, train_tc_meta = filter_by_machine(train_emb, train_meta, 'ToyCar')
    test_tc, test_tc_meta = filter_by_machine(test_emb, test_meta, 'ToyCar')

    if train_tc is None or test_tc is None:
        print("[Error] No ToyCar samples found!")
        return

    print(f"  Train ToyCar: {train_tc.shape[0]} samples")
    print(f"  Test ToyCar:  {test_tc.shape[0]} samples")

    train_domain_labels = np.array(train_tc_meta['domain_labels'])
    test_anomaly_labels = np.array(test_tc_meta['anomaly_labels'])
    test_domain_labels = np.array(test_tc_meta['domain_labels'])

    # ---- Build scorer ----
    print(f"\n[Scorer] DomainWiseDensityScorer (local_density, squared euclidean)")
    scorer = DomainWiseDensityScorer(
        k_source=knn_cfg.get('k_source', 16),
        k_target=knn_cfg.get('k_target', 9),
        k_score=knn_cfg.get('k_score', 5),
        metric=knn_cfg.get('metric', 'euclidean'),
        n_mix_support=knn_cfg.get('n_mix_support', None),
        alpha=knn_cfg.get('alpha', 0.9),
        n_jobs=knn_cfg.get('n_jobs', -1),
        score_normalization=knn_cfg.get('score_normalization', 'local_density'),
    )

    scorer.fit(train_tc, train_domain_labels)
    anomaly_scores, scores_source, scores_target = scorer.score_with_details(test_tc)

    # ---- Filter Source domain test samples ----
    src_mask = test_domain_labels == 0
    src_anomaly = test_anomaly_labels[src_mask]
    src_scores = anomaly_scores[src_mask]

    n_normal = (src_anomaly == 0).sum()
    n_anomaly = (src_anomaly == 1).sum()
    print(f"\n[Source Domain] Normal: {n_normal}, Anomaly: {n_anomaly}")
    print(f"  Normal scores:  mean={src_scores[src_anomaly==0].mean():.4f}, "
          f"std={src_scores[src_anomaly==0].std():.4f}")
    print(f"  Anomaly scores: mean={src_scores[src_anomaly==1].mean():.4f}, "
          f"std={src_scores[src_anomaly==1].std():.4f}")

    # ---- Generate visualization ----
    print("\n[Plot] Generating visualization...")
    results_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle(f'ToyCar Score Distribution (Layer {best_layer}, Source Domain)\n'
                 f'DomainWiseDensityScorer — Squared Euclidean, Local Density',
                 fontsize=14, fontweight='bold')

    # ---- Left: Boxplot ----
    ax1 = axes[0]
    normal_scores = src_scores[src_anomaly == 0]
    anomaly_only = src_scores[src_anomaly == 1]

    bp = ax1.boxplot(
        [normal_scores, anomaly_only],
        labels=['Normal', 'Anomaly'],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='.', markersize=4, alpha=0.5),
    )
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#F44336')
    bp['boxes'][1].set_alpha(0.7)
    bp['medians'][0].set_color('white')
    bp['medians'][1].set_color('white')

    ax1.set_ylabel('Anomaly Score (density-normalized)', fontsize=11)
    ax1.set_title('Score Boxplot', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # ---- Right: KDE Distribution ----
    ax2 = axes[1]
    sns.kdeplot(normal_scores, ax=ax2, label=f'Normal (n={n_normal})',
                color='#2196F3', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(anomaly_only, ax=ax2, label=f'Anomaly (n={n_anomaly})',
                color='#F44336', fill=True, alpha=0.3, linewidth=2)

    # Add vertical lines for means
    ax2.axvline(normal_scores.mean(), color='#1565C0', linestyle='--',
                linewidth=1.5, alpha=0.8, label=f'Normal mean={normal_scores.mean():.3f}')
    ax2.axvline(anomaly_only.mean(), color='#C62828', linestyle='--',
                linewidth=1.5, alpha=0.8, label=f'Anomaly mean={anomaly_only.mean():.3f}')

    ax2.set_xlabel('Anomaly Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Score KDE Distribution', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='both', alpha=0.3)

    # ---- Compute AUC for annotation ----
    from sklearn.metrics import roc_auc_score
    if len(np.unique(src_anomaly)) > 1:
        auc = roc_auc_score(src_anomaly, src_scores)
        fig.text(0.5, 0.01,
                 f'AUC (Source): {auc:.4f}  |  '
                 f'Normal mean: {normal_scores.mean():.4f}  |  '
                 f'Anomaly mean: {anomaly_only.mean():.4f}  |  '
                 f'Separation: {anomaly_only.mean() - normal_scores.mean():.4f}',
                 ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    output_path = os.path.join(results_dir, 'ToyCar_Score_Distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Save] Plot saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
