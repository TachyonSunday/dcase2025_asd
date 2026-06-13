import os
import glob
import numpy as np
import librosa
import torch
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import csv
import pickle
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

# 屏蔽常规警告
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = r"./"
DEV_DIR = os.path.join(BASE_DIR, "Development Dataset")
EVAL_DIR = os.path.join(BASE_DIR, "Evaluation Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "Submissions")
MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models")

for d in [OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Audio & Feature processing params
SR, N_FFT, HOP_LENGTH, N_MELS = 16000, 1024, 512, 128
PATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model Setup
# ---------------------------------------------------------
print("[Info] Loading pre-trained ResNet18...")
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # 移除分类层，仅提取 512 维特征
resnet.eval()
resnet.to(DEVICE)

def extract_resnet_features(file_path):
    """提取音频的 Log-Mel 频谱并输入 ResNet 获取深度特征"""
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)  # [128, T]
    
    # 归一化到 [0, 1]
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val > min_val:
        log_mel = (log_mel - min_val) / (max_val - min_val)
        
    T = log_mel.shape[1]
    if T < PATCH_SIZE:
        log_mel = np.pad(log_mel, ((0, 0), (0, PATCH_SIZE - T)), mode='constant')
        T = PATCH_SIZE

    # 滑动窗口切片 (50% overlap)
    step = PATCH_SIZE // 2
    patches = [log_mel[:, i : i + PATCH_SIZE] for i in range(0, T - PATCH_SIZE + 1, step)]
    patches_np = np.array(patches)  # [N, 128, 128]
    
    # 扩展为 3 通道以适配 ResNet
    patches_t = torch.tensor(patches_np, dtype=torch.float32).unsqueeze(1)
    patches_t = patches_t.repeat(1, 3, 1, 1).to(DEVICE)  # [N, 3, 128, 128]
    
    with torch.no_grad():
        features = resnet(patches_t)  # [N, 512]
        
    return features.cpu().numpy()

# ---------------------------------------------------------
# Anomaly Detection Pipeline
# ---------------------------------------------------------
def train_knn_detector(machine_name, train_files):
    X_train = []
    for f in tqdm(train_files, desc=f"[{machine_name}] Training", leave=False):
        X_train.append(extract_resnet_features(f))
        
    X_train = np.vstack(X_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled)
    
    # 保存权重
    with open(os.path.join(MODEL_DIR, f"{machine_name}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"{machine_name}_knn.pkl"), 'wb') as f:
        pickle.dump(knn, f)
        
    return scaler, knn

def evaluate_detector(test_files, scaler, knn):
    y_true, y_scores, file_scores = [], [], {}
    
    for f in tqdm(test_files, desc="Evaluating", leave=False):
        feats = extract_resnet_features(f)
        feats_scaled = scaler.transform(feats)
        
        distances, _ = knn.kneighbors(feats_scaled)
        patch_scores = distances.mean(axis=1)
        
        # Max pooling: 任何切片异常即代表整体异常
        final_score = float(np.max(patch_scores))
        file_scores[f] = final_score
        
        filename = os.path.basename(f).lower()
        if "normal" in filename or "anomaly" in filename:
            y_true.append(0 if "normal" in filename else 1)
            y_scores.append(final_score)
            
    auc, p_auc = 0.0, 0.0
    if len(set(y_true)) > 1:
        auc = metrics.roc_auc_score(y_true, y_scores)
        p_auc = metrics.roc_auc_score(y_true, y_scores, max_fpr=0.1)
        
    return auc, p_auc, file_scores

# ---------------------------------------------------------
# Visualization Helper
# ---------------------------------------------------------
def plot_results(results_dict):
    """绘制所有机器的 AUC/pAUC 性能对比图"""
    if not results_dict:
        return
        
    # 按 AUC 排序，提升图表可读性
    sorted_items = sorted(results_dict.items(), key=lambda x: x[1]['auc'], reverse=True)
    machines = [item[0] for item in sorted_items]
    aucs = [item[1]['auc'] for item in sorted_items]
    paucs = [item[1]['p_auc'] for item in sorted_items]

    x = np.arange(len(machines))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, aucs, width, label='AUC', color='#4C72B0')
    plt.bar(x + width/2, paucs, width, label='pAUC', color='#DD8452')

    plt.ylabel('Score')
    plt.title('Anomaly Detection Performance by Machine Type')
    plt.xticks(x, machines, rotation=15)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 自动保存图表
    plot_path = os.path.join(OUTPUT_DIR, "performance_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[Info] Performance plot saved to {plot_path}")
    plt.show()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- DCASE Audio Anomaly Detection Pipeline ---")
    
    if os.path.exists(DEV_DIR):
        machine_types = [d for d in os.listdir(DEV_DIR) if os.path.isdir(os.path.join(DEV_DIR, d))]
    else:
        machine_types = []
        print(f"[Warning] Development Dataset directory not found: {DEV_DIR}")

    trained_assets = {}
    performance_metrics = {}
    
    # 1. 训练与验证阶段
    for machine in machine_types:
        train_files = glob.glob(os.path.join(DEV_DIR, machine, "train", "*.wav"))
        test_files = glob.glob(os.path.join(DEV_DIR, machine, "test", "*.wav"))
        
        if not train_files or not test_files:
            continue
            
        scaler, knn = train_knn_detector(machine, train_files)
        auc, p_auc, _ = evaluate_detector(test_files, scaler, knn)
        
        print(f"{machine.ljust(10)} | AUC: {auc:.4f} | pAUC: {p_auc:.4f}")
        
        trained_assets[machine] = (scaler, knn)
        performance_metrics[machine] = {'auc': auc, 'p_auc': p_auc}

    # 绘制验证集性能可视化
    plot_results(performance_metrics)

    # 2. 盲测集预测与导出阶段
    if os.path.exists(EVAL_DIR):
        print("\n[Info] Starting evaluation on blind test set...")
        eval_machines = [d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d))]
        
        for machine in eval_machines:
            eval_files = glob.glob(os.path.join(EVAL_DIR, machine, "test", "*.wav"))
            if not eval_files or machine not in trained_assets:
                continue
                
            scaler, knn = trained_assets[machine]
            _, _, file_scores = evaluate_detector(eval_files, scaler, knn)
            
            csv_file = os.path.join(OUTPUT_DIR, f"anomaly_score_{machine}.csv")
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                for filepath, score in file_scores.items():
                    writer.writerow([os.path.basename(filepath), score])
                    
        print("[Info] CSV generation completed.")
    else:
        print("[Info] No Evaluation Dataset found. Skipping test generation.")