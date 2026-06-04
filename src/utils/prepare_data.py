"""
DCASE 2025 Task 2 数据准备脚本 —— 解压所有机器类型, 提取特征, 组织训练/测试集。
"""

import os
import sys
import glob
import csv
import zipfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from tqdm import tqdm

# 加入项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.pipeline import FeaturePipeline


def parse_dcase_filename(filename: str) -> Dict[str, str]:
    """
    从 DCASE 文件名中解析元信息。

    示例::

        section_00_source_train_normal_0001_car_B1_spd_31V_mic_1.wav
        → {"section": "00", "subset": "source", "split": "train",
           "label": "normal", "index": "0001", "car": "B1",
           "speed": "31V", "mic": "1"}
    """
    base = filename.replace(".wav", "")
    parts = base.split("_")
    info = {
        "section": parts[1],
        "subset": parts[2],
        "split": parts[3],
        "label": parts[4],
        "index": parts[5],
    }
    # 解析 car_XX, spd_XX, mic_X
    for p in parts[6:]:
        if p in ("car", "spd", "mic", "speed"):
            continue
        if p.endswith("V"):
            info["speed"] = p
        elif p.isdigit():
            info["mic"] = p
        else:
            # car model (e.g., A1, B2, E1)
            if "car" not in info:
                info["car"] = p
    return info


def load_attributes(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    加载 DCASE attributes CSV 文件。

    返回
    ----
    dict[str, dict]
        {filename: {d1p: "car", d1v: "B1", d2p: "speed", d2v: "31V", ...}}
    """
    attrs = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file_name"]
            attrs[fname] = {
                "car": row.get("d1v", ""),
                "speed": row.get("d2v", ""),
                "mic": row.get("d3v", ""),
            }
    return attrs


def get_domain_id(attrs: Dict[str, str], num_speeds: int = 5) -> int:
    """
    根据 (car, speed) 组合计算域 ID, 用于 DANN 训练。

    每个唯一的 car-speed 组合被视为一个独立的域。
    """
    car_models = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2"]
    speeds = ["28V", "31V", "34V", "37V", "40V"]

    car = attrs.get("car", "unknown")
    speed = attrs.get("speed", "unknown")

    car_idx = car_models.index(car) if car in car_models else 0
    speed_idx = speeds.index(speed) if speed in speeds else 0

    return car_idx * len(speeds) + speed_idx


def prepare_machine_type(
    machine_type: str,
    zip_path: str,
    output_base: str,
    pipeline: FeaturePipeline,
    max_train_files: Optional[int] = None,
) -> Dict[str, int]:
    """
    预处理单个机器类型: 解压 → 特征提取 → 组织到训练/测试目录。

    参数
    ----
    machine_type : str
        机器类型名称 (如 "ToyCar")。
    zip_path : str
        ZIP 文件路径。
    output_base : str
        处理后数据输出根目录。
    pipeline : FeaturePipeline
        特征提取流水线实例。
    max_train_files : int, 可选
        限制训练文件数量 (用于快速测试)。

    返回
    ----
    dict[str, int]
        {"train": 训练文件数, "test_normal": 测试正常数, "test_anomaly": 测试异常数}
    """
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw", machine_type)
    os.makedirs(raw_dir, exist_ok=True)

    # 解压
    if not os.path.exists(os.path.join(raw_dir, machine_type)):
        print(f"  解压 {machine_type}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    inner_dir = os.path.join(raw_dir, machine_type)
    train_dir = os.path.join(inner_dir, "train")
    test_dir = os.path.join(inner_dir, "test")
    csv_path = os.path.join(inner_dir, "attributes_00.csv")

    # 加载属性表
    attrs = load_attributes(csv_path) if os.path.exists(csv_path) else {}

    stats = {"train": 0, "test_normal": 0, "test_anomaly": 0}

    # ---- 处理训练集 (仅正常样本) ----
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    if max_train_files:
        train_files = train_files[:max_train_files]

    train_out = os.path.join(output_base, f"{machine_type}/train/")
    os.makedirs(train_out, exist_ok=True)

    for wav_path in tqdm(train_files, desc=f"{machine_type} train"):
        fname = os.path.basename(wav_path)
        out_path = os.path.join(train_out, os.path.splitext(fname)[0] + ".pt")
        try:
            pipeline.process_file(wav_path, out_path)
            stats["train"] += 1
        except Exception as e:
            print(f"  跳过 {fname}: {e}")

    # ---- 处理测试集 (正常 + 异常) ----
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    test_out = os.path.join(output_base, f"{machine_type}/test/")
    os.makedirs(test_out, exist_ok=True)

    for wav_path in tqdm(test_files, desc=f"{machine_type} test"):
        fname = os.path.basename(wav_path)
        info = parse_dcase_filename(fname)
        out_path = os.path.join(test_out, os.path.splitext(fname)[0] + ".pt")
        try:
            pipeline.process_file(wav_path, out_path)
            if info["label"] == "normal":
                stats["test_normal"] += 1
            else:
                stats["test_anomaly"] += 1
        except Exception as e:
            print(f"  跳过 {fname}: {e}")

    return stats


def main():
    """主入口: 预处理所有机器类型。"""
    import argparse
    parser = argparse.ArgumentParser(description="DCASE 2025 T2 数据准备")
    parser.add_argument("--machine", type=str, default=None,
                       help="仅处理指定机器类型 (ToyCar, fan, gearbox, ...)")
    parser.add_argument("--max-train", type=int, default=None,
                       help="限制训练文件数量 (快速测试)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    zip_dir = "data/raw/dcase2025_dev"
    processed_base = "data/processed/dcase2025"

    pipeline = FeaturePipeline(args.config)

    machine_types = [
        "ToyCar", "ToyTrain", "bearing", "fan",
        "gearbox", "slider", "valve",
    ]

    if args.machine:
        machine_types = [args.machine]

    all_stats = {}
    for mt in machine_types:
        zip_path = os.path.join(zip_dir, f"dev_{mt}.zip")
        if not os.path.exists(zip_path):
            print(f"⚠️ 跳过 {mt}: 未找到 {zip_path}")
            continue
        print(f"\n{'='*50}")
        print(f"处理: {mt}")
        print(f"{'='*50}")
        stats = prepare_machine_type(mt, zip_path, processed_base, pipeline,
                                     max_train_files=args.max_train)
        all_stats[mt] = stats

    # 汇总
    print(f"\n{'='*50}")
    print("预处理完成!")
    print(f"{'='*50}")
    total_train = sum(s["train"] for s in all_stats.values())
    total_tn = sum(s["test_normal"] for s in all_stats.values())
    total_ta = sum(s["test_anomaly"] for s in all_stats.values())
    for mt, s in all_stats.items():
        print(f"  {mt:12s}: train={s['train']:4d}  test_normal={s['test_normal']:3d}  test_anomaly={s['test_anomaly']:3d}")
    print(f"  {'总计':12s}: train={total_train:4d}  test_normal={total_tn:3d}  test_anomaly={total_ta:3d}")


if __name__ == "__main__":
    main()
