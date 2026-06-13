"""
check_dataset.py
============================================================
DCASE 2025 Task 2 - 数据集完整性校验脚本
============================================================
功能:
    1. 遍历 dev_data/raw/ 目录，统计每个机器类型的 wav 文件数量
    2. 随机抽取音频文件，验证采样率、通道数、时长
    3. 检查 attributes_00.csv 是否存在
"""

import os
import sys
import random
import csv
from pathlib import Path
from collections import defaultdict

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("[Error] 缺少依赖，请运行: pip install librosa soundfile")
    sys.exit(1)

# 项目根目录 (tools/ 的上一级)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_DATA_DIR = _PROJECT_ROOT / "dev_data" / "raw"

EXPECTED_MACHINES = [
    "ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"
]

EXPECTED_SUBDIRS = ["train", "test", "supplemental"]


def count_wav_files(directory: Path) -> int:
    """递归统计目录下的 .wav 文件数量"""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*.wav"))


def check_csv_exists(machine_dir: Path) -> bool:
    """检查 attributes_00.csv 是否存在"""
    csv_path = machine_dir / "attributes_00.csv"
    return csv_path.exists()


def get_random_wav(machine_dir: Path) -> Path:
    """从机器目录下随机选取一个 wav 文件"""
    all_wavs = list(machine_dir.rglob("*.wav"))
    if not all_wavs:
        return None
    return random.choice(all_wavs)


def analyze_audio(wav_path: Path) -> dict:
    """
    使用 librosa 读取音频，返回元信息
    """
    try:
        # 获取音频信息（不加载全部数据）
        info = sf.info(str(wav_path))
        return {
            "path": str(wav_path),
            "filename": wav_path.name,
            "sr": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "format": info.format,
        }
    except Exception as e:
        # 如果 soundfile 失败，尝试 librosa
        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=False, duration=1.0)
            return {
                "path": str(wav_path),
                "filename": wav_path.name,
                "sr": sr,
                "channels": 1 if y.ndim == 1 else y.shape[0],
                "duration": len(y) / sr if y.ndim == 1 else y.shape[1] / sr,
                "frames": len(y) if y.ndim == 1 else y.shape[1],
                "format": "loaded via librosa",
            }
        except Exception as e2:
            return {
                "path": str(wav_path),
                "filename": wav_path.name,
                "error": str(e2),
            }


def print_section(title: str):
    """打印分隔线标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_section("DCASE 2025 Task 2 - 数据集完整性校验")

    if not DEV_DATA_DIR.exists():
        print(f"\n[Error] 数据目录不存在: {DEV_DATA_DIR}")
        print("        请先运行 download_data.py 下载数据")
        sys.exit(1)

    # ---- 发现所有机器目录 ----
    found_machines = sorted([d.name for d in DEV_DATA_DIR.iterdir() if d.is_dir()])
    print(f"\n数据目录: {DEV_DATA_DIR.resolve()}")
    print(f"发现的机器类型 ({len(found_machines)}): {', '.join(found_machines)}")

    # 检查缺失的机器
    missing = [m for m in EXPECTED_MACHINES if m not in found_machines]
    if missing:
        print(f"\n[警告] 以下机器类型目录缺失: {', '.join(missing)}")

    extra = [m for m in found_machines if m not in EXPECTED_MACHINES]
    if extra:
        print(f"\n[信息] 发现额外目录: {', '.join(extra)}")

    # ---- 统计每个机器的文件数量 ----
    print_section("文件数量统计")

    all_stats = {}
    total_wavs = 0
    all_wav_paths = []

    # 表头
    print(f"\n{'机器类型':<15} {'train':>8} {'test':>8} {'supplemental':>14} {'CSV':>5} {'合计':>8}")
    print("-" * 65)

    for machine in sorted(found_machines):
        machine_dir = DEV_DATA_DIR / machine
        stats = {}

        for subdir in EXPECTED_SUBDIRS:
            subdir_path = machine_dir / subdir
            count = count_wav_files(subdir_path)
            stats[subdir] = count
            total_wavs += count

            # 收集 wav 路径
            if subdir_path.exists():
                all_wav_paths.extend(subdir_path.rglob("*.wav"))

        csv_exists = check_csv_exists(machine_dir)
        stats["csv"] = csv_exists
        stats["total"] = sum(stats[s] for s in EXPECTED_SUBDIRS)

        all_stats[machine] = stats

        csv_mark = "Y" if csv_exists else "N"
        print(
            f"{machine:<15} "
            f"{stats['train']:>8} "
            f"{stats['test']:>8} "
            f"{stats['supplemental']:>14} "
            f"{csv_mark:>5} "
            f"{stats['total']:>8}"
        )

    print("-" * 65)
    grand_total = sum(s["total"] for s in all_stats.values())
    print(f"{'合计':<15} {'':>8} {'':>8} {'':>14} {'':>5} {grand_total:>8}")

    # ---- 随机音频校验 ----
    print_section("随机音频校验")

    if all_wav_paths:
        random.seed(42)
        sample_wav = random.choice(all_wav_paths)
        print(f"\n随机抽取: {sample_wav}")

        audio_info = analyze_audio(sample_wav)

        if "error" in audio_info:
            print(f"\n[Error] 音频读取失败: {audio_info['error']}")
        else:
            print(f"  文件名:    {audio_info['filename']}")
            print(f"  采样率:    {audio_info['sr']} Hz")
            print(f"  通道数:    {audio_info['channels']}")
            print(f"  时长:      {audio_info['duration']:.2f} 秒")
            print(f"  帧数:      {audio_info['frames']}")
            print(f"  格式:      {audio_info['format']}")

        # 额外多抽几个做快速校验
        print(f"\n--- 额外抽样 5 个文件快速校验 ---")
        samples = random.sample(all_wav_paths, min(5, len(all_wav_paths)))
        for wav in samples:
            info = analyze_audio(wav)
            if "error" in info:
                print(f"  [Error] {wav.name}: {info['error']}")
            else:
                print(
                    f"  [OK] {wav.name:<50} "
                    f"sr={info['sr']}Hz, ch={info['channels']}, "
                    f"dur={info['duration']:.1f}s"
                )
    else:
        print("\n[警告] 未找到任何 .wav 文件!")

    # ---- CSV 内容预览 ----
    print_section("attributes_00.csv 内容预览")

    for machine in sorted(found_machines):
        csv_path = DEV_DATA_DIR / machine / "attributes_00.csv"
        if csv_path.exists():
            print(f"\n--- {machine}/attributes_00.csv ---")
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i < 3:  # 只显示前 3 行
                            print(f"  行{i}: {row}")
                        else:
                            break
                # 统计行数
                with open(csv_path, "r", encoding="utf-8") as f:
                    total_rows = sum(1 for _ in f)
                print(f"  ... (共 {total_rows} 行)")
            except Exception as e:
                print(f"  [Error] 无法读取: {e}")
        else:
            print(f"\n--- {machine}/attributes_00.csv ---")
            print(f"  [缺失] CSV 文件不存在")

    # ---- 目录树概览 ----
    print_section("目录结构概览")
    for machine in sorted(found_machines):
        machine_dir = DEV_DATA_DIR / machine
        print(f"\n  {machine}/")
        for subdir in EXPECTED_SUBDIRS:
            subdir_path = machine_dir / subdir
            if subdir_path.exists():
                wav_count = count_wav_files(subdir_path)
                print(f"    ├── {subdir}/  ({wav_count} .wav 文件)")
            else:
                print(f"    ├── {subdir}/  [不存在]")
        csv_path = machine_dir / "attributes_00.csv"
        csv_mark = "存在" if csv_path.exists() else "不存在"
        print(f"    └── attributes_00.csv  [{csv_mark}]")

    # ---- 总结 ----
    print_section("校验总结")
    csv_count = sum(1 for s in all_stats.values() if s["csv"])
    print(f"\n  机器类型总数:    {len(found_machines)}/{len(EXPECTED_MACHINES)}")
    print(f"  WAV 文件总数:    {total_wavs}")
    print(f"  CSV 文件覆盖:    {csv_count}/{len(found_machines)}")
    print(f"  数据目录:        {DEV_DATA_DIR.resolve()}")

    if len(found_machines) == len(EXPECTED_MACHINES) and csv_count == len(EXPECTED_MACHINES):
        print(f"\n  [PASS] 数据集完整性校验通过!")
    else:
        print(f"\n  [WARN] 数据集可能存在不完整项，请检查上方详情")


if __name__ == "__main__":
    main()
