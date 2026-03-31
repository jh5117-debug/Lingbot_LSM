"""
Quick validation script to check that all dependencies and paths are correct
before starting training. Run this first!

Usage:
    python validate_setup.py \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act/ \
        --lingbot_code_dir /home/nvme02/lingbot-world/code/lingbot-world \
        --dataset_dir /home/nvme02/lingbot-world/datasets/csgo_processed/
"""

import argparse
import os
import sys


def check_path(path, description, is_dir=True):
    exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {description}: {path}")
    return exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--lingbot_code_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    all_ok = True
    print("\n=== Checking Model Weights ===")
    all_ok &= check_path(args.ckpt_dir, "Checkpoint directory")
    all_ok &= check_path(os.path.join(args.ckpt_dir, "low_noise_model"), "Low noise model")
    all_ok &= check_path(os.path.join(args.ckpt_dir, "high_noise_model"), "High noise model")
    all_ok &= check_path(os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"), "VAE weights", is_dir=False)
    all_ok &= check_path(os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"), "T5 weights", is_dir=False)
    all_ok &= check_path(os.path.join(args.ckpt_dir, "google", "umt5-xxl"), "T5 tokenizer")

    # Check config.json
    config_path = os.path.join(args.ckpt_dir, "low_noise_model", "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n  Low noise model config: {config}")
    else:
        print(f"\n  [MISSING] config.json at {config_path}")
        all_ok = False

    print("\n=== Checking LingBot Code ===")
    all_ok &= check_path(args.lingbot_code_dir, "LingBot code directory")
    all_ok &= check_path(os.path.join(args.lingbot_code_dir, "wan", "modules", "model.py"), "WanModel", is_dir=False)
    all_ok &= check_path(os.path.join(args.lingbot_code_dir, "wan", "modules", "vae2_1.py"), "VAE module", is_dir=False)
    all_ok &= check_path(os.path.join(args.lingbot_code_dir, "wan", "modules", "t5.py"), "T5 module", is_dir=False)
    all_ok &= check_path(os.path.join(args.lingbot_code_dir, "wan", "utils", "cam_utils.py"), "Camera utils", is_dir=False)

    # Try importing
    print("\n=== Testing Imports ===")
    sys.path.insert(0, args.lingbot_code_dir)
    try:
        from wan.modules.model import WanModel
        print("  [OK] WanModel import")
    except Exception as e:
        print(f"  [FAIL] WanModel import: {e}")
        all_ok = False

    try:
        from wan.modules.vae2_1 import Wan2_1_VAE
        print("  [OK] VAE import")
    except Exception as e:
        print(f"  [FAIL] VAE import: {e}")
        all_ok = False

    try:
        from wan.modules.t5 import T5EncoderModel
        print("  [OK] T5 import")
    except Exception as e:
        print(f"  [FAIL] T5 import: {e}")
        all_ok = False

    try:
        import accelerate
        print(f"  [OK] accelerate {accelerate.__version__}")
    except ImportError:
        print("  [FAIL] accelerate not installed")
        all_ok = False

    try:
        from peft import LoraConfig
        print("  [OK] peft (LoRA)")
    except ImportError:
        print("  [FAIL] peft not installed (pip install peft)")
        all_ok = False

    try:
        import flash_attn
        print(f"  [OK] flash_attn {flash_attn.__version__}")
    except ImportError:
        print("  [WARN] flash_attn not installed (training may be slower)")

    try:
        import deepspeed
        print(f"  [OK] deepspeed {deepspeed.__version__}")
    except ImportError:
        print("  [WARN] deepspeed not installed (needed for ZeRO)")

    if args.dataset_dir:
        print("\n=== Checking Dataset ===")
        all_ok &= check_path(args.dataset_dir, "Dataset directory")
        train_csv = os.path.join(args.dataset_dir, "metadata_train.csv")
        val_csv = os.path.join(args.dataset_dir, "metadata_val.csv")
        all_ok &= check_path(train_csv, "Train metadata", is_dir=False)
        all_ok &= check_path(val_csv, "Val metadata", is_dir=False)

        if os.path.exists(train_csv):
            import csv
            with open(train_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"  Train samples: {len(rows)}")
            if rows:
                clip_path = os.path.join(args.dataset_dir, rows[0]["clip_path"])
                print(f"  First clip: {clip_path}")
                all_ok &= check_path(clip_path, "First clip directory")
                all_ok &= check_path(os.path.join(clip_path, "video.mp4"), "Video file", is_dir=False)
                all_ok &= check_path(os.path.join(clip_path, "poses.npy"), "Poses file", is_dir=False)
                all_ok &= check_path(os.path.join(clip_path, "action.npy"), "Action file", is_dir=False)
                all_ok &= check_path(os.path.join(clip_path, "intrinsics.npy"), "Intrinsics file", is_dir=False)

        if os.path.exists(val_csv):
            import csv
            with open(val_csv) as f:
                rows = list(csv.DictReader(f))
            print(f"  Val samples: {len(rows)}")

    print("\n=== GPU Check ===")
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"  Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    except Exception as e:
        print(f"  [WARN] CUDA not available: {e}")

    print()
    if all_ok:
        print("ALL CHECKS PASSED! Ready to train.")
    else:
        print("SOME CHECKS FAILED. Fix the issues above before training.")
    print()


if __name__ == "__main__":
    main()
