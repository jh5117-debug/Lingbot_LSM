"""
Aggregate ablation evaluation results into a comparison table.

Reads all eval_report.json files under --ablation_dir and prints
a formatted comparison table for paper writing.

Usage:
    python aggregate_ablation.py --ablation_dir /path/to/ablation/output
"""

import argparse
import csv
import json
import os


# Display order for ablation methods (controls table row order)
METHOD_ORDER = [
    "zeroshot",
    "lora_finetune",
    "single_model",
    "epoch_2",
    "epoch_4",
    "epoch_6",
    "epoch_8",
    "epoch_10",
    "final_dual_model",
]

METHOD_DISPLAY_NAMES = {
    "zeroshot":          "Base (zero-shot)",
    "lora_finetune":     "LoRA fine-tune",
    "single_model":      "Single-model FT",
    "epoch_2":           "Dual-model (epoch 2)",
    "epoch_4":           "Dual-model (epoch 4)",
    "epoch_6":           "Dual-model (epoch 6)",
    "epoch_8":           "Dual-model (epoch 8)",
    "epoch_10":          "Dual-model (epoch 10)",
    "final_dual_model":  "Dual-model (ours, final)",
}


def load_report(report_path):
    with open(report_path) as f:
        return json.load(f)


def format_metric(value, digits=3):
    if value is None:
        return "  —  "
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--ablation_dir", type=str, required=True,
                        help="Directory containing ablation subdirectories")
    args = parser.parse_args()

    ablation_dir = args.ablation_dir

    # ---- Discover all ablation results ----
    results = {}
    for method_name in os.listdir(ablation_dir):
        report_path = os.path.join(ablation_dir, method_name, "eval_report.json")
        if os.path.exists(report_path):
            try:
                report = load_report(report_path)
                agg = report.get("aggregate_metrics", {})
                results[method_name] = {
                    "psnr": agg.get("psnr", {}).get("mean"),
                    "ssim": agg.get("ssim", {}).get("mean"),
                    "lpips": agg.get("lpips", {}).get("mean"),
                    "gen_time_s": agg.get("gen_time_s", {}).get("mean"),
                    "num_evaluated": report.get("config", {}).get("num_evaluated", 0),
                }
            except Exception as e:
                print(f"[WARN] Failed to load {report_path}: {e}")

    if not results:
        print(f"No eval_report.json files found under {ablation_dir}")
        return

    # ---- Sort by method order ----
    ordered_methods = []
    for m in METHOD_ORDER:
        if m in results:
            ordered_methods.append(m)
    # Append any methods not in the order list
    for m in sorted(results.keys()):
        if m not in ordered_methods:
            ordered_methods.append(m)

    # ---- Print table ----
    col_method = 28
    col_metric = 9

    header = (
        f"{'Method':<{col_method}} | "
        f"{'PSNR↑':>{col_metric}} | "
        f"{'SSIM↑':>{col_metric}} | "
        f"{'LPIPS↓':>{col_metric}} | "
        f"{'Time(s)':>{col_metric}} | "
        f"{'#Clips':>6}"
    )
    sep = "-" * len(header)

    print()
    print("Ablation Study Results")
    print(sep)
    print(header)
    print(sep)

    table_rows = []
    for method in ordered_methods:
        r = results[method]
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row = (
            f"{display_name:<{col_method}} | "
            f"{format_metric(r['psnr'], 2):>{col_metric}} | "
            f"{format_metric(r['ssim'], 4):>{col_metric}} | "
            f"{format_metric(r['lpips'], 4):>{col_metric}} | "
            f"{format_metric(r['gen_time_s'], 1):>{col_metric}} | "
            f"{r['num_evaluated']:>6}"
        )
        print(row)
        table_rows.append({
            "method": display_name,
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "lpips": r["lpips"],
            "gen_time_s": r["gen_time_s"],
            "num_evaluated": r["num_evaluated"],
        })

    print(sep)
    print()

    # ---- Save CSV ----
    csv_path = os.path.join(ablation_dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "psnr", "ssim", "lpips", "gen_time_s", "num_evaluated"]
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved: {csv_path}")

    # ---- Save JSON ----
    json_path = os.path.join(ablation_dir, "ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump({"methods": table_rows, "raw": results}, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
