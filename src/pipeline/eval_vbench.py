"""
eval_vbench.py — 多模型 VBench 评测脚本

功能：
  1. 批量推理：对测试集每张图片，调用各模型的推理脚本生成视频
  2. VBench 评分：对生成的视频调用 VBench（custom_input 模式）评分
  3. 汇总结果：将各模型各维度分数汇总为 CSV 并打印 Markdown 对比表格

模型配置通过 eval_model_configs.yaml 传入。
若配置文件不存在，自动生成模板并提示用户填写后重新运行。

YAML 模板示例（复制到 eval_model_configs.yaml 并填写路径后使用）：
----------------------------------------------------------------------
groupA:
  name: "LingBot-World (ours)"
  infer_script: "src/pipeline/infer_v2.py"
  ckpt_dir: ""        # 用户填写基础模型目录
  lora_path: ""       # 可选，留空则不使用
  use_memory: false   # 是否启用 Memory Bank
  extra_args: []      # 额外命令行参数，会追加在命令末尾

groupB:
  name: "Yume-1.5"
  infer_script: ""    # 用户填写 Yume 推理脚本路径
  ckpt_dir: ""        # 用户填写
  extra_args: []

groupC:
  name: "HunyuanVideo-World 1.5"
  infer_script: ""    # 用户填写
  ckpt_dir: ""        # 用户填写
  extra_args: []
----------------------------------------------------------------------

用法：
  python eval_vbench.py \\
      --test_images_dir eval_data/images/ \\
      --test_traj_dir   eval_data/trajectories/ \\
      --output_dir      outputs/eval_vbench/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 日志设置（对齐 infer_v2.py 格式）
# ---------------------------------------------------------------------------

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认 VBench 维度（对应论文 Table 2 的 6 个）
# ---------------------------------------------------------------------------

DEFAULT_DIMENSIONS = [
    "imaging_quality",
    "aesthetic_quality",
    "dynamic_degree",
    "motion_smoothness",
    "temporal_flickering",
    "subject_consistency",
]

# ---------------------------------------------------------------------------
# eval_model_configs.yaml 模板
# ---------------------------------------------------------------------------

_YAML_TEMPLATE = """\
# eval_model_configs.yaml — 评测模型配置
# 填写以下字段后运行 eval_vbench.py（或 run_eval.sh）

groupA:
  name: "LingBot-World (ours)"
  infer_script: "src/pipeline/infer_v2.py"
  ckpt_dir: ""        # 基础模型目录（必填）
  lora_path: ""       # LoRA 权重路径（可选，留空则不使用）
  use_memory: false   # 是否启用 Memory Bank
  extra_args: []      # 额外命令行参数列表，示例: ["--sample_steps", "50"]
  launcher: "torchrun"  # 可选 python 或 torchrun（默认 torchrun）

groupB:
  name: "Yume-1.5"
  infer_script: ""    # Yume 推理脚本路径（必填）
  ckpt_dir: ""        # 必填
  extra_args: []
  launcher: "python"  # 或 torchrun

groupC:
  name: "HunyuanVideo-World 1.5"
  infer_script: ""    # 必填
  ckpt_dir: ""        # 必填
  extra_args: []
  launcher: "python"  # 或 torchrun
"""

# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="多模型 VBench 批量评测脚本"
    )

    parser.add_argument(
        "--test_images_dir", type=str, default="eval_data/images/",
        help="测试图片目录（.jpg/.png，默认 eval_data/images/）",
    )
    parser.add_argument(
        "--test_traj_dir", type=str, default="eval_data/trajectories/",
        help="相机轨迹目录（与图片同名，后缀不同，默认 eval_data/trajectories/）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/eval_vbench/",
        help="生成视频和评测结果的根目录（默认 outputs/eval_vbench/）",
    )
    parser.add_argument(
        "--model_config", type=str, default="eval_model_configs.yaml",
        help="模型配置 YAML 文件路径（默认 eval_model_configs.yaml）",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["groupA", "groupB", "groupC"],
        help="要评测的模型 key 列表（默认 groupA groupB groupC）",
    )
    parser.add_argument(
        "--skip_inference", action="store_true", default=False,
        help="跳过推理，直接对已有视频评分",
    )
    parser.add_argument(
        "--skip_vbench", action="store_true", default=False,
        help="跳过 VBench 评分，只做推理",
    )
    parser.add_argument(
        "--dimensions", type=str, nargs="+", default=DEFAULT_DIMENSIONS,
        help="VBench 评测维度列表（默认覆盖论文 Table 2 的 6 个维度）",
    )
    parser.add_argument(
        "--vbench_mode", type=str, default="custom_input",
        choices=["custom_input"],
        help="VBench 评测模式（目前仅支持 custom_input）",
    )
    parser.add_argument(
        "--frame_num", type=int, default=81,
        help="每个视频生成帧数（默认 81）",
    )
    parser.add_argument(
        "--size", type=str, default="480*832",
        help="分辨率（默认 480*832）",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="First-person view of CS:GO competitive gameplay",
        help="生成 prompt（默认 First-person view of CS:GO competitive gameplay）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 模型配置加载
# ---------------------------------------------------------------------------

def _load_or_create_model_config(config_path: str) -> dict:
    """加载模型配置 YAML；若不存在则生成模板并退出。"""
    if not os.path.exists(config_path):
        logger.warning(
            f"模型配置文件 '{config_path}' 不存在，正在自动生成模板..."
        )
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(_YAML_TEMPLATE)
        logger.info(
            f"模板已生成：{config_path}\n"
            f"请填写各模型的 ckpt_dir、infer_script 等字段后重新运行。"
        )
        sys.exit(0)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        logger.error(f"配置文件 '{config_path}' 内容为空，请填写后重新运行。")
        sys.exit(1)

    return config


# ---------------------------------------------------------------------------
# 测试集加载
# ---------------------------------------------------------------------------

def _collect_test_images(images_dir: str, traj_dir: str) -> list:
    """收集测试集图片和对应轨迹文件，返回 [(img_path, traj_path), ...] 列表。"""
    images_dir = Path(images_dir)
    traj_dir = Path(traj_dir)

    if not images_dir.exists():
        logger.error(f"测试图片目录不存在：{images_dir}")
        sys.exit(1)

    if not traj_dir.exists():
        logger.warning(f"相机轨迹目录不存在：{traj_dir}，将跳过所有需要轨迹的样本")
        # 不 exit，因为某些模型可能不需要轨迹，这里只是 warning

    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])

    if not image_files:
        logger.error(f"测试图片目录中没有图片：{images_dir}")
        sys.exit(1)

    pairs = []
    for img_path in image_files:
        # 查找同名轨迹文件（后缀不限）
        traj_path = None
        if traj_dir.exists():
            for candidate in traj_dir.iterdir():
                if candidate.stem == img_path.stem:
                    traj_path = candidate
                    break

        if traj_path is None:
            logger.warning(
                f"找不到图片 '{img_path.name}' 对应的轨迹文件，跳过该样本"
            )
            continue

        pairs.append((img_path, traj_path))

    logger.info(f"共找到 {len(pairs)} 对有效测试样本")
    return pairs


# ---------------------------------------------------------------------------
# 单模型批量推理
# ---------------------------------------------------------------------------

def _run_inference_for_model(
    model_key: str,
    model_cfg: dict,
    test_pairs: list,
    output_dir: Path,
    args,
) -> Path:
    """对单个模型跑所有测试样本的推理，返回该模型的视频输出目录。"""
    model_name = model_cfg.get("name", model_key)
    model_video_dir = output_dir / model_key
    model_video_dir.mkdir(parents=True, exist_ok=True)

    infer_script = model_cfg.get("infer_script", "")
    ckpt_dir = model_cfg.get("ckpt_dir", "")
    lora_path = model_cfg.get("lora_path", "")
    use_memory = model_cfg.get("use_memory", False)
    extra_args = model_cfg.get("extra_args", [])

    if not infer_script:
        logger.error(
            f"模型 '{model_key}' 的 infer_script 未配置，跳过推理。"
        )
        return model_video_dir

    total = len(test_pairs)
    logger.info(
        f"[{model_name}] 开始推理，共 {total} 张图片，"
        f"输出目录：{model_video_dir}"
    )

    for idx, (img_path, traj_path) in enumerate(test_pairs, start=1):
        output_video = model_video_dir / f"{img_path.stem}.mp4"

        # 断点续跑：已存在则跳过
        if output_video.exists():
            logger.info(
                f"[{model_name}] ({idx}/{total}) 已存在，跳过：{output_video.name}"
            )
            continue

        logger.info(
            f"[{model_name}] ({idx}/{total}) 推理：{img_path.name} → {output_video.name}"
        )

        # 拼接推理命令，根据 launcher 决定使用 torchrun 还是 python
        launcher = model_cfg.get("launcher", "torchrun")
        if launcher == "torchrun":
            cmd = [
                "torchrun", "--nproc_per_node=1",
                str(infer_script),
            ]
        else:
            cmd = [
                "python",
                str(infer_script),
            ]
        cmd += [
            "--ckpt_dir", str(ckpt_dir),
            "--image", str(img_path),
            "--action_path", str(traj_path),
            "--save_file", str(output_video),
            "--prompt", args.prompt,
            "--frame_num", str(args.frame_num),
            "--size", args.size,
        ]

        # LoRA 权重（组A专有，组B/C通过 extra_args 覆盖）
        if lora_path:
            cmd += ["--lora_path", str(lora_path)]

        # Memory Bank（组A专有）
        if use_memory:
            cmd += ["--use_memory"]

        # 额外参数（组B/C用此覆盖命令行格式）
        if extra_args:
            cmd += [str(a) for a in extra_args]

        logger.info(f"执行命令：{' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # 让输出直接打印到 stdout/stderr
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"[{model_name}] ({idx}/{total}) 推理失败（returncode={e.returncode}），"
                f"跳过：{img_path.name}"
            )
            continue
        except Exception as e:
            logger.warning(
                f"[{model_name}] ({idx}/{total}) 推理异常：{e}，跳过：{img_path.name}"
            )
            continue

    logger.info(f"[{model_name}] 推理完成，视频保存至：{model_video_dir}")
    return model_video_dir


# ---------------------------------------------------------------------------
# VBench 评分
# ---------------------------------------------------------------------------

def _check_vbench_installed():
    """检查 VBench 是否已安装；未安装则打印提示并退出。"""
    try:
        import importlib
        importlib.import_module("vbench")
    except ImportError:
        logger.error(
            "VBench 未安装。请按以下步骤安装：\n"
            "  pip install vbench\n"
            "或参考官方文档：https://github.com/Vchitect/VBench"
        )
        sys.exit(1)


def _run_vbench_for_model(
    model_key: str,
    model_name: str,
    video_dir: Path,
    dimensions: list,
    vbench_mode: str,
    output_dir: Path,
) -> dict:
    """对单个模型的视频目录跑所有维度的 VBench 评分。

    返回 {dimension: score} 字典。
    """
    scores = {}
    vbench_result_dir = output_dir / "vbench_results" / model_key
    vbench_result_dir.mkdir(parents=True, exist_ok=True)

    for dim in dimensions:
        logger.info(f"[{model_name}] VBench 评测维度：{dim}")

        cmd = [
            "vbench", "evaluate",
            "--videos_path", str(video_dir),
            "--dimension", dim,
            "--mode", vbench_mode,
            "--output_path", str(vbench_result_dir),
        ]

        logger.info(f"执行命令：{' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"[{model_name}] VBench 维度 '{dim}' 评测失败"
                f"（returncode={e.returncode}），跳过。"
            )
            scores[dim] = None
            continue

        # 解析 VBench 输出 JSON
        score = _parse_vbench_result(vbench_result_dir, dim)
        scores[dim] = score
        logger.info(f"[{model_name}] {dim}: {score}")

    return scores


def _parse_vbench_result(result_dir: Path, dimension: str) -> float | None:
    """从 VBench 输出目录中解析指定维度的分数。

    VBench custom_input 模式通常输出 <dimension>_results.json，
    其结构为 {"<dimension>": [[score, ...], total_score], ...}。
    """
    # VBench 不同版本输出文件名可能有差异，尝试多个候选路径
    candidates = [
        result_dir / f"{dimension}_eval_results.json",
        result_dir / f"{dimension}_results.json",
        result_dir / f"results_{dimension}.json",
        result_dir / "results.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # custom_input 模式格式：{"video_results": [...], "dimension_score": 0.xxxx}
                if "dimension_score" in data:
                    return float(data["dimension_score"])

                # 典型格式：{dimension: [[per_video_scores], aggregate_score]}
                if dimension in data:
                    value = data[dimension]
                    if isinstance(value, list) and len(value) >= 2:
                        # aggregate_score 通常在第二位
                        return float(value[1])
                    elif isinstance(value, (int, float)):
                        return float(value)

                # 备用：直接取 "score" 键
                if "score" in data:
                    return float(data["score"])

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"解析 VBench 结果文件失败：{candidate}，错误：{e}"
                )
                continue

    logger.warning(
        f"未找到维度 '{dimension}' 的 VBench 结果文件，目录：{result_dir}"
    )
    return None


# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------

def _summarize_results(
    all_scores: dict,
    model_configs: dict,
    dimensions: list,
    output_dir: Path,
):
    """将所有模型的评分汇总为 CSV 并打印 Markdown 对比表格。

    参数：
        all_scores: {model_key: {dimension: score}}
        model_configs: {model_key: {..., name: ...}}
        dimensions: 维度列表
        output_dir: 输出根目录
    """
    csv_path = output_dir / "results_summary.csv"

    # ---- 写 CSV ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model"] + dimensions
        writer.writerow(header)

        for model_key, dim_scores in all_scores.items():
            model_name = model_configs.get(model_key, {}).get("name", model_key)
            row = [model_name]
            for dim in dimensions:
                score = dim_scores.get(dim)
                row.append(f"{score:.4f}" if score is not None else "N/A")
            writer.writerow(row)

    logger.info(f"CSV 汇总已保存：{csv_path}")

    # ---- 打印 Markdown 表格（对齐论文 Table 2 格式）----
    _print_markdown_table(all_scores, model_configs, dimensions)


def _print_markdown_table(
    all_scores: dict,
    model_configs: dict,
    dimensions: list,
):
    """打印 Markdown 格式对比表格。"""
    # 列宽计算
    model_col_width = max(
        len("Model"),
        max((len(model_configs.get(k, {}).get("name", k)) for k in all_scores), default=5),
    )
    dim_col_widths = {
        dim: max(len(dim), 6) for dim in dimensions
    }

    # 表头
    header_parts = [f"{'Model':<{model_col_width}}"]
    for dim in dimensions:
        header_parts.append(f"{dim:^{dim_col_widths[dim]}}")
    header_line = " | ".join(header_parts)

    separator_parts = ["-" * model_col_width]
    for dim in dimensions:
        separator_parts.append("-" * dim_col_widths[dim])
    separator_line = "-+-".join(separator_parts)

    print("\n" + "=" * len(header_line))
    print("  VBench 评测结果（Table 2 对齐格式）")
    print("=" * len(header_line))
    print("| " + header_line + " |")
    print("|-" + separator_line + "-|")

    for model_key, dim_scores in all_scores.items():
        model_name = model_configs.get(model_key, {}).get("name", model_key)
        row_parts = [f"{model_name:<{model_col_width}}"]
        for dim in dimensions:
            score = dim_scores.get(dim)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            row_parts.append(f"{score_str:^{dim_col_widths[dim]}}")
        print("| " + " | ".join(row_parts) + " |")

    print("=" * len(header_line) + "\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # 加载模型配置（若不存在则自动生成模板并退出）
    model_configs = _load_or_create_model_config(args.model_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 过滤请求的模型
    requested_models = args.models
    available_models = [k for k in requested_models if k in model_configs]
    missing_models = [k for k in requested_models if k not in model_configs]
    if missing_models:
        logger.warning(
            f"以下模型在配置文件中不存在，已跳过：{missing_models}"
        )
    if not available_models:
        logger.error("没有有效的模型可以评测，请检查 --models 和配置文件。")
        sys.exit(1)

    logger.info(
        f"将评测的模型：{[model_configs[k].get('name', k) for k in available_models]}"
    )

    # ---- Step 1：收集测试集 ----
    test_pairs = _collect_test_images(args.test_images_dir, args.test_traj_dir)

    # ---- Step 2：批量推理 ----
    model_video_dirs = {}
    if not args.skip_inference:
        logger.info("=" * 60)
        logger.info("开始批量推理阶段")
        logger.info("=" * 60)
        for model_key in available_models:
            model_cfg = model_configs[model_key]
            video_dir = _run_inference_for_model(
                model_key=model_key,
                model_cfg=model_cfg,
                test_pairs=test_pairs,
                output_dir=output_dir,
                args=args,
            )
            model_video_dirs[model_key] = video_dir
    else:
        logger.info("--skip_inference 已设置，跳过推理阶段，使用已有视频。")
        for model_key in available_models:
            video_dir = output_dir / model_key
            if not video_dir.exists():
                logger.warning(
                    f"跳过推理，但模型 '{model_key}' 的视频目录不存在：{video_dir}"
                )
            model_video_dirs[model_key] = video_dir

    # ---- Step 3：VBench 评分 ----
    all_scores = {}
    if not args.skip_vbench:
        logger.info("=" * 60)
        logger.info("开始 VBench 评分阶段")
        logger.info("=" * 60)
        _check_vbench_installed()

        for model_key in available_models:
            model_name = model_configs[model_key].get("name", model_key)
            video_dir = model_video_dirs[model_key]

            if not video_dir.exists():
                logger.warning(
                    f"模型 '{model_name}' 的视频目录不存在，跳过 VBench 评分：{video_dir}"
                )
                all_scores[model_key] = {dim: None for dim in args.dimensions}
                continue

            scores = _run_vbench_for_model(
                model_key=model_key,
                model_name=model_name,
                video_dir=video_dir,
                dimensions=args.dimensions,
                vbench_mode=args.vbench_mode,
                output_dir=output_dir,
            )
            all_scores[model_key] = scores
    else:
        logger.info("--skip_vbench 已设置，跳过 VBench 评分阶段。")

    # ---- Step 4：汇总结果 ----
    if all_scores:
        logger.info("=" * 60)
        logger.info("汇总评测结果")
        logger.info("=" * 60)
        _summarize_results(
            all_scores=all_scores,
            model_configs=model_configs,
            dimensions=args.dimensions,
            output_dir=output_dir,
        )
    else:
        logger.info("无评分数据可汇总（仅执行了推理或全部跳过）。")

    logger.info("eval_vbench.py 全部完成。")


if __name__ == "__main__":
    main()
