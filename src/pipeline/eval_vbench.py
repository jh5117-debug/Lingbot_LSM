"""
eval_vbench.py — 多模型 VBench 评测脚本

功能：
  1. 批量推理：对测试集每张图片，调用各模型的推理脚本生成视频
     - 若 YAML 中 video_dir 非空，则跳过推理，直接使用已有视频（demo 模式）
  2. VBench 评分：对生成的视频调用 VBench（custom_input 模式）评分
  3. 汇总结果：
     - results_summary.csv（aggregate 分数，每模型一行）
     - results_per_clip.csv（per-clip 分数，每 clip×model 一行）

模型配置通过 eval_model_configs.yaml 传入。
若配置文件不存在，自动生成模板并提示用户填写后重新运行。

YAML 模板示例（复制到 eval_model_configs.yaml 并填写路径后使用）：
----------------------------------------------------------------------
baseline:
  name: "Baseline"
  video_dir: "outputs/inference/baseline"   # 有值 → demo 模式，直接使用已有视频
  infer_script: ""                           # demo 模式下不需要填
  ckpt_dir: ""

v3_mem:
  name: "v3 + Memory (epoch 5)"
  video_dir: "outputs/inference/v3_stage1_dual_epoch_5_mem"

yume:
  name: "Yume-1.5"
  video_dir: ""         # 空 → full pipeline 模式，走推理生成
  infer_script: "..."
  ckpt_dir: "..."
  launcher: "python"
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
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
# 项目根目录（用于将相对 video_dir 解析为绝对路径）
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (_SCRIPT_DIR / "../..").resolve()

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
#
# video_dir 规则：
#   - 非空 → demo 模式，直接使用已有视频，跳过推理
#   - 空或不存在 → full pipeline 模式，走推理生成（需填 infer_script + ckpt_dir）
# 每个 group 独立决定，无需全局 skip_inference。

baseline:
  name: "Baseline"
  video_dir: ""         # 有值则跳过推理，直接使用该目录的已有视频
  infer_script: "src/pipeline/infer_v2.py"
  ckpt_dir: ""          # 基础模型目录（必填，full pipeline 模式）
  lora_path: ""         # LoRA 权重路径（可选，留空则不使用）
  use_memory: false     # 是否启用 Memory Bank
  extra_args: []        # 额外命令行参数列表，示例: ["--sample_steps", "50"]
  launcher: "torchrun"  # 可选 python 或 torchrun（默认 torchrun）

groupB:
  name: "Yume-1.5"
  video_dir: ""         # 空 → full pipeline 模式
  infer_script: ""      # Yume 推理脚本路径（必填）
  ckpt_dir: ""          # 必填
  extra_args: []
  launcher: "python"    # 或 torchrun

groupC:
  name: "HunyuanVideo-World 1.5"
  video_dir: ""         # 空 → full pipeline 模式
  infer_script: ""      # 必填
  ckpt_dir: ""          # 必填
  extra_args: []
  launcher: "python"    # 或 torchrun
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
        "--models", type=str, nargs="+", default=["baseline", "groupB", "groupC"],
        help="要评测的模型 key 列表（默认 baseline groupB groupC）",
    )
    parser.add_argument(
        "--skip_inference", action="store_true", default=False,
        help="跳过所有模型的推理（全局快捷方式），直接对已有视频评分",
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
# Clip 名称归一化
# ---------------------------------------------------------------------------

# 时间戳后缀模式：_v<数字>_YYYYMMDD_HHMMSS
_TIMESTAMP_SUFFIX_RE = re.compile(r'_v[0-9]+_\d{8}_\d{6}$')


def _normalize_clip_name(filename: str) -> str:
    """归一化 clip 名称：去掉 .mp4 后缀和时间戳后缀。

    示例：
      "clip001_v3_20260421_153000.mp4" → "clip001"
      "clip001.mp4"                   → "clip001"
      "clip001"                       → "clip001"
      "/path/to/clip001_v3_20260421_153000.mp4" → "clip001"
    """
    name = Path(filename).name   # 取纯文件名，防止完整路径输入
    # 去掉 .mp4 后缀（大小写不敏感）
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    # 去掉时间戳后缀
    name = _TIMESTAMP_SUFFIX_RE.sub("", name)
    return name


# ---------------------------------------------------------------------------
# 单模型批量推理
# ---------------------------------------------------------------------------

def _run_inference_for_model(
    model_key: str,
    model_cfg: dict,
    test_pairs: list,
    output_dir: Path,
    args,
) -> Optional[Path]:
    """对单个模型跑所有测试样本的推理，返回该模型的视频输出目录。

    若 model_cfg['video_dir'] 非空，则为 demo 模式：
      - 解析路径（相对 PROJECT_ROOT），检查存在性
      - 直接返回该路径（不跑推理）
      - 若目录不存在则 logger.error 并返回 None

    否则为 full pipeline 模式，走推理逻辑，生成视频到 output_dir/{model_key}/。
    """
    model_name = model_cfg.get("name", model_key)

    # ---- demo 模式检查 ----
    video_dir_cfg = model_cfg.get("video_dir", "")
    if video_dir_cfg:
        # 解析绝对路径
        vd_path = Path(video_dir_cfg)
        if not vd_path.is_absolute():
            vd_path = PROJECT_ROOT / vd_path
        vd_path = vd_path.resolve()

        if not vd_path.exists():
            logger.error(
                f"[{model_name}] demo 模式：video_dir 指定的目录不存在：{vd_path}"
            )
            return None

        logger.info(
            f"[{model_name}] demo 模式：直接使用已有视频目录 {vd_path}，跳过推理。"
        )
        return vd_path

    # ---- full pipeline 模式 ----
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
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Optional[float]]]]:
    """对单个模型的视频目录跑所有维度的 VBench 评分。

    返回 (aggregate_scores, per_clip_scores)：
      - aggregate_scores: {dimension: score}
      - per_clip_scores:  {dimension: {clip_name: score}}
    """
    scores: Dict[str, Optional[float]] = {}
    per_clip_scores: Dict[str, Dict[str, Optional[float]]] = {}

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
            per_clip_scores[dim] = {}
            continue

        # 解析 aggregate 分数
        score = _parse_vbench_result(vbench_result_dir, dim)
        scores[dim] = score
        logger.info(f"[{model_name}] {dim}: {score}")

        # 解析 per-clip 分数
        clip_scores = _parse_vbench_per_clip(vbench_result_dir, dim)
        per_clip_scores[dim] = clip_scores
        if clip_scores:
            logger.info(
                f"[{model_name}] {dim}: 解析到 {len(clip_scores)} 条 per-clip 分数"
            )

    return scores, per_clip_scores


def _parse_vbench_result(result_dir: Path, dimension: str) -> Optional[float]:
    """从 VBench 输出目录中解析指定维度的 aggregate 分数。

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


def _parse_vbench_per_clip(
    result_dir: Path, dimension: str
) -> Dict[str, Optional[float]]:
    """从 VBench 输出 JSON 中解析 per-video 分数。

    支持两种 video_results 格式：
      1. {"video_results": [[filename, score], ...], "dimension_score": 0.xxx}
      2. {"video_results": {"filename.mp4": score, ...}}

    返回 {clip_name: score} 字典（key 为 _normalize_clip_name 后的结果）。
    若 video_results 不存在或解析失败，返回空字典（graceful degrade）。
    """
    candidates = [
        result_dir / f"{dimension}_eval_results.json",
        result_dir / f"{dimension}_results.json",
        result_dir / f"results_{dimension}.json",
        result_dir / "results.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)

            video_results = data.get("video_results")
            if video_results is None:
                continue

            clip_scores: Dict[str, Optional[float]] = {}

            if isinstance(video_results, list):
                # 格式 1: [[filename, score], ...]
                for entry in video_results:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        filename = str(entry[0])
                        try:
                            score = float(entry[1])
                        except (TypeError, ValueError):
                            score = None
                        clip_name = _normalize_clip_name(filename)
                        clip_scores[clip_name] = score

            elif isinstance(video_results, dict):
                # 格式 2: {"filename.mp4": score, ...}
                for filename, raw_score in video_results.items():
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = None
                    clip_name = _normalize_clip_name(str(filename))
                    clip_scores[clip_name] = score

            return clip_scores

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            logger.warning(
                f"解析 per-clip 结果失败：{candidate}，错误：{e}"
            )
            continue

    return {}


# ---------------------------------------------------------------------------
# Per-clip 汇总
# ---------------------------------------------------------------------------

def _summarize_per_clip_results(
    all_per_clip: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    model_configs: dict,
    dimensions: list,
    output_dir: Path,
    model_video_dirs: Dict[str, Optional[Path]],
):
    """将 per-clip 评分汇总为 results_per_clip.csv。

    参数：
        all_per_clip:      {model_key: {dim: {clip_name: score}}}
        model_configs:     {model_key: {..., name: ...}}
        dimensions:        维度列表
        output_dir:        输出根目录
        model_video_dirs:  {model_key: video_dir_path}，用于构造 video_path 列

    输出列：clip_name, model_key, model_name, {dim1_score, dim2_score, ...}, video_path

    若 all_per_clip 全为空，不写 CSV，仅打印 info 说明原因。
    """
    # 检查是否有任何 per-clip 数据
    has_any_data = any(
        any(clip_dict for clip_dict in dim_dict.values())
        for dim_dict in all_per_clip.values()
    )
    if not has_any_data:
        logger.info(
            "per-clip 数据全为空（VBench 可能不支持 per-clip 输出或全部解析失败），"
            "跳过 results_per_clip.csv 写入。"
        )
        return

    csv_path = output_dir / "results_per_clip.csv"

    # 收集所有 (model_key, clip_name) 组合
    rows = []
    for model_key, dim_dict in all_per_clip.items():
        model_name = model_configs.get(model_key, {}).get("name", model_key)
        video_dir = model_video_dirs.get(model_key)

        # 收集该模型下所有出现的 clip 名
        all_clip_names: set = set()
        for clip_dict in dim_dict.values():
            all_clip_names.update(clip_dict.keys())

        for clip_name in sorted(all_clip_names):
            # 构造 video_path
            video_path = _resolve_video_path(clip_name, video_dir)

            row = {
                "clip_name": clip_name,
                "model_key": model_key,
                "model_name": model_name,
                "video_path": str(video_path) if video_path else "",
            }

            for dim in dimensions:
                dim_clip_dict = dim_dict.get(dim, {})
                score = dim_clip_dict.get(clip_name)
                row[f"{dim}_score"] = f"{score:.4f}" if score is not None else "N/A"

            rows.append(row)

    if not rows:
        logger.info("per-clip 数据为空，跳过 results_per_clip.csv 写入。")
        return

    # 写 CSV
    fieldnames = (
        ["clip_name", "model_key", "model_name"]
        + [f"{dim}_score" for dim in dimensions]
        + ["video_path"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Per-clip CSV 已保存：{csv_path}（共 {len(rows)} 行）")


def _resolve_video_path(
    clip_name: str, video_dir: Optional[Path]
) -> Optional[Path]:
    """尝试定位 clip 对应的视频文件路径。

    策略：
    1. 精确路径：{video_dir}/{clip_name}.mp4 若存在则返回
    2. Glob 匹配：在 video_dir 中找 *{clip_name}*.mp4，取第一个匹配（支持带时间戳文件名）
    3. 均不匹配：返回推断路径 {video_dir}/{clip_name}.mp4（可能不存在）
    """
    if video_dir is None:
        return None

    exact = video_dir / f"{clip_name}.mp4"
    if exact.exists():
        return exact

    # glob 匹配（支持带时间戳后缀的文件名）
    try:
        matches = sorted(video_dir.glob(f"*{clip_name}*.mp4"))
        if matches:
            return matches[0]
    except Exception:
        pass

    # 推断路径
    return exact


# ---------------------------------------------------------------------------
# 结果汇总（aggregate）
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

    # ---- Step 1：收集测试集（仅在需要 full pipeline 推理时加载）----
    # 判断是否有任何模型需要 full pipeline 推理
    need_inference = (
        not args.skip_inference
        and any(
            not model_configs[k].get("video_dir", "")
            for k in available_models
        )
    )

    if need_inference:
        test_pairs = _collect_test_images(args.test_images_dir, args.test_traj_dir)
    else:
        test_pairs = []
        logger.info("所有模型均为 demo 模式或已设置 --skip_inference，跳过测试集加载。")

    # ---- Step 2：批量推理（每个模型独立决定 demo / full pipeline）----
    model_video_dirs: Dict[str, Optional[Path]] = {}

    if args.skip_inference:
        logger.info("--skip_inference 已设置，跳过所有模型推理阶段，使用已有视频。")
        for model_key in available_models:
            model_cfg = model_configs[model_key]
            video_dir_cfg = model_cfg.get("video_dir", "")
            if video_dir_cfg:
                # demo 模式：使用 YAML 指定路径（与 _run_inference_for_model demo 分支一致）
                vd_path = Path(video_dir_cfg)
                if not vd_path.is_absolute():
                    vd_path = PROJECT_ROOT / vd_path
                vd_path = vd_path.resolve()
                if not vd_path.exists():
                    logger.warning(
                        f"[{model_cfg.get('name', model_key)}] --skip_inference: "
                        f"video_dir 指定目录不存在：{vd_path}"
                    )
                model_video_dirs[model_key] = vd_path
            else:
                # full pipeline 输出目录
                video_dir = output_dir / model_key
                if not video_dir.exists():
                    logger.warning(
                        f"跳过推理，但模型 '{model_key}' 的视频目录不存在：{video_dir}"
                    )
                model_video_dirs[model_key] = video_dir
    else:
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
            # video_dir 可能为 None（demo 模式但目录不存在）
            model_video_dirs[model_key] = video_dir

    # ---- Step 3：VBench 评分 ----
    all_scores: Dict[str, Dict[str, Optional[float]]] = {}
    all_per_clip: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    if not args.skip_vbench:
        logger.info("=" * 60)
        logger.info("开始 VBench 评分阶段")
        logger.info("=" * 60)
        _check_vbench_installed()

        for model_key in available_models:
            model_name = model_configs[model_key].get("name", model_key)
            video_dir = model_video_dirs.get(model_key)

            if video_dir is None or not video_dir.exists():
                logger.warning(
                    f"模型 '{model_name}' 的视频目录不存在或为 None，跳过 VBench 评分"
                    + (f"：{video_dir}" if video_dir else "")
                )
                all_scores[model_key] = {dim: None for dim in args.dimensions}
                all_per_clip[model_key] = {dim: {} for dim in args.dimensions}
                continue

            aggregate, per_clip = _run_vbench_for_model(
                model_key=model_key,
                model_name=model_name,
                video_dir=video_dir,
                dimensions=args.dimensions,
                vbench_mode=args.vbench_mode,
                output_dir=output_dir,
            )
            all_scores[model_key] = aggregate
            all_per_clip[model_key] = per_clip
    else:
        logger.info("--skip_vbench 已设置，跳过 VBench 评分阶段。")

    # ---- Step 4：汇总结果 ----
    if all_scores:
        logger.info("=" * 60)
        logger.info("汇总评测结果")
        logger.info("=" * 60)

        # 4a: aggregate CSV + Markdown 打印
        _summarize_results(
            all_scores=all_scores,
            model_configs=model_configs,
            dimensions=args.dimensions,
            output_dir=output_dir,
        )

        # 4b: per-clip CSV（若有数据）
        _summarize_per_clip_results(
            all_per_clip=all_per_clip,
            model_configs=model_configs,
            dimensions=args.dimensions,
            output_dir=output_dir,
            model_video_dirs=model_video_dirs,
        )
    else:
        logger.info("无评分数据可汇总（仅执行了推理或全部跳过）。")

    logger.info("eval_vbench.py 全部完成。")


if __name__ == "__main__":
    main()
