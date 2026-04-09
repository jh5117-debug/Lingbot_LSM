"""
Qualitative Visualization for CSGO Fine-tuned LingBot-World.

Generates comparison images and side-by-side videos for paper figures.

Outputs:
  1. Frame comparison grids (PNG) — GT vs generated frames at selected timesteps
  2. Side-by-side videos (MP4) — GT | Generated (| Model B) with text overlay
  3. Summary mosaics (PNG) — best/worst clips by PSNR from eval_report.json

Dependencies: OpenCV, numpy, tqdm (no matplotlib needed).

Usage:
    # Single model visualization
    python visualize_results.py \
        --gen_dirs /path/to/eval_output/videos \
        --model_names "Ours" \
        --clip_dir /path/to/processed_csgo_v3/val \
        --output_dir /path/to/vis_output

    # Two-model comparison
    python visualize_results.py \
        --gen_dirs /path/to/base_eval/videos,/path/to/ft_eval/videos \
        --model_names "Base,Fine-tuned" \
        --clip_dir /path/to/processed_csgo_v3/val \
        --output_dir /path/to/vis_output \
        --eval_report /path/to/ft_eval/eval_report.json

    # Custom frame selection
    python visualize_results.py \
        --gen_dirs /path/to/eval/videos \
        --model_names "Ours" \
        --clip_dir /path/to/processed_csgo_v3/val \
        --output_dir /path/to/vis_output \
        --sample_frames "0,10,30,50,70,80"
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================

VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832
VIDEO_FPS = 16
MAX_FRAMES = 81

# Drawing style
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)  # white (BGR)
LABEL_BG_COLOR = (0, 0, 0)  # black background for text
LABEL_PAD = 6


# ============================================================
# Video I/O utilities
# ============================================================

def read_video_frames(video_path, max_frames=MAX_FRAMES):
    """
    Read video frames as numpy array.
    Returns: [F, H, W, 3] uint8 array (BGR, OpenCV native).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    return np.stack(frames)


def save_video_mp4(frames, save_path, fps=VIDEO_FPS):
    """Save [F, H, W, 3] uint8 BGR frames as MP4."""
    F, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    for t in range(F):
        writer.write(frames[t])
    writer.release()


# ============================================================
# Drawing utilities
# ============================================================

def put_label(img, text, position="top-left"):
    """Draw a text label with background on the image (in-place)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    if position == "top-left":
        x, y = LABEL_PAD, LABEL_PAD + th
    elif position == "top-right":
        x, y = img.shape[1] - tw - LABEL_PAD, LABEL_PAD + th
    elif position == "bottom-left":
        x, y = LABEL_PAD, img.shape[0] - LABEL_PAD
    else:
        x, y = LABEL_PAD, LABEL_PAD + th

    # Background rectangle
    cv2.rectangle(
        img,
        (x - LABEL_PAD, y - th - LABEL_PAD),
        (x + tw + LABEL_PAD, y + baseline + LABEL_PAD),
        LABEL_BG_COLOR, -1,
    )
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS,
                cv2.LINE_AA)
    return img


def add_border(img, color=(128, 128, 128), thickness=1):
    """Add a thin border around an image (in-place)."""
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                  color, thickness)
    return img


# ============================================================
# Clip discovery
# ============================================================

def discover_clips(gen_dirs, clip_dir, num_clips=0):
    """
    Find clip names that exist in all gen_dirs and in clip_dir.
    Returns list of clip_name strings.
    """
    # Collect clip names from the first gen dir
    first_gen = gen_dirs[0]
    gen_clips = set()
    for fname in os.listdir(first_gen):
        if fname.endswith("_gen.mp4"):
            clip_name = fname.replace("_gen.mp4", "")
            gen_clips.add(clip_name)

    # Intersect with all other gen dirs
    for gd in gen_dirs[1:]:
        gd_clips = set()
        for fname in os.listdir(gd):
            if fname.endswith("_gen.mp4"):
                clip_name = fname.replace("_gen.mp4", "")
                gd_clips.add(clip_name)
        gen_clips = gen_clips & gd_clips

    # Filter to those with GT video
    valid = []
    for clip_name in sorted(gen_clips):
        gt_video = os.path.join(clip_dir, clip_name, "video.mp4")
        if os.path.exists(gt_video):
            valid.append(clip_name)
        else:
            logging.warning(f"GT video not found for {clip_name}, skipping")

    if num_clips > 0:
        valid = valid[:num_clips]

    logging.info(f"Found {len(valid)} clips with both GT and generated videos")
    return valid


def select_clips_by_metric(eval_report_path, clip_dir, gen_dirs,
                            num_clips=5, metric="psnr"):
    """
    Select best and worst clips by a metric from eval_report.json.
    Returns (best_clips, worst_clips) as lists of clip_name strings.
    """
    with open(eval_report_path) as f:
        report = json.load(f)

    per_clip = report.get("per_clip", [])

    # Filter to clips that have the metric and exist in all dirs
    scored = []
    for entry in per_clip:
        name = entry.get("clip_name", "")
        val = entry.get(metric)
        if name and val is not None and not (isinstance(val, float) and
                                              (val != val)):  # NaN check
            # Verify existence
            gt_ok = os.path.exists(os.path.join(clip_dir, name, "video.mp4"))
            gen_ok = all(
                os.path.exists(os.path.join(gd, f"{name}_gen.mp4"))
                for gd in gen_dirs
            )
            if gt_ok and gen_ok:
                scored.append((name, val))

    if not scored:
        logging.warning("No scored clips found in eval report")
        return [], []

    # Sort: higher PSNR/SSIM = better, higher LPIPS = worse
    reverse = metric in ("psnr", "ssim")
    scored.sort(key=lambda x: x[1], reverse=reverse)

    best = [name for name, _ in scored[:num_clips]]
    worst = [name for name, _ in scored[-num_clips:]]

    logging.info(f"Best {num_clips} clips by {metric}: {best}")
    logging.info(f"Worst {num_clips} clips by {metric}: {worst}")
    return best, worst


# ============================================================
# 1. Frame comparison grid
# ============================================================

def generate_frame_grid(clip_name, clip_dir, gen_dirs, model_names,
                         sample_frames, output_dir):
    """
    Generate a frame comparison grid PNG.
    Row 0: GT, Row 1+: one per model.
    Columns: one per sample frame.
    """
    gt_video_path = os.path.join(clip_dir, clip_name, "video.mp4")
    gt_frames = read_video_frames(gt_video_path)

    gen_frames_list = []
    for gd in gen_dirs:
        gen_path = os.path.join(gd, f"{clip_name}_gen.mp4")
        gen_frames_list.append(read_video_frames(gen_path))

    num_rows = 1 + len(gen_dirs)
    num_cols = len(sample_frames)

    # Get frame dimensions from first GT frame
    fh, fw = gt_frames.shape[1], gt_frames.shape[2]

    # Build grid
    grid_h = num_rows * fh
    grid_w = num_cols * fw
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for col_idx, frame_idx in enumerate(sample_frames):
        # GT row
        if frame_idx < len(gt_frames):
            cell = gt_frames[frame_idx].copy()
        else:
            cell = np.zeros((fh, fw, 3), dtype=np.uint8)
        put_label(cell, f"Frame {frame_idx}", "top-right")
        if col_idx == 0:
            put_label(cell, "GT", "top-left")
        add_border(cell)
        grid[0:fh, col_idx * fw:(col_idx + 1) * fw] = cell

        # Model rows
        for row_idx, (gen_frames, mname) in enumerate(
                zip(gen_frames_list, model_names), start=1):
            if frame_idx < len(gen_frames):
                cell = gen_frames[frame_idx].copy()
            else:
                cell = np.zeros((fh, fw, 3), dtype=np.uint8)
            if col_idx == 0:
                put_label(cell, mname, "top-left")
            add_border(cell)
            grid[row_idx * fh:(row_idx + 1) * fh,
                 col_idx * fw:(col_idx + 1) * fw] = cell

    save_path = os.path.join(output_dir, f"comparison_{clip_name}.png")
    cv2.imwrite(save_path, grid)
    logging.info(f"  Saved frame grid: {save_path}")
    return save_path


# ============================================================
# 2. Side-by-side video
# ============================================================

def generate_sidebyside_video(clip_name, clip_dir, gen_dirs, model_names,
                               output_dir):
    """
    Generate a side-by-side comparison video.
    2-way: [GT | Model A]
    3-way: [GT | Model A | Model B]
    Each panel has a text label overlay.
    """
    gt_video_path = os.path.join(clip_dir, clip_name, "video.mp4")
    gt_frames = read_video_frames(gt_video_path)

    gen_frames_list = []
    for gd in gen_dirs:
        gen_path = os.path.join(gd, f"{clip_name}_gen.mp4")
        gen_frames_list.append(read_video_frames(gen_path))

    # Align frame counts
    min_f = min(len(gt_frames), *(len(gf) for gf in gen_frames_list))
    gt_frames = gt_frames[:min_f]
    gen_frames_list = [gf[:min_f] for gf in gen_frames_list]

    num_panels = 1 + len(gen_dirs)
    fh, fw = gt_frames.shape[1], gt_frames.shape[2]

    # Separator width
    sep_w = 4
    total_w = num_panels * fw + (num_panels - 1) * sep_w

    # Build composite frames
    composite = np.zeros((min_f, fh, total_w, 3), dtype=np.uint8)

    for t in range(min_f):
        x_offset = 0

        # GT panel
        panel = gt_frames[t].copy()
        if t < VIDEO_FPS:  # show label for first second
            put_label(panel, "GT", "top-left")
        composite[t, :, x_offset:x_offset + fw] = panel
        x_offset += fw

        # Separator
        composite[t, :, x_offset:x_offset + sep_w] = 255  # white line
        x_offset += sep_w

        # Model panels
        for i, (gen_frames, mname) in enumerate(
                zip(gen_frames_list, model_names)):
            panel = gen_frames[t].copy()
            if t < VIDEO_FPS:
                put_label(panel, mname, "top-left")
            composite[t, :, x_offset:x_offset + fw] = panel
            x_offset += fw

            if i < len(gen_frames_list) - 1:
                composite[t, :, x_offset:x_offset + sep_w] = 255
                x_offset += sep_w

    save_path = os.path.join(output_dir, f"sidebyside_{clip_name}.mp4")
    save_video_mp4(composite, save_path, VIDEO_FPS)
    logging.info(f"  Saved side-by-side video: {save_path}")
    return save_path


# ============================================================
# 3. Summary mosaic
# ============================================================

def generate_mosaic(clip_names, clip_dir, gen_dirs, model_names,
                     output_dir, mosaic_name="mosaic"):
    """
    Generate a mosaic PNG showing first-frame comparisons for multiple clips.
    Layout: rows = clips, columns = [GT, Model A, Model B, ...]
    """
    if not clip_names:
        logging.warning(f"No clips for mosaic '{mosaic_name}', skipping")
        return None

    num_models = len(gen_dirs)
    num_cols = 1 + num_models  # GT + models
    num_rows = len(clip_names)

    # Read first frame from each to get dimensions
    sample_gt = os.path.join(clip_dir, clip_names[0], "video.mp4")
    sample_frame = read_video_frames(sample_gt, max_frames=1)[0]
    fh, fw = sample_frame.shape[:2]

    # Scale down for mosaic readability (half resolution)
    scale = 0.5
    th, tw = int(fh * scale), int(fw * scale)

    grid_h = num_rows * th
    grid_w = num_cols * tw
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for row_idx, clip_name in enumerate(clip_names):
        y0 = row_idx * th

        # GT first frame
        gt_video_path = os.path.join(clip_dir, clip_name, "video.mp4")
        gt_frame = read_video_frames(gt_video_path, max_frames=1)[0]
        gt_thumb = cv2.resize(gt_frame, (tw, th), interpolation=cv2.INTER_AREA)
        label = f"GT / {clip_name}" if row_idx == 0 else clip_name
        put_label(gt_thumb, label, "top-left")
        add_border(gt_thumb)
        grid[y0:y0 + th, 0:tw] = gt_thumb

        # Model first frames
        for col_idx, (gd, mname) in enumerate(zip(gen_dirs, model_names),
                                                start=1):
            gen_path = os.path.join(gd, f"{clip_name}_gen.mp4")
            if os.path.exists(gen_path):
                gen_frame = read_video_frames(gen_path, max_frames=1)[0]
                gen_thumb = cv2.resize(gen_frame, (tw, th),
                                       interpolation=cv2.INTER_AREA)
            else:
                gen_thumb = np.zeros((th, tw, 3), dtype=np.uint8)
            if row_idx == 0:
                put_label(gen_thumb, mname, "top-left")
            add_border(gen_thumb)
            grid[y0:y0 + th, col_idx * tw:(col_idx + 1) * tw] = gen_thumb

    save_path = os.path.join(output_dir, f"{mosaic_name}.png")
    cv2.imwrite(save_path, grid)
    logging.info(f"Saved mosaic: {save_path}")
    return save_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative Visualization for CSGO LingBot-World"
    )
    parser.add_argument("--gen_dirs", type=str, required=True,
                        help="Comma-separated paths to generated video directories "
                             "(one per model)")
    parser.add_argument("--model_names", type=str, required=True,
                        help="Comma-separated display names for each model "
                             "(e.g. 'Base,Epoch3,Final')")
    parser.add_argument("--clip_dir", type=str, required=True,
                        help="GT clips directory (each clip is a subdirectory "
                             "with video.mp4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualizations")
    parser.add_argument("--eval_report", type=str, default=None,
                        help="Path to eval_report.json for best/worst selection")
    parser.add_argument("--num_clips", type=int, default=5,
                        help="Number of clips to visualize (default: 5)")
    parser.add_argument("--sample_frames", type=str, default="0,20,40,60,80",
                        help="Comma-separated frame indices for grid "
                             "(default: '0,20,40,60,80')")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    # ---- Parse arguments ----
    gen_dirs = [p.strip() for p in args.gen_dirs.split(",")]
    model_names = [n.strip() for n in args.model_names.split(",")]
    sample_frames = [int(x.strip()) for x in args.sample_frames.split(",")]

    if len(gen_dirs) != len(model_names):
        raise ValueError(
            f"Number of gen_dirs ({len(gen_dirs)}) must match "
            f"model_names ({len(model_names)})"
        )

    for gd in gen_dirs:
        if not os.path.isdir(gd):
            raise FileNotFoundError(f"Generated video directory not found: {gd}")
    if not os.path.isdir(args.clip_dir):
        raise FileNotFoundError(f"Clip directory not found: {args.clip_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Models: {model_names}")
    logging.info(f"Gen dirs: {gen_dirs}")
    logging.info(f"Clip dir: {args.clip_dir}")
    logging.info(f"Sample frames: {sample_frames}")

    # ---- Discover clips ----
    all_clips = discover_clips(gen_dirs, args.clip_dir, num_clips=0)

    if not all_clips:
        logging.error("No valid clips found. Check paths and file naming.")
        return

    # Select clips for visualization
    vis_clips = all_clips[:args.num_clips]

    # ---- 1. Frame comparison grids ----
    logging.info("=" * 60)
    logging.info("Generating frame comparison grids...")
    logging.info("=" * 60)

    grids_dir = os.path.join(args.output_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)

    for clip_name in tqdm(vis_clips, desc="Frame grids"):
        try:
            generate_frame_grid(
                clip_name, args.clip_dir, gen_dirs, model_names,
                sample_frames, grids_dir,
            )
        except Exception as e:
            logging.error(f"Failed to generate grid for {clip_name}: {e}")

    # ---- 2. Side-by-side videos ----
    logging.info("=" * 60)
    logging.info("Generating side-by-side videos...")
    logging.info("=" * 60)

    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for clip_name in tqdm(vis_clips, desc="Side-by-side"):
        try:
            generate_sidebyside_video(
                clip_name, args.clip_dir, gen_dirs, model_names, videos_dir,
            )
        except Exception as e:
            logging.error(f"Failed to generate video for {clip_name}: {e}")

    # ---- 3. Summary mosaics ----
    logging.info("=" * 60)
    logging.info("Generating summary mosaics...")
    logging.info("=" * 60)

    if args.eval_report and os.path.exists(args.eval_report):
        best_clips, worst_clips = select_clips_by_metric(
            args.eval_report, args.clip_dir, gen_dirs,
            num_clips=args.num_clips, metric="psnr",
        )

        if best_clips:
            generate_mosaic(
                best_clips, args.clip_dir, gen_dirs, model_names,
                args.output_dir, mosaic_name="mosaic_best",
            )
        if worst_clips:
            generate_mosaic(
                worst_clips, args.clip_dir, gen_dirs, model_names,
                args.output_dir, mosaic_name="mosaic_worst",
            )
    else:
        if args.eval_report:
            logging.warning(f"Eval report not found: {args.eval_report}")
        logging.info("No eval report provided, generating mosaic from first "
                     f"{args.num_clips} clips")
        generate_mosaic(
            vis_clips, args.clip_dir, gen_dirs, model_names,
            args.output_dir, mosaic_name="mosaic_sample",
        )

    # ---- Summary ----
    logging.info("")
    logging.info("=" * 60)
    logging.info("VISUALIZATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"  Frame grids:       {grids_dir}/")
    logging.info(f"  Side-by-side:      {videos_dir}/")
    logging.info(f"  Mosaics:           {args.output_dir}/mosaic_*.png")
    logging.info(f"Clips visualized: {len(vis_clips)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
