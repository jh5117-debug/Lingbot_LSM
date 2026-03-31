"""
infer.py — 带 Surprise-Driven Memory 的长视频推理脚本

在不修改 lingbot-world 原始代码的前提下，通过替换模型实现 Memory 注入：
1. 加载标准 WanI2V 管道
2. 将 low/high_noise_model 替换为 WanModelWithMemory
3. 按 clip 逐段生成，维护跨 clip 的 MemoryBank
4. 每段生成后，用帧间 VAE latent cosine distance 计算 Surprise，更新 MemoryBank

Surprise score 说明：
  - 训练时用 NFPHead loss 监督（见 nfp_head.py）
  - 推理时用帧间 VAE latent cosine distance 作为代理指标：
    surprise[t] = 1 - cosine_sim(latent_t.flatten(), latent_{t-1}.flatten())
  - 值越大 = 场景变化越大 = 越值得存入记忆

用法：
  # 单 clip（等价于原始 generate.py，验证接口）：
  python src/pipeline/infer.py \\
    --ckpt_dir /path/to/lingbot-world \\
    --image first_frame.jpg --action_path ./poses --prompt "..." \\
    --output_path ./output.mp4

  # 长视频多 clip：
  python src/pipeline/infer.py \\
    --ckpt_dir /path/to/lingbot-world \\
    --image first_frame.jpg --action_path ./poses --prompt "..." \\
    --num_clips 5 --clip_stride 40 \\
    --memory_size 8 --memory_top_k 4 \\
    --output_path ./output_long.mp4

  # 快速验证流程（2步）：
  python src/pipeline/infer.py --dry_run --ckpt_dir /path/to/lingbot-world \\
    --image first_frame.jpg --action_path ./poses --prompt "..."
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---- 引入 lingbot-world ----
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))          # → src/pipeline/
_SRC_DIR = os.path.dirname(_PIPELINE_DIR)                           # → src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                           # → Lingbot_LSM/
_LINGBOT_WORLD = os.path.join(_PROJECT_ROOT, 'refs', 'lingbot-world')
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS
from wan.image2video import WanI2V
from wan.utils.utils import save_video

# ---- 引入 memory 模块 ----
_MODULE_DIR = _SRC_DIR  # infer.py 在 src/pipeline/ 下，memory_module 在 src/ 下
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from memory_module.memory_bank import MemoryBank
from memory_module.model_with_memory import WanModelWithMemory


# ---------------------------------------------------------------------------
# 管道转换：WanI2V → WanI2VWithMemory
# ---------------------------------------------------------------------------

def convert_pipeline_to_memory(
    pipeline: WanI2V,
    memory_layers: Optional[List[int]] = None,
    max_memory_size: int = 8,
) -> WanI2V:
    """将已加载的 WanI2V 管道的两个 DiT 模型原地替换为 WanModelWithMemory。

    Args:
        pipeline:        已初始化的 WanI2V 实例
        memory_layers:   要插入 Memory Cross-Attention 的 block 索引，None = 全部
        max_memory_size: Memory Bank 容量 K

    Returns:
        原 pipeline 对象（in-place 修改，low/high_noise_model 已替换）
    """
    logger.info("Converting low_noise_model to WanModelWithMemory...")
    pipeline.low_noise_model = WanModelWithMemory.from_wan_model(
        pipeline.low_noise_model,
        memory_layers=memory_layers,
        max_memory_size=max_memory_size,
    )

    logger.info("Converting high_noise_model to WanModelWithMemory...")
    pipeline.high_noise_model = WanModelWithMemory.from_wan_model(
        pipeline.high_noise_model,
        memory_layers=memory_layers,
        max_memory_size=max_memory_size,
    )

    logger.info("Pipeline conversion done.")
    return pipeline


# ---------------------------------------------------------------------------
# Surprise score 计算（推理时的代理指标）
# ---------------------------------------------------------------------------

def compute_frame_surprises(latent: torch.Tensor) -> List[float]:
    """用帧间 VAE latent cosine distance 计算每帧的 Surprise score。

    Args:
        latent: [z_dim, lat_f, h, w]

    Returns:
        surprises: List[float]，长度 lat_f，值域 [0, 2]
    """
    surprises = []
    for t in range(latent.shape[1]):
        if t == 0:
            surprises.append(1.0)  # 第一帧无前帧参照，默认高 surprise
        else:
            prev = latent[:, t - 1].flatten().float()
            curr = latent[:, t].flatten().float()
            cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
            surprises.append(float(1.0 - cos_sim))
    return surprises


# ---------------------------------------------------------------------------
# 核心推理函数：单 clip 生成 + Memory Bank 更新
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_clip_with_memory(
    pipeline: WanI2V,
    prompt: str,
    first_frame: Image.Image,
    action_path: str,
    memory_bank: MemoryBank,
    memory_top_k: int = 4,
    clip_start_frame: int = 0,
    frame_num: int = 81,
    sampling_steps: int = 40,
    guide_scale: float = 5.0,
    sample_solver: str = 'unipc',
    shift: float = 5.0,
    seed: int = 42,
    max_area: int = 720 * 1280,
    offload_model: bool = True,
) -> torch.Tensor:
    """生成一个 clip，注入 MemoryBank 检索到的历史帧，并在生成后更新 MemoryBank。

    Args:
        pipeline:        已转换为 WanModelWithMemory 的 WanI2V 管道
        prompt:          文本提示
        first_frame:     本 clip 的起始帧（PIL Image）
        action_path:     包含 poses.npy / intrinsics.npy 的目录
        memory_bank:     当前 MemoryBank 实例（跨 clip 共享）
        memory_top_k:    每次检索的记忆帧数
        clip_start_frame: 本 clip 在完整视频中的起始帧索引（用于 timestep 记录）
        frame_num:       本 clip 帧数（4n+1）
        其余参数:        对应 WanI2V.generate() 的同名参数

    Returns:
        video: [3, frame_num, H, W]，值域 [-1, 1]
    """
    cfg = pipeline.config
    device = pipeline.device

    # ---- Step 1：计算 camera pose 嵌入（用于 MemoryBank 检索） ----
    # 先通过 pipeline._prepare_camera_emb 得到 c2ws_plucker_emb，
    # 再用 WanModelWithMemory.get_projected_frame_embs 投影到模型空间

    # 重用 pipeline 内部的 camera 处理逻辑（读取 poses/intrinsics）
    # 由于 WanI2V.generate() 不暴露中间值，我们自己计算 plucker embedding
    c2ws_plucker_emb_raw = _compute_plucker_emb(
        action_path=action_path,
        frame_num=frame_num,
        control_type=pipeline.control_type,
        max_area=max_area,
        vae_stride=pipeline.vae_stride,
        patch_size=pipeline.patch_size,
        device=device,
        param_dtype=cfg.param_dtype,
    )  # [1, C, lat_f, lat_h, lat_w] or None

    # ---- Step 2：检索 MemoryBank ----
    memory_states = None
    if c2ws_plucker_emb_raw is not None and memory_bank.size() > 0:
        # 投影到模型空间，取中间帧作为检索 query
        low_noise_model = pipeline.low_noise_model
        if isinstance(low_noise_model, WanModelWithMemory):
            frame_embs = low_noise_model.get_projected_frame_embs(
                c2ws_plucker_emb_raw.to(device)
            )  # [lat_f, dim]
            query_emb = frame_embs[len(frame_embs) // 2]  # 取中间帧 [dim]
            retrieved = memory_bank.retrieve(query_emb, top_k=memory_top_k, device=device)
            if retrieved is not None:
                memory_states = retrieved.unsqueeze(0)  # [1, K, dim]
                logger.info(
                    "clip_start=%d: retrieved %d memory frames",
                    clip_start_frame, retrieved.shape[0],
                )

    # ---- Step 3：调用 generate()，注入 memory_states ----
    # WanI2V.generate() 不接受 memory_states，通过 monkey-patch 注入
    # 将 memory_states 附到 dit_cond_dict 中，MemoryBlockWrapper 会从中读取
    _patch_pipeline_memory(pipeline, memory_states)
    try:
        video = pipeline.generate(
            input_prompt=prompt,
            img=first_frame,
            action_path=action_path,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model,
        )
    finally:
        _unpatch_pipeline_memory(pipeline)

    if video is None:
        return None  # 非 rank 0 进程

    # ---- Step 4：更新 MemoryBank ----
    if c2ws_plucker_emb_raw is not None:
        _update_memory_bank(
            bank=memory_bank,
            video=video,
            c2ws_plucker_emb_raw=c2ws_plucker_emb_raw,
            pipeline=pipeline,
            device=device,
            clip_start_frame=clip_start_frame,
        )

    return video


# ---------------------------------------------------------------------------
# 长视频推理主函数
# ---------------------------------------------------------------------------

def generate_long_video(
    pipeline: WanI2V,
    prompt: str,
    first_frame: Image.Image,
    action_path: str,
    num_clips: int = 1,
    clip_stride: int = 40,
    frame_num: int = 81,
    memory_size: int = 8,
    memory_top_k: int = 4,
    sampling_steps: int = 40,
    guide_scale: float = 5.0,
    sample_solver: str = 'unipc',
    shift: float = 5.0,
    seed: int = 42,
    max_area: int = 720 * 1280,
    offload_model: bool = True,
) -> List[torch.Tensor]:
    """逐 clip 生成长视频，跨 clip 维护 MemoryBank。

    Args:
        num_clips:    总 clip 数
        clip_stride:  每个 clip 起始帧相对前一 clip 的偏移帧数（视频帧，非 latent 帧）
        frame_num:    每个 clip 的帧数（4n+1）
        memory_size:  MemoryBank 容量 K
        memory_top_k: 每次检索的记忆帧数

    Returns:
        clips: 每个 clip 的视频张量 List[Tensor [3, frame_num, H, W]]
    """
    bank = MemoryBank(max_size=memory_size)
    clips = []

    current_first_frame = first_frame

    for clip_idx in range(num_clips):
        clip_start = clip_idx * clip_stride
        logger.info(
            "Generating clip %d/%d (start_frame=%d)...",
            clip_idx + 1, num_clips, clip_start,
        )

        video = generate_clip_with_memory(
            pipeline=pipeline,
            prompt=prompt,
            first_frame=current_first_frame,
            action_path=action_path,
            memory_bank=bank,
            memory_top_k=memory_top_k,
            clip_start_frame=clip_start,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            sample_solver=sample_solver,
            shift=shift,
            seed=seed + clip_idx,  # 每 clip 不同 seed
            max_area=max_area,
            offload_model=offload_model,
        )

        if video is not None:
            clips.append(video)
            logger.info(
                "Clip %d done. MemoryBank: %s", clip_idx + 1, bank
            )

            # 下一 clip 的起始帧 = 当前 clip 的最后一帧
            last_frame_np = ((video[:, -1].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                             ).clip(0, 255).astype(np.uint8)
            current_first_frame = Image.fromarray(last_frame_np)

    return clips


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _compute_plucker_emb(
    action_path, frame_num, control_type, max_area,
    vae_stride, patch_size, device, param_dtype
):
    """复现 WanI2V.generate() 中的 camera embedding 处理逻辑。返回 [1, C, lat_f, lat_h, lat_w]。"""
    if action_path is None:
        return None

    from wan.utils.cam_utils import (
        compute_relative_poses, interpolate_camera_poses,
        get_plucker_embeddings, get_Ks_transformed,
    )
    from einops import rearrange as ein_rearrange

    c2ws = np.load(os.path.join(action_path, "poses.npy"))
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]

    Ks_raw = np.load(os.path.join(action_path, "intrinsics.npy"))
    Ks_t = torch.from_numpy(Ks_raw).float()

    # 复用 WanI2V 中的分辨率计算（近似：假设 480×832 输入）
    h_org, w_org = 480, 832
    aspect_ratio = h_org / w_org
    import math
    lat_h = round(math.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(math.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    lat_f = (frame_num - 1) // vae_stride[0] + 1

    Ks = get_Ks_transformed(Ks_t, height_org=h_org, width_org=w_org,
                             height_resize=h, width_resize=w,
                             height_final=h, width_final=w)[0]

    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, (len(c2ws) - 1) // 4 + 1),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)

    only_rays_d = (control_type == 'act')
    plucker = get_plucker_embeddings(
        c2ws_infer.to(device), Ks.to(device), h, w, only_rays_d=only_rays_d
    )

    plucker = ein_rearrange(plucker,
                            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                            c1=int(h // lat_h), c2=int(w // lat_w))
    plucker = plucker[None]
    plucker = ein_rearrange(plucker, 'b (f h w) c -> b c f h w',
                            f=lat_f, h=lat_h, w=lat_w)

    if control_type == 'act':
        wasd = np.load(os.path.join(action_path, "action.npy"))[:frame_num]
        wasd_t = torch.from_numpy(wasd[::4]).float().to(device)
        wasd_t = wasd_t[:, None, None, :].repeat(1, h, w, 1)
        wasd_t = ein_rearrange(wasd_t, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                               c1=int(h // lat_h), c2=int(w // lat_w))
        wasd_t = ein_rearrange(wasd_t[None], 'b (f h w) c -> b c f h w',
                               f=lat_f, h=lat_h, w=lat_w)
        plucker = torch.cat([plucker, wasd_t], dim=1)

    return plucker.to(param_dtype)


def _patch_pipeline_memory(pipeline: WanI2V, memory_states: Optional[torch.Tensor]):
    """将 memory_states 注入到 pipeline 的两个模型的 forward 中。

    FIX[B-01]: 使用 functools.partial 而非 types.MethodType，避免 self 双重绑定问题。
    """
    if memory_states is None:
        return

    import functools

    for name in ('low_noise_model', 'high_noise_model'):
        model = getattr(pipeline, name)
        if isinstance(model, WanModelWithMemory):
            model._original_forward = model.forward

            def _make_patched(m, mem):
                @functools.wraps(m.forward)
                def _patched(x, t, context, seq_len, y=None, dit_cond_dict=None):
                    return WanModelWithMemory.forward(
                        m, x, t, context, seq_len,
                        y=y, dit_cond_dict=dit_cond_dict,
                        memory_states=mem.to(next(m.parameters()).device),
                    )
                return _patched

            model.forward = _make_patched(model, memory_states)


def _unpatch_pipeline_memory(pipeline: WanI2V):
    """还原 patch。"""
    for name in ('low_noise_model', 'high_noise_model'):
        model = getattr(pipeline, name)
        if hasattr(model, '_original_forward'):
            model.forward = model._original_forward
            del model._original_forward


def _update_memory_bank(
    bank: MemoryBank,
    video: torch.Tensor,
    c2ws_plucker_emb_raw: torch.Tensor,
    pipeline: WanI2V,
    device: torch.device,
    clip_start_frame: int,
):
    """生成完成后，用当前 clip 的 latent 和 pose_emb 更新 MemoryBank。"""
    # FIX[B-02]: offload_model=True 时 VAE 可能已在 CPU，需先移回 device 再 encode
    vae_device = next(pipeline.vae.model.parameters()).device
    if vae_device != device:
        pipeline.vae.model.to(device)
    latent = pipeline.vae.encode([video.to(device)])[0]  # [z_dim, lat_f, h, w]

    # 每帧的 Surprise score
    surprises = compute_frame_surprises(latent)

    # 每帧的 pose embedding（在模型空间）
    low_noise_model = pipeline.low_noise_model
    if isinstance(low_noise_model, WanModelWithMemory):
        frame_embs = low_noise_model.get_projected_frame_embs(
            c2ws_plucker_emb_raw.to(device)
        )  # [lat_f, dim]
    else:
        return  # 没有转换成功，跳过

    lat_f = latent.shape[1]
    vae_stride_t = pipeline.vae_stride[0]

    for t in range(lat_f):
        bank.update(
            pose_emb=frame_embs[t].cpu(),
            latent=latent[:, t].cpu(),
            surprise_score=surprises[t],
            timestep=clip_start_frame + t * vae_stride_t,
        )

    logger.info(
        "MemoryBank updated: %s", bank
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Inference with Surprise-Driven Memory")

    # 路径
    p.add_argument("--ckpt_dir", required=True, help="lingbot-world checkpoint 目录")
    p.add_argument("--image", required=True, help="起始帧图像路径")
    p.add_argument("--action_path", default=None, help="camera poses 目录（含 poses.npy / intrinsics.npy）")
    p.add_argument("--output_path", default=None, help="输出视频路径（默认自动生成）")

    # 生成参数
    p.add_argument("--prompt", default="", help="文本提示")
    p.add_argument("--num_clips", type=int, default=1, help="总 clip 数（默认 1，等价于单段生成）")
    p.add_argument("--clip_stride", type=int, default=40, help="相邻 clip 起始帧偏移（视频帧数）")
    p.add_argument("--frame_num", type=int, default=81, help="每 clip 帧数（4n+1）")
    p.add_argument("--sampling_steps", type=int, default=40)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=str, default="1280*720")
    p.add_argument("--task", type=str, default="i2v-A14B")

    # Memory 参数
    p.add_argument("--memory_size", type=int, default=8, help="Memory Bank 容量 K")
    p.add_argument("--memory_top_k", type=int, default=4, help="每次检索的记忆帧数")
    p.add_argument("--memory_layers", type=int, nargs="*", default=None,
                   help="插入 Memory Cross-Attention 的 block 索引，默认为全部")

    # 运行参数
    p.add_argument("--offload_model", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true", default=False,
                   help="只跑 2 步验证流程，不保存完整视频")

    return p.parse_args()


def main():
    args = _parse_args()

    cfg = WAN_CONFIGS[args.task]
    max_area = MAX_AREA_CONFIGS.get(args.size, 720 * 1280)

    if args.dry_run:
        logger.info("DRY RUN: sampling_steps forced to 2, num_clips to 1")
        args.sampling_steps = 2
        args.num_clips = 1

    # ---- 加载 WanI2V 管道 ----
    logger.info("Loading WanI2V pipeline from %s ...", args.ckpt_dir)
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        offload_model=args.offload_model,
    )

    # ---- 转换为 WanModelWithMemory ----
    pipeline = convert_pipeline_to_memory(
        pipeline,
        memory_layers=args.memory_layers,
        max_memory_size=args.memory_size,
    )

    # ---- 加载起始帧 ----
    first_frame = Image.open(args.image).convert("RGB")
    logger.info("First frame: %s", args.image)

    # ---- 生成 ----
    clips = generate_long_video(
        pipeline=pipeline,
        prompt=args.prompt,
        first_frame=first_frame,
        action_path=args.action_path,
        num_clips=args.num_clips,
        clip_stride=args.clip_stride,
        frame_num=args.frame_num,
        memory_size=args.memory_size,
        memory_top_k=args.memory_top_k,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        sample_solver=args.sample_solver,
        shift=args.shift,
        seed=args.seed,
        max_area=max_area,
        offload_model=args.offload_model,
    )

    if not clips:
        logger.info("No clips generated (non-rank-0 process).")
        return

    # ---- 拼接并保存 ----
    if args.output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"memory_gen_{args.num_clips}clips_{ts}.mp4"

    if len(clips) == 1:
        full_video = clips[0]
    else:
        # 简单拼接（不做 overlap blending，后续可改进）
        full_video = torch.cat(clips, dim=1)  # [3, total_frames, H, W]

    save_video(
        tensor=full_video[None],
        save_file=args.output_path,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info("Saved to %s (frames: %d)", args.output_path, full_video.shape[1])


if __name__ == "__main__":
    main()
