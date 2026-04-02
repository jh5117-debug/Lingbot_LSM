"""
infer_v2.py — CSGO 推理脚本，完全对齐 csgo-finetune-v2/inference_csgo.py，
             在其基础上支持 WanModelWithMemory + Memory Bank。

与 inference_csgo.py 的关系：
  - 保留所有原有参数与行为（LoRA 加载、全参微调加载、WanI2V 推理）
  - 新增 --use_memory / --memory_max_size，控制是否启用 Memory Bank
  - 当 --use_memory False 时，与 inference_csgo.py 行为完全一致

用法（无 memory，等价于 inference_csgo.py）：
    torchrun --nproc_per_node=8 infer_v2.py \\
        --ckpt_dir /path/to/lingbot-world-base-act/ \\
        --lora_path /path/to/lora_weights.pth \\
        --image /path/to/image.jpg \\
        --action_path /path/to/clip/ \\
        --prompt "First-person view of CS:GO gameplay" \\
        --size 480*832 --frame_num 81

用法（启用 memory）：
    torchrun --nproc_per_node=8 infer_v2.py \\
        --ckpt_dir /path/to/lingbot-world-base-act/ \\
        --lora_path /path/to/lora_weights.pth \\
        --image /path/to/image.jpg \\
        --action_path /path/to/clip/ \\
        --prompt "First-person view of CS:GO gameplay" \\
        --size 480*832 --frame_num 81 \\
        --use_memory --memory_max_size 50
"""

import argparse
import logging
import os
import sys
from os.path import abspath, dirname, join

import torch

# ---------------------------------------------------------------------------
# sys.path 设置
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(abspath(__file__))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, 'refs', 'lingbot-world')

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CSGO 推理脚本（完全对齐 inference_csgo.py + Memory Bank 扩展）"
    )

    # ---- v2 原有参数（与 inference_csgo.py 完全一致）----
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="基础模型目录（lingbot-world checkpoint）")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重 .pth 文件路径（可选）")
    parser.add_argument("--ft_model_dir", type=str, default=None,
                        help="全参微调 low_noise_model 目录（可选，与 --lora_path 互斥）")
    parser.add_argument("--image", type=str, required=True,
                        help="初始帧图像路径")
    parser.add_argument("--action_path", type=str, default=None,
                        help="动作数据路径（action.npy 或含 poses.npy 的目录）")
    parser.add_argument("--prompt", type=str,
                        default="First-person view of CS:GO competitive gameplay",
                        help="文本描述")
    parser.add_argument("--save_file", type=str, default="output_csgo.mp4",
                        help="输出视频路径")
    parser.add_argument("--size", type=str, default="480*832",
                        help="分辨率，如 '480*832'")
    parser.add_argument("--frame_num", type=int, default=81,
                        help="帧数（默认 81）")
    parser.add_argument("--sample_steps", type=int, default=70,
                        help="采样步数（默认 70）")
    parser.add_argument("--sample_shift", type=float, default=10.0,
                        help="sigma shift（默认 10.0）")
    parser.add_argument("--guide_scale", type=float, default=5.0,
                        help="CFG scale（默认 5.0）")
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)

    # ---- Memory Enhancement 参数（新增）----
    parser.add_argument("--use_memory", action="store_true", default=False,
                        help="Enable Memory Bank for cross-chunk inference")
    parser.add_argument("--memory_max_size", type=int, default=50,
                        help="Memory Bank max capacity (default 50)")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Number of chunks for episode generation (default 1 = no chunking)")
    parser.add_argument("--memory_top_k", type=int, default=4,
                        help="Top-K frames to retrieve from Memory Bank")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# LoRA 加载（与 inference_csgo.py 完全一致）
# ---------------------------------------------------------------------------

def _load_lora_and_prepare_ckpt(args) -> str:
    """加载 LoRA 权重并合并，返回合并后的临时 ckpt_dir。

    流程与 inference_csgo.py 完全一致：
      1. 加载 WanModel（control_type='act'）
      2. 从 lora_state_dict 自动检测 target_modules 和 lora_rank
      3. inject_adapter_in_model(LoraConfig(...), model)
      4. 键名映射：lora_A.weight → lora_A.default.weight
      5. 合并 LoRA 权重：module.merge()
      6. model.save_pretrained(tmp_dir/low_noise_model)
      7. 符号链接其他文件（high_noise_model, VAE, T5 等）
      8. 返回 tmp_dir
    """
    logger.info("Loading base model + LoRA weights for inference...")

    from wan.modules.model import WanModel
    from peft import LoraConfig, inject_adapter_in_model

    # Step 1：加载 base low_noise_model（control_type='act'，control_dim=7）
    model = WanModel.from_pretrained(
        args.ckpt_dir, subfolder="low_noise_model",
        torch_dtype=torch.bfloat16, control_type="act",
    )

    # Step 2：从 lora_state_dict 自动检测 target_modules 和 lora_rank
    lora_state = torch.load(args.lora_path, map_location="cpu")
    target_modules = set()
    for key in lora_state.keys():
        # 提取模块名，例如 "blocks.0.self_attn.q.lora_A.default.weight" → "blocks.0.self_attn.q"
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ("lora_A", "lora_B"):
                module_name = ".".join(parts[:i])
                target_modules.add(module_name)
                break

    target_modules = sorted(list(target_modules))
    logger.info("Detected %d LoRA target modules", len(target_modules))

    # 自动检测 lora_rank（从 lora_A.shape[0]）
    lora_rank = None
    for key, val in lora_state.items():
        if "lora_A" in key:
            lora_rank = val.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Cannot detect lora_rank from lora_state_dict — no lora_A key found.")
    logger.info("Detected lora_rank=%d", lora_rank)

    # Step 3：inject_adapter_in_model
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)

    # Step 4：键名映射（lora_A.weight → lora_A.default.weight）
    mapped_state = {}
    for key, val in lora_state.items():
        if "lora_A.weight" in key and "default" not in key:
            key = key.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in key and "default" not in key:
            key = key.replace("lora_B.weight", "lora_B.default.weight")
        mapped_state[key] = val

    result = model.load_state_dict(mapped_state, strict=False)
    logger.info(
        "Loaded LoRA weights: %d keys, missing=%d, unexpected=%d",
        len(mapped_state), len(result.missing_keys), len(result.unexpected_keys),
    )

    # Step 5：合并 LoRA 权重（module.merge()）
    import peft.tuners.lora as lora_module
    for _name, _module in model.named_modules():
        if isinstance(_module, lora_module.Linear):
            _module.merge()

    # Step 6：保存合并后的模型到临时目录
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='act_')
    merged_ckpt = os.path.join(tmp_dir, "low_noise_model")
    model.save_pretrained(merged_ckpt)
    logger.info("Saved merged model to %s", merged_ckpt)

    # Step 7：符号链接其他文件
    for item in ["high_noise_model", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                 "google", "configuration.json"]:
        src = os.path.join(args.ckpt_dir, item)
        dst = os.path.join(tmp_dir, item)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    del model, lora_state, mapped_state
    torch.cuda.empty_cache()

    return tmp_dir


def _load_ft_model_and_prepare_ckpt(args) -> str:
    """全参微调模型：符号链接 ft_model_dir → tmp_dir/low_noise_model，返回 tmp_dir。

    与 inference_csgo.py 完全一致。
    """
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='act_')
    os.symlink(args.ft_model_dir, os.path.join(tmp_dir, "low_noise_model"))
    for item in ["high_noise_model", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                 "google", "configuration.json"]:
        src = os.path.join(args.ckpt_dir, item)
        dst = os.path.join(tmp_dir, item)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    return tmp_dir


# ---------------------------------------------------------------------------
# Memory 辅助（复用 infer.py 中的逻辑）
# ---------------------------------------------------------------------------

def _convert_pipeline_to_memory(pipeline, memory_max_size: int):
    """将 WanI2V 管道的 low_noise_model 替换为 WanModelWithMemory。

    注意：只转换 low_noise_model，不转换 high_noise_model。
    原因：high_noise_model 的 forward 签名可能与 low_noise_model 不同，
    且 memory 模块只在低噪声侧有意义（高噪声侧负责全局结构，
    不需要细粒度的历史帧检索）。high_noise_model 保持原始 WanModel 不变。
    """
    from memory_module.model_with_memory import WanModelWithMemory

    logger.info("Converting low_noise_model to WanModelWithMemory (max_size=%d)...", memory_max_size)
    pipeline.low_noise_model = WanModelWithMemory.from_wan_model(
        pipeline.low_noise_model,
        memory_layers=None,          # 全部 blocks（与 infer.py 默认行为一致）
        max_memory_size=memory_max_size,
    )
    # high_noise_model 保持原始 WanModel，不做转换（见函数注释）

    logger.info("Pipeline conversion to WanModelWithMemory done.")
    return pipeline


def _patch_pipeline_memory(pipeline, memory_states):
    """将 memory_states 通过 monkey-patch 注入到 pipeline low_noise_model 的 forward 中。

    注意：只 patch low_noise_model，不 patch high_noise_model。
    high_noise_model 保持原始 WanModel（未转换为 WanModelWithMemory），
    其 forward 签名与 WanModelWithMemory 不同，不应注入 memory。
    复用 infer.py 的 _patch_pipeline_memory 逻辑（FIX[B-01]：使用 functools.partial）。
    """
    if memory_states is None:
        return

    import functools
    from memory_module.model_with_memory import WanModelWithMemory

    model = pipeline.low_noise_model
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


def _unpatch_pipeline_memory(pipeline):
    """还原 _patch_pipeline_memory 的 monkey-patch（仅 low_noise_model）。"""
    model = pipeline.low_noise_model
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        del model._original_forward


def _compute_chunk_surprises_v2(pipeline, frame_embs, video_latent):
    """Compute per-frame surprise for a generated chunk.

    In inference we do not have teacher-forced targets or a training-time batch,
    but we can still reuse NFPHead's per-frame next-frame objective by feeding a
    frame-level proxy sequence and comparing against the generated chunk latent.
    This keeps inference-time write policy aligned with the training objective
    much better than the previous frame-to-frame latent cosine heuristic.
    """
    from memory_module.model_with_memory import WanModelWithMemory
    from memory_module.nfp_head import NFPHead

    model = pipeline.low_noise_model
    lat_f = frame_embs.shape[0]
    if lat_f == 0:
        return []

    if not isinstance(model, WanModelWithMemory):
        return [1.0] * lat_f

    nfp_head = getattr(model, "nfp_head", None)
    if nfp_head is None or lat_f == 1:
        return [1.0] * lat_f

    with torch.no_grad():
        # We do not have full training-time hidden states exposed by WanI2V.generate(),
        # so we use the projected per-frame conditioning embeddings as a frame-level proxy.
        pred_latent = nfp_head.forward_per_frame(
            frame_embs.unsqueeze(0).to(next(nfp_head.parameters()).device),
            lat_f,
            num_spatial_per_frame=1,
        )  # [1, lat_f, z_dim]

        nfp_loss_dict = NFPHead.compute_loss_per_frame(
            pred_latent,
            video_latent.to(pred_latent.device),
            mse_weight=1.0,
            cosine_weight=1.0,
        )
        per_frame_surprise = (
            nfp_loss_dict["per_frame_surprise"].detach().float().cpu()
        )  # [lat_f - 1]

    surprises = per_frame_surprise.tolist()
    last_surprise = surprises[-1] if surprises else 1.0
    surprises.append(last_surprise)  # no next-frame target for the last frame

    logger.debug(
        "NFP proxy surprise: mean=%.4f, max=%.4f, min=%.4f",
        sum(surprises) / len(surprises), max(surprises), min(surprises),
    )
    return surprises


def _update_memory_bank_v2(bank, frame_embs, surprises, chunk_id, vae_stride_t=4):
    """Write frame embeddings + surprise scores into MemoryBank."""
    lat_f = frame_embs.shape[0]
    if len(surprises) != lat_f:
        raise ValueError(
            f"surprises length mismatch: got {len(surprises)}, expected {lat_f}"
        )

    bank.increment_age()
    for t in range(lat_f):
        bank.update(
            key_state=frame_embs[t].cpu(),
            value_visual=frame_embs[t].cpu(),
            surprise_score=surprises[t],
            timestep=chunk_id * lat_f * vae_stride_t + t * vae_stride_t,
            chunk_id=chunk_id,
        )

    logger.info("MemoryBank updated (chunk %d): %s", chunk_id, bank)
    stats = bank.get_stats()
    for k, v in stats.items():
        logger.info("  %s: %.4f", k, v)


def _write_chunk_action_dir(action_path, frame_offset, chunk_frame_num, tmp_dir):
    """Write a per-chunk subset of action data to a temp dir.

    WanI2V.generate() re-reads poses.npy/action.npy/intrinsics.npy from
    action_path on every call, slicing to frame_num. To inject correct
    per-chunk control, we write the chunk's subset to a temp dir.

    Args:
        action_path: original action data directory
        frame_offset: start frame index in the original episode
        chunk_frame_num: number of frames in this chunk
        tmp_dir: temporary directory to write to

    Returns:
        str: path to temp dir with chunk's action data
    """
    import numpy as np
    import shutil

    chunk_dir = os.path.join(tmp_dir, f"chunk_{frame_offset}")
    os.makedirs(chunk_dir, exist_ok=True)

    # poses.npy: [N, 4, 4]
    poses_full = np.load(os.path.join(action_path, "poses.npy"))
    chunk_end = min(frame_offset + chunk_frame_num, len(poses_full))
    np.save(os.path.join(chunk_dir, "poses.npy"), poses_full[frame_offset:chunk_end])

    # action.npy: [N, action_dim]
    action_file = os.path.join(action_path, "action.npy")
    if os.path.exists(action_file):
        action_full = np.load(action_file)
        np.save(os.path.join(chunk_dir, "action.npy"), action_full[frame_offset:chunk_end])

    # intrinsics.npy: shared across frames
    intrinsics_file = os.path.join(action_path, "intrinsics.npy")
    if os.path.exists(intrinsics_file):
        shutil.copy2(intrinsics_file, os.path.join(chunk_dir, "intrinsics.npy"))

    return chunk_dir


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- Step 1：处理 LoRA / 全参微调，准备最终 ckpt_dir ----
    if args.lora_path:
        args.ckpt_dir = _load_lora_and_prepare_ckpt(args)
    elif args.ft_model_dir:
        args.ckpt_dir = _load_ft_model_and_prepare_ckpt(args)

    # ---- Step 2：分布式初始化（与 inference_csgo.py 完全一致）----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    from wan.distributed.util import init_distributed_group
    from PIL import Image
    import torch.distributed as dist

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if rank == 0:
        logger.info("Rank 0 / World %d", world_size)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if args.ulysses_size > 1:
        init_distributed_group()

    # ---- Step 3：加载 WanI2V 管道（与 inference_csgo.py 完全一致）----
    cfg = WAN_CONFIGS["i2v-A14B"]

    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
    )

    # ---- Step 4：Memory 初始化（新增，仅当 --use_memory 时）----
    memory_bank = None
    if args.use_memory:
        from memory_module.memory_bank import MemoryBank

        logger.info(
            "Memory Bank enabled (max_size=%d). Converting pipeline to WanModelWithMemory...",
            args.memory_max_size,
        )
        wan_i2v = _convert_pipeline_to_memory(wan_i2v, memory_max_size=args.memory_max_size)
        memory_bank = MemoryBank(max_size=args.memory_max_size)
        logger.info("MemoryBank created: %s", memory_bank)

    # ---- Step 5: Generate video ----
    img = Image.open(args.image).convert("RGB")
    max_area = MAX_AREA_CONFIGS[args.size]
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.use_memory and memory_bank is not None and args.num_chunks > 1:
        # Chunked memory inference (v2 fix: real cross-chunk loop)
        from memory_module.model_with_memory import WanModelWithMemory
        import numpy as np
        import tempfile
        from wan.utils.cam_utils import (
            compute_relative_poses, interpolate_camera_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        from einops import rearrange

        # Load action data
        if args.action_path is None:
            logger.error("--action_path is required for chunked memory inference.")
            return

        c2ws = np.load(os.path.join(args.action_path, "poses.npy"))
        Ks_raw = torch.from_numpy(
            np.load(os.path.join(args.action_path, "intrinsics.npy"))
        ).float()
        wasd_action = np.load(os.path.join(args.action_path, "action.npy"))

        # Parse resolution
        h_str, w_str = args.size.split("*")
        h, w = int(h_str), int(w_str)

        total_frames = min(args.frame_num, ((len(c2ws) - 1) // 4) * 4 + 1)
        frames_per_chunk = max(5, total_frames // args.num_chunks)
        # Ensure frames_per_chunk is 4n+1
        frames_per_chunk = ((frames_per_chunk - 1) // 4) * 4 + 1

        all_videos = []
        chunk_id = 0
        frame_offset = 0

        # Temp dir for per-chunk action slices
        tmp_action_root = tempfile.mkdtemp(prefix='lingbot_chunk_')

        try:
            while frame_offset < total_frames:
                chunk_end = min(frame_offset + frames_per_chunk, total_frames)
                chunk_frame_num = chunk_end - frame_offset
                if chunk_frame_num < 5:
                    break
                # Ensure 4n+1
                chunk_frame_num = ((chunk_frame_num - 1) // 4) * 4 + 1
                chunk_end = frame_offset + chunk_frame_num

                logger.info(
                    "=== Chunk %d: frames %d-%d (%d frames) | bank_size=%d ===",
                    chunk_id, frame_offset, chunk_end - 1, chunk_frame_num, memory_bank.size(),
                )

                # Write per-chunk action data to temp dir
                # This ensures generate() reads the correct slice, not the full episode
                chunk_action_dir = _write_chunk_action_dir(
                    args.action_path, frame_offset, chunk_frame_num, tmp_action_root
                )

                # Prepare plucker embedding for this chunk (for memory retrieval/update)
                chunk_c2ws = c2ws[frame_offset:chunk_end]
                chunk_wasd = wasd_action[frame_offset:chunk_end]

                lat_h = round(np.sqrt(max_area * (h/w)) // wan_i2v.vae_stride[1] // wan_i2v.patch_size[1] * wan_i2v.patch_size[1])
                lat_w = round(np.sqrt(max_area / (h/w)) // wan_i2v.vae_stride[2] // wan_i2v.patch_size[2] * wan_i2v.patch_size[2])
                lat_f = (chunk_frame_num - 1) // wan_i2v.vae_stride[0] + 1

                Ks = get_Ks_transformed(Ks_raw, 480, 832, h, w, h, w)
                Ks_single = Ks[0]

                c2ws_infer = interpolate_camera_poses(
                    src_indices=np.linspace(0, len(chunk_c2ws) - 1, len(chunk_c2ws)),
                    src_rot_mat=chunk_c2ws[:, :3, :3],
                    src_trans_vec=chunk_c2ws[:, :3, 3],
                    tgt_indices=np.linspace(0, len(chunk_c2ws) - 1, lat_f),
                )
                c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
                Ks_repeated = Ks_single.repeat(len(c2ws_infer), 1).to(device)
                c2ws_infer = c2ws_infer.to(device)

                chunk_wasd_sub = torch.from_numpy(chunk_wasd[::4]).float().to(device)
                if len(chunk_wasd_sub) > lat_f:
                    chunk_wasd_sub = chunk_wasd_sub[:lat_f]
                elif len(chunk_wasd_sub) < lat_f:
                    pad = chunk_wasd_sub[-1:].repeat(lat_f - len(chunk_wasd_sub), 1)
                    chunk_wasd_sub = torch.cat([chunk_wasd_sub, pad], dim=0)

                c2ws_plucker_emb = get_plucker_embeddings(
                    c2ws_infer, Ks_repeated, h, w, only_rays_d=True
                )
                c1 = int(h // lat_h)
                c2 = int(w // lat_w)
                c2ws_plucker_emb = rearrange(
                    c2ws_plucker_emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=c1, c2=c2,
                )
                c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
                c2ws_plucker_emb = rearrange(
                    c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w,
                ).to(torch.bfloat16)

                wasd_tensor = chunk_wasd_sub[:, None, None, :].repeat(1, h, w, 1)
                wasd_tensor = rearrange(
                    wasd_tensor, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=c1, c2=c2,
                )
                wasd_tensor = wasd_tensor[None, ...]
                wasd_tensor = rearrange(
                    wasd_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w,
                ).to(torch.bfloat16)
                c2ws_plucker_emb_full = torch.cat([c2ws_plucker_emb, wasd_tensor], dim=1)

                model = wan_i2v.low_noise_model
                with torch.no_grad():
                    chunk_frame_embs = model.get_projected_frame_embs(
                        c2ws_plucker_emb_full.to(device)
                    )  # [lat_f, dim]

                # Retrieve from memory bank using real pose embeddings
                memory_states = None
                if memory_bank.size() > 0:
                    # Use first frame as query
                    query_emb = chunk_frame_embs[0]  # [dim]

                    retrieved = memory_bank.retrieve(
                        query_emb, top_k=args.memory_top_k, device=device
                    )
                    if retrieved is not None:
                        memory_states = retrieved["value_visuals"].unsqueeze(0)  # [1, K, dim]
                        logger.info(
                            "Retrieved %d frames, sim_mean=%.4f",
                            memory_states.shape[1],
                            retrieved["similarities"].mean().item(),
                        )

                # Inject memory and generate
                # Use chunk_action_dir so generate() reads the CORRECT chunk's control
                _patch_pipeline_memory(wan_i2v, memory_states)
                try:
                    video = wan_i2v.generate(
                        args.prompt,
                        img,
                        action_path=chunk_action_dir,  # per-chunk action data
                        max_area=max_area,
                        frame_num=chunk_frame_num,
                        shift=args.sample_shift,
                        sample_solver="unipc",
                        sampling_steps=args.sample_steps,
                        guide_scale=args.guide_scale,
                        seed=42 + chunk_id,
                        offload_model=False,
                    )
                finally:
                    _unpatch_pipeline_memory(wan_i2v)

                # --- Multi-GPU state sync ---
                # WanI2V.generate() only returns decoded video on rank 0.
                # Rank 0 computes the chunk update payload, then broadcasts the
                # next-step state so every rank keeps an identical bank/img view.
                sync_payload = [None]
                if rank == 0:
                    if video is None:
                        raise RuntimeError(
                            "Rank 0 did not receive generated video from WanI2V.generate()."
                        )

                    vae_device = next(wan_i2v.vae.model.parameters()).device
                    if vae_device != device:
                        wan_i2v.vae.model.to(device)
                    with torch.no_grad():
                        video_latent = wan_i2v.vae.encode([video.to(device)])[0]

                    chunk_surprises = _compute_chunk_surprises_v2(
                        pipeline=wan_i2v,
                        frame_embs=chunk_frame_embs,
                        video_latent=video_latent,
                    )
                    sync_payload[0] = {
                        "surprises": chunk_surprises,
                        "last_frame": video[:, -1].detach().cpu(),
                    }
                    all_videos.append(video)

                if dist.is_initialized():
                    dist.broadcast_object_list(sync_payload, src=0)

                payload = sync_payload[0]
                if payload is None:
                    raise RuntimeError("Failed to synchronize chunk state from rank 0.")

                _update_memory_bank_v2(
                    bank=memory_bank,
                    frame_embs=chunk_frame_embs,
                    surprises=payload["surprises"],
                    chunk_id=chunk_id,
                )

                # Use last frame as first frame for next chunk on every rank.
                from torchvision.transforms.functional import to_pil_image
                last_frame = payload["last_frame"].clamp(-1, 1)
                img = to_pil_image((last_frame + 1) / 2)

                chunk_id += 1
                frame_offset = chunk_end

        finally:
            # Clean up temp action dirs
            import shutil
            shutil.rmtree(tmp_action_root, ignore_errors=True)

        # Concatenate all chunks
        if rank == 0 and all_videos:
            video = torch.cat(all_videos, dim=1)  # [C, total_F, H, W]
            logger.info("Chunked episode complete: %d chunks, %d total frames",
                        chunk_id, video.shape[1])

    elif args.use_memory and memory_bank is not None:
        # Single-chunk memory inference with real pose query
        from memory_module.model_with_memory import WanModelWithMemory

        memory_states = None
        if memory_bank.size() > 0:
            # Build real pose query from action_path if available
            model = wan_i2v.low_noise_model
            if args.action_path is not None and isinstance(model, WanModelWithMemory):
                import numpy as np
                from wan.utils.cam_utils import (
                    compute_relative_poses, interpolate_camera_poses,
                    get_plucker_embeddings, get_Ks_transformed,
                )
                from einops import rearrange

                h_str, w_str = args.size.split("*")
                h_val, w_val = int(h_str), int(w_str)
                sc2ws = np.load(os.path.join(args.action_path, "poses.npy"))
                sKs_raw = torch.from_numpy(
                    np.load(os.path.join(args.action_path, "intrinsics.npy"))
                ).float()
                s_frame_num = min(args.frame_num, ((len(sc2ws) - 1) // 4) * 4 + 1)
                sc2ws = sc2ws[:s_frame_num]
                s_lat_f = (s_frame_num - 1) // wan_i2v.vae_stride[0] + 1
                s_lat_h = round(np.sqrt(max_area * (h_val/w_val)) // wan_i2v.vae_stride[1] // wan_i2v.patch_size[1] * wan_i2v.patch_size[1])
                s_lat_w = round(np.sqrt(max_area / (h_val/w_val)) // wan_i2v.vae_stride[2] // wan_i2v.patch_size[2] * wan_i2v.patch_size[2])

                sKs = get_Ks_transformed(sKs_raw, 480, 832, h_val, w_val, h_val, w_val)
                sc2ws_infer = interpolate_camera_poses(
                    np.linspace(0, len(sc2ws)-1, len(sc2ws)),
                    sc2ws[:, :3, :3], sc2ws[:, :3, 3],
                    np.linspace(0, len(sc2ws)-1, s_lat_f),
                )
                sc2ws_infer = compute_relative_poses(sc2ws_infer, framewise=True)
                s_Ks_rep = sKs[0].repeat(len(sc2ws_infer), 1).to(device)
                sc2ws_infer = sc2ws_infer.to(device)

                s_plucker = get_plucker_embeddings(sc2ws_infer, s_Ks_rep, h_val, w_val, only_rays_d=True)
                sc1, sc2 = int(h_val // s_lat_h), int(w_val // s_lat_w)
                s_plucker = rearrange(s_plucker, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=sc1, c2=sc2)
                s_plucker = rearrange(s_plucker[None], 'b (f h w) c -> b c f h w', f=s_lat_f, h=s_lat_h, w=s_lat_w).to(torch.bfloat16)

                s_wasd = np.load(os.path.join(args.action_path, "action.npy"))[:s_frame_num]
                s_wasd_sub = torch.from_numpy(s_wasd[::4]).float().to(device)[:s_lat_f]
                if len(s_wasd_sub) < s_lat_f:
                    s_wasd_sub = torch.cat([s_wasd_sub, s_wasd_sub[-1:].repeat(s_lat_f - len(s_wasd_sub), 1)])
                s_wasd_t = rearrange(
                    s_wasd_sub[:, None, None, :].repeat(1, h_val, w_val, 1),
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=sc1, c2=sc2
                )
                s_wasd_t = rearrange(s_wasd_t[None], 'b (f h w) c -> b c f h w', f=s_lat_f, h=s_lat_h, w=s_lat_w).to(torch.bfloat16)
                s_plucker_full = torch.cat([s_plucker, s_wasd_t], dim=1)

                with torch.no_grad():
                    query_emb = model.get_projected_frame_embs(
                        s_plucker_full.to(device)
                    )[0]  # First frame as query [dim]
            else:
                # Fallback: zero query (no action_path)
                query_emb = torch.zeros(model.dim, device=device)
                logger.warning("No action_path for single-chunk, using zero query (degenerate).")

            retrieved = memory_bank.retrieve(query_emb, top_k=args.memory_top_k, device=device)
            if retrieved is not None:
                memory_states = retrieved["value_visuals"].unsqueeze(0)
                logger.info("Retrieved %d memory frames, sim_mean=%.4f",
                            memory_states.shape[1], retrieved["similarities"].mean().item())

        _patch_pipeline_memory(wan_i2v, memory_states)
        try:
            video = wan_i2v.generate(
                args.prompt, img,
                action_path=args.action_path, max_area=max_area,
                frame_num=args.frame_num, shift=args.sample_shift,
                sample_solver="unipc", sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale, seed=42, offload_model=False,
            )
        finally:
            _unpatch_pipeline_memory(wan_i2v)

    else:
        # No memory: identical to inference_csgo.py
        video = wan_i2v.generate(
            args.prompt, img,
            action_path=args.action_path, max_area=max_area,
            frame_num=args.frame_num, shift=args.sample_shift,
            sample_solver="unipc", sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale, seed=42, offload_model=False,
        )

    # ---- Step 6: Save output ----
    if rank == 0 and video is not None:
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1, normalize=True, value_range=(-1, 1),
        )
        logger.info("Saved video -> %s", args.save_file)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
