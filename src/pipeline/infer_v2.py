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
                        help="是否启用 Memory Bank（默认 False，保持后向兼容）")
    parser.add_argument("--memory_max_size", type=int, default=50,
                        help="Memory Bank 最大容量（默认 50）")

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


def _update_memory_bank(bank, video, pipeline, device, clip_start_frame: int):
    """用当前 clip 的 VAE latent 更新 MemoryBank（surprise-driven）。

    直接复用 infer.py 的 _update_memory_bank 逻辑。
    由于 infer_v2 场景下没有预先计算 c2ws_plucker_emb_raw（单 clip 模式），
    此处用 VAE latent 的帧间 cosine distance 作为 surprise score，
    pose_emb 占位为 latent 的空间均值（退化为纯 latent 检索）。

    注意：如果 action_path 提供了 poses/intrinsics，可调用
    infer.py 中的 _compute_plucker_emb 获得真实 pose_emb；
    这里为保持简洁，直接用 latent 均值作为 pose_emb 索引。
    """
    import torch.nn.functional as F
    from memory_module.memory_bank import MemoryBank

    # FIX[B-02]：offload_model=True 时 VAE 可能已在 CPU，需先移回 device
    vae_device = next(pipeline.vae.model.parameters()).device
    if vae_device != device:
        pipeline.vae.model.to(device)

    with torch.no_grad():
        latent = pipeline.vae.encode([video.to(device)])[0]  # [z_dim, lat_f, h, w]

    lat_f = latent.shape[1]
    vae_stride_t = pipeline.vae_stride[0]

    for t in range(lat_f):
        # Surprise score：帧间 cosine distance（第一帧默认 1.0）
        if t == 0:
            surprise = 1.0
        else:
            prev = latent[:, t - 1].flatten().float()
            curr = latent[:, t].flatten().float()
            cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
            surprise = float(1.0 - cos_sim)

        # pose_emb：使用 latent 空间均值作为帧标识嵌入（退化索引）
        pose_emb = latent[:, t].mean(dim=(-2, -1)).cpu()  # [z_dim]

        bank.update(
            pose_emb=pose_emb,
            latent=latent[:, t].cpu(),
            surprise_score=surprise,
            timestep=clip_start_frame + t * vae_stride_t,
        )

    logger.info("MemoryBank updated: %s", bank)


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

    # ---- Step 5：加载图像，生成视频 ----
    img = Image.open(args.image).convert("RGB")
    max_area = MAX_AREA_CONFIGS[args.size]
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.use_memory and memory_bank is not None:
        # Memory 推理：检索 → generate → 更新
        from memory_module.model_with_memory import WanModelWithMemory

        # 检索 memory（首 clip 时 bank 为空，memory_states=None）
        memory_states = None
        if memory_bank.size() > 0:
            # 使用随机嵌入作为 query（无 camera 信息时的退化查询）
            query_dim = memory_bank.frames[0].pose_emb.shape[0]
            query_emb = torch.zeros(query_dim, device=device)
            retrieved = memory_bank.retrieve(query_emb, top_k=4, device=device)
            if retrieved is not None:
                memory_states = retrieved.unsqueeze(0)  # [1, K, dim]
                logger.info("Retrieved %d memory frames for inference.", retrieved.shape[0])

        # 注入 memory_states 并生成
        _patch_pipeline_memory(wan_i2v, memory_states)
        try:
            video = wan_i2v.generate(
                args.prompt,
                img,
                action_path=args.action_path,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale,
                seed=42,
                offload_model=False,
            )
        finally:
            _unpatch_pipeline_memory(wan_i2v)

        # 更新 MemoryBank（仅 rank 0 有 video）
        if rank == 0 and video is not None:
            _update_memory_bank(
                bank=memory_bank,
                video=video,
                pipeline=wan_i2v,
                device=device,
                clip_start_frame=0,
            )

    else:
        # 无 memory 路径：与 inference_csgo.py 完全一致
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=args.action_path,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=42,
            offload_model=False,
        )

    # ---- Step 6：保存输出（与 inference_csgo.py 完全一致）----
    if rank == 0 and video is not None:
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logger.info("Saved video → %s", args.save_file)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
