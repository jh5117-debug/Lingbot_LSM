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
    parser.add_argument("--num_clips", type=int, default=1,
                        help="Memory 模式下生成的 clip 数量（默认 1，等价于原行为）")

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


def _update_memory_bank(bank, video, pipeline, device, clip_start_frame: int,
                        c2ws_plucker_emb=None, nfp_head=None, last_hidden_states=None):
    """用当前 clip 的 VAE latent 更新 MemoryBank（surprise-driven）。

    问题3修复：pose_emb 优先用 get_projected_frame_embs() 计算模型空间嵌入（dim=5120），
    与 MemoryCrossAttention 期待的 Key/Value 维度一致；退化时用 latent 均值（dim=16）。

    问题4修复（随问题3一起）：改用投影嵌入后，K/V 具有真正的视觉语义投影特征。

    问题5修复：surprise score 接口新增 nfp_head 参数，当前保留帧间 cosine distance 退化方案。
    # TODO：完整对齐需在推理时通过 forward hook 捕获 model.blocks[-1] 的 hidden_states，
    # 传入 nfp_head(hidden_states) 计算真实预测误差；当前用帧间 cosine distance 代理。

    Args:
        bank:               MemoryBank 实例
        video:              当前 clip 的视频帧（Tensor 或类 PIL 格式，与 VAE encode 兼容）
        pipeline:           WanI2V 管道实例（含 vae）
        device:             目标设备
        clip_start_frame:   当前 clip 在完整视频序列中的起始帧索引
        c2ws_plucker_emb:   可选，[1, C, lat_f, lat_h, lat_w]，来自 dit_cond_dict；
                            提供时用 get_projected_frame_embs() 计算 5120 维 pose_emb（问题3修复）
        nfp_head:           可选，NFPHead 实例；接口预留，当前退化为 cosine distance（问题5修复）
        last_hidden_states: 可选，[1, L, 5120]，forward hook 捕获的 model.blocks[-1] 输出；
                            提供时用 NFPHead 计算 clip-level surprise（M-1 修复）
    """
    import torch.nn.functional as F
    from memory_module.memory_bank import MemoryBank
    from memory_module.model_with_memory import WanModelWithMemory

    # FIX[B-02]：offload_model=True 时 VAE 可能已在 CPU，需先移回 device
    vae_device = next(pipeline.vae.model.parameters()).device
    if vae_device != device:
        pipeline.vae.model.to(device)

    with torch.no_grad():
        latent = pipeline.vae.encode([video.to(device)])[0]  # [z_dim, lat_f, h, w]

    lat_f = latent.shape[1]
    vae_stride_t = pipeline.vae_stride[0]

    # 问题3修复：计算 per-frame pose embedding（5120维，与 MemoryCrossAttention 对齐）
    model = pipeline.low_noise_model
    frame_embs = None
    if c2ws_plucker_emb is not None and isinstance(model, WanModelWithMemory):
        with torch.no_grad():
            frame_embs = model.get_projected_frame_embs(
                c2ws_plucker_emb.to(device)
            )  # [lat_f, dim=5120]

    # BLOCK-2 修复：16维退化路径与 MemoryCrossAttention Linear(5120,5120) 不兼容，
    # 必须在此处拦截，避免将 dim=16 的 pose_emb 存入 bank 后在 retrieve() 时 crash。
    if frame_embs is None:
        logger.warning(
            "_update_memory_bank: c2ws_plucker_emb not provided or model is not "
            "WanModelWithMemory (model=%s); skipping bank update to prevent "
            "MemoryCrossAttention dim mismatch (expected dim=5120).",
            type(model).__name__,
        )
        return

    # M-1 修复：使用 NFPHead 计算 clip-level surprise（若 last_hidden_states 可用）
    from memory_module.nfp_head import NFPHead as _NFPHead
    clip_surprise = None
    if last_hidden_states is not None and hasattr(model, 'nfp_head') and model.nfp_head is not None:
        with torch.no_grad():
            _hs = last_hidden_states.to(device).to(
                next(model.nfp_head.parameters()).dtype
            )  # [1, L, 5120]
            _pred = model.nfp_head.forward(_hs)  # [1, z_dim=16]
            _actual = latent.float()[:, -1].mean(dim=[-2, -1]).unsqueeze(0)  # [1, 16] BLOCK-B 修复：最后帧空间均值，与 train_v2_stage1.py:770 一致
            clip_surprise = _NFPHead.compute_surprise(
                _pred.float(), _actual
            ).item()  # scalar
        logger.info("NFP clip surprise: %.4f", clip_surprise)

    for t in range(lat_f):
        if clip_surprise is not None:
            # M-1 修复：使用 NFPHead clip-level surprise（per-clip 粒度对齐训练）
            surprise = clip_surprise
        elif t == 0:
            surprise = 1.0
        else:
            prev = latent[:, t - 1].flatten().float()
            curr = latent[:, t].flatten().float()
            cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
            surprise = float(1.0 - cos_sim)

        pose_emb = frame_embs[t].cpu()  # [dim=5120]，由 BLOCK-2 保证必为 5120 维

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
        from memory_module.model_with_memory import WanModelWithMemory
        all_videos = []
        current_img = img

        # BLOCK-1 修复：预加载 action_path 中的 poses/actions/intrinsics，
        # 构建 c2ws_plucker_emb 传入 _update_memory_bank()，确保问题3修复（5120维 pose_emb）生效。
        _c2ws_plucker_emb_for_bank = None
        # M-4 修复：广播 memory_states 初始化（多卡分布式推理时各 rank 保持一致）
        _broadcast_memory_states = None
        if args.action_path and os.path.isdir(args.action_path):
            try:
                import numpy as _np
                from pipeline.dataloader import build_dit_cond_dict as _build_dit_cond_dict
                _poses_np = _np.load(os.path.join(args.action_path, "poses.npy"))
                _actions_np = _np.load(os.path.join(args.action_path, "action.npy"))
                _intrinsics_np = _np.load(os.path.join(args.action_path, "intrinsics.npy"))
                _h, _w = [int(x) for x in args.size.split("*")]
                _cond = _build_dit_cond_dict(
                    poses=torch.from_numpy(_poses_np).float(),
                    actions=torch.from_numpy(_actions_np).float(),
                    intrinsics=torch.from_numpy(_intrinsics_np).float(),
                    height=_h,
                    width=_w,
                )
                _c2ws_plucker_emb_for_bank = _cond["c2ws_plucker_emb"][0]
                logger.info(
                    "Preloaded c2ws_plucker_emb for MemoryBank, shape=%s",
                    tuple(_c2ws_plucker_emb_for_bank.shape),
                )
            except Exception as _e:
                logger.warning(
                    "Could not build c2ws_plucker_emb from action_path=%s: %s; "
                    "MemoryBank updates will be skipped (see BLOCK-1 fix).",
                    args.action_path, _e,
                )
        else:
            logger.warning(
                "action_path is None or not a directory (%s); "
                "MemoryBank updates will be skipped.", args.action_path,
            )

        for clip_idx in range(args.num_clips):
            logger.info("Generating clip %d/%d ...", clip_idx + 1, args.num_clips)

            # M-4：如果有广播来的 memory_states，优先使用（多卡一致性）
            if _broadcast_memory_states is not None:
                memory_states = _broadcast_memory_states
                _broadcast_memory_states = None  # 消费后清空
                logger.info("Clip %d: using broadcast memory_states (M-4 fix)", clip_idx + 1)
            else:
                # 检索 memory（首 clip 时 bank 为空，memory_states=None）
                memory_states = None
                if memory_bank.size() > 0:
                    # HIGH-1 修复：用 get_projected_frame_embs 计算真实 pose query
                    model_lnm = wan_i2v.low_noise_model
                    if isinstance(model_lnm, WanModelWithMemory) and _c2ws_plucker_emb_for_bank is not None:
                        with torch.no_grad():
                            _qfe = model_lnm.get_projected_frame_embs(
                                _c2ws_plucker_emb_for_bank.to(device)
                            )  # [lat_f, dim=5120]
                        query_emb = _qfe[0].to(device)  # [5120]，当前 clip 第一帧 pose emb
                    else:
                        query_dim = memory_bank.frames[0].pose_emb.shape[0]
                        query_emb = torch.zeros(query_dim, device=device)
                        logger.warning("Clip %d: falling back to zero query (no pose data)", clip_idx + 1)
                    retrieved = memory_bank.retrieve(query_emb, top_k=4, device=device)
                    if retrieved is not None:
                        memory_states = retrieved.unsqueeze(0)  # [1, K, dim]
                        logger.info("Clip %d: retrieved %d memory frames.", clip_idx + 1, retrieved.shape[0])

            # M-1 修复：注册 forward hook 捕获 model.blocks[-1] hidden_states（供 NFPHead 使用）
            _nfp_captured_hs = {}
            _nfp_hook_handle = None
            _model_lnm = wan_i2v.low_noise_model
            if isinstance(_model_lnm, WanModelWithMemory):
                def _nfp_capture_hook(module, inp, out):
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    _nfp_captured_hs['hs'] = hs.detach().cpu()
                _nfp_hook_handle = _model_lnm.blocks[-1].register_forward_hook(_nfp_capture_hook)

            # 注入 memory_states 并生成
            _patch_pipeline_memory(wan_i2v, memory_states)
            try:
                video = wan_i2v.generate(
                    args.prompt,
                    current_img,
                    action_path=args.action_path,
                    max_area=max_area,
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver="unipc",
                    sampling_steps=args.sample_steps,
                    guide_scale=args.guide_scale,
                    seed=42 + clip_idx,
                    offload_model=False,
                )
            finally:
                _unpatch_pipeline_memory(wan_i2v)
                # M-1 修复：移除 forward hook
                if _nfp_hook_handle is not None:
                    _nfp_hook_handle.remove()
                    _nfp_hook_handle = None
            _last_hs_for_nfp = _nfp_captured_hs.pop('hs', None)

            if rank == 0 and video is not None:
                # HIGH-3 修复：确保存入 all_videos 的是 torch.Tensor
                import numpy as _np_h3
                _video_tensor = torch.from_numpy(video.copy()) if isinstance(video, _np_h3.ndarray) else video
                all_videos.append(_video_tensor)
                # 更新 Memory Bank（BLOCK-1 修复：传入 c2ws_plucker_emb）
                _update_memory_bank(
                    bank=memory_bank,
                    video=video,
                    pipeline=wan_i2v,
                    device=device,
                    clip_start_frame=clip_idx * args.frame_num,
                    c2ws_plucker_emb=_c2ws_plucker_emb_for_bank,
                    last_hidden_states=_last_hs_for_nfp,
                )
                # 使用最后一帧作为下一 clip 的初始帧
                # video shape: [C=3, T, H, W]，取最后时间步 [:, -1] → [C, H, W]
                from PIL import Image as PILImage
                import numpy as _np_frame
                last_frame_chw = video[:, -1]  # [C=3, H, W]
                if hasattr(last_frame_chw, 'cpu'):
                    last_frame_chw = last_frame_chw.cpu().float().numpy()
                # CHW → HWC，[-1,1] → [0,255]
                last_frame_hwc = last_frame_chw.transpose(1, 2, 0)
                last_frame_np = (last_frame_hwc * 127.5 + 127.5).clip(0, 255).astype(_np_frame.uint8)
                current_img = PILImage.fromarray(last_frame_np)
                logger.info("Clip %d: memory bank updated. Size=%d", clip_idx + 1, memory_bank.size())

            # M-4 修复：广播 memory_states 给所有 rank（仅多卡 Ulysses 模式需要）
            if world_size > 1 and dist.is_initialized():
                # rank 0 预计算下一 clip 的 memory_states 供广播
                if rank == 0 and memory_bank.size() > 0:
                    _m4_model = wan_i2v.low_noise_model
                    if isinstance(_m4_model, WanModelWithMemory) and _c2ws_plucker_emb_for_bank is not None:
                        with torch.no_grad():
                            _m4_qfe = _m4_model.get_projected_frame_embs(
                                _c2ws_plucker_emb_for_bank.to(device)
                            )
                        _m4_q = _m4_qfe[0].to(device)
                        _m4_retrieved = memory_bank.retrieve(_m4_q, top_k=4, device=device)
                        _m4_states = _m4_retrieved.unsqueeze(0) if _m4_retrieved is not None else None
                    else:
                        _m4_states = None
                else:
                    _m4_states = None
                # 广播 memory_states（使用 broadcast_object_list 支持 None 和 Tensor）
                _m4_bcast = [_m4_states]
                dist.broadcast_object_list(_m4_bcast, src=0)
                _broadcast_memory_states = _m4_bcast[0]
                if _broadcast_memory_states is not None:
                    _broadcast_memory_states = _broadcast_memory_states.to(device)
            else:
                _broadcast_memory_states = None

        # 拼接所有 clips
        video = torch.cat(all_videos, dim=1) if all_videos else None  # BLOCK-A 修复：沿时间维度 T(dim=1) 拼接，[C, T*N, H, W]

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
