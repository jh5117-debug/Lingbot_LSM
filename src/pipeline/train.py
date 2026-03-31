"""
train.py — LingBot-World Memory Enhancement 训练脚本

在 lingbot-csgo-finetune/train_lingbot_csgo.py 基础上进行修改：
  - 替换 WanModel → WanModelWithMemory（via from_wan_model()）
  - Stage1：冻结 DiT 所有参数，只训 memory_blocks（MemoryCrossAttention + NFPHead + memory_norm）
  - Stage2：全参数解冻，DiT lr=lr_dit，记忆模块 lr=lr  # PENDING[D-03]
  - 增加 NFP loss：L_total = L_diffusion + 0.1 * L_nfp
  - 训练时 memory_states=None（memory bank 是推理时机制，训练时不使用）
  - 支持 DeepSpeed ZeRO-3 + Accelerate

使用方式：
    accelerate launch --config_file src/configs/accelerate_config.yaml \\
        src/pipeline/train.py \\
        --dataset_root /data/csgo_processed \\
        --metadata_train /data/csgo_processed/metadata_train.csv \\
        --ckpt_dir /models/lingbot-world-base-act \\
        --output_dir /output/csgo_memory_stage1 \\
        --stage 1 \\
        --num_epochs 10

参考：
  - refs/lingbot-csgo-finetune/train_lingbot_csgo.py  训练循环基础
  - src/memory_module/model_with_memory.py            WanModelWithMemory 接口
  - src/memory_module/nfp_head.py                     NFPHead.compute_loss 接口
  - src/pipeline/dataloader.py                        CSGODataset / build_dataloader 接口
  - refs/DiffSynth-Studio/diffsynth/diffusion/flow_match.py  FlowMatchScheduler 参考
"""

import argparse
import gc
import logging
import os
import sys
import time
import warnings
from functools import wraps
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint as torch_checkpoint

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 日志（code_standards.md §4）
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path：引入 lingbot-world 模块 和 src/ 下的 memory_module
# 本文件位于 src/pipeline/，需要向上两层到 Lingbot_LSM/，再进入 refs/lingbot-world
# ---------------------------------------------------------------------------
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))  # → src/pipeline/
_SRC_DIR = os.path.dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = os.path.join(_PROJECT_ROOT, 'refs', 'lingbot-world')
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
# memory_module 在 src/ 下，需要将 src/ 加入 sys.path 供延迟导入使用
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
# pipeline/ 下的 dataloader.py 需要将 pipeline/ 加入 sys.path（直接运行时无 parent package）
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。所有超参数通过参数传入，不 hardcode（code_standards.md §2）。"""
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement 训练脚本（Stage1/Stage2）"
    )

    # ---- 路径 ----
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="CSGO 预处理数据集根目录（包含 train/clips/...）",
    )
    parser.add_argument(
        "--metadata_train", type=str, required=True,
        help="训练集 metadata CSV 路径（如 /data/metadata_train.csv）",
    )
    parser.add_argument(
        "--metadata_val", type=str, default=None,
        help="验证集 metadata CSV 路径（可选；有则在每个 epoch 结束后跑 eval loop）",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="lingbot-world 预训练权重目录（含 low_noise_model 子目录）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="checkpoint 保存目录",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="从指定 checkpoint 目录恢复训练（accelerator.load_state）",
    )
    parser.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="DeepSpeed config json 路径（可选；通过 accelerate config 传入时可不指定）",
    )

    # ---- 训练阶段 ----
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help="训练阶段：1=冻结DiT只训记忆模块；2=全参数解冻联合微调",
    )

    # ---- 批次与数据 ----
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="每卡 batch size（默认 1，与 train_lingbot_csgo.py 一致）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader 工作进程数（code_standards.md §3 默认 4）",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="视频高度（数据集固定 480）",
    )
    parser.add_argument(
        "--width", type=int, default=832,
        help="视频宽度（数据集固定 832）",
    )
    # B-02: --num_frames 参数化，替换训练/验证循环中所有 num_frames=81 硬编码
    parser.add_argument(
        "--num_frames", type=int, default=81,
        help="每 clip 帧数（VAE 约束 4n+1，默认 81 对应 n=20）",
    )

    # ---- 训练超参数 ----
    parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="训练总 epoch 数",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="记忆模块（Stage1）或整体（Stage2 记忆模块部分）学习率",
    )
    parser.add_argument(
        "--lr_dit", type=float, default=1e-5,
        help="Stage2 DiT 参数学习率（Stage1 时忽略）",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="梯度累积步数（有效批大小 = batch_size × gradient_accumulation_steps × num_gpus）",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="梯度裁剪阈值（code_standards.md §2 要求）",
    )
    parser.add_argument(
        "--nfp_loss_weight", type=float, default=0.1,
        help="NFP loss 权重（L_total = L_diffusion + nfp_loss_weight * L_nfp）",
    )

    # ---- 日志与 checkpoint ----
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="每 N steps 保存一次 checkpoint",
    )
    parser.add_argument(
        "--log_every", type=int, default=10,
        help="每 N steps 打印一次 loss / lr",
    )

    # ---- 调试 ----
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只跑 2 steps 验证训练流程（code_standards.md §2）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Flow Matching Schedule（直接复用 train_lingbot_csgo.py 逻辑）
# ---------------------------------------------------------------------------

class FlowMatchingSchedule:
    """预计算 Flow Matching sigma schedule 及训练权重。

    参考 train_lingbot_csgo.py LingBotTrainer.__init__ + DiffSynth FlowMatchScheduler。

    LingBot/Wan 使用 shift=10.0 的 sigma schedule（与 DiffSynth default shift=5 不同）。
    只训 low_noise_model 负责的低噪声段（timestep < 0.947 * 1000 = 947）。

    Args:
        num_train_timesteps: sigma schedule 步数，默认 1000
        shift:               schedule shift 参数，LingBot 为 10.0
        boundary:            low_noise_model 有效范围上界（0.947），高于此的时间步不训练
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 10.0,
        boundary: float = 0.947,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.boundary = boundary

        # sigma schedule：shift=10.0 的 Wan/LingBot schedule
        # 公式：sigma_shifted = shift * sigma_linear / (1 + (shift - 1) * sigma_linear)
        sigmas_linear = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas_linear / (1 + (shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * num_train_timesteps

        # 有效训练范围（low_noise_model 负责 t < 947）
        max_timestep = boundary * num_train_timesteps  # 947
        self.valid_train_indices = torch.where(self.timesteps_schedule < max_timestep)[0]

        # 训练权重（DiffSynth FlowMatchScheduler.set_training_weight）
        # 高斯加权，以 t=500 为中心，避免极端噪声步主导训练
        x = self.timesteps_schedule
        steps = num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        self.training_weights = y_shifted * (steps / y_shifted.sum())

        logger.info(
            "FlowMatchingSchedule: shift=%.1f，有效时间步 %d/%d（t < %.0f）",
            shift, len(self.valid_train_indices), num_train_timesteps, max_timestep,
        )

    def sample_timestep(self) -> Tuple[float, torch.Tensor, float]:
        """从有效训练范围均匀随机采样一个时间步。

        Returns:
            sigma:           float，噪声比例 [0, 1]
            t:               Tensor scalar，时间步值（传给模型的 t）
            training_weight: float，对应该时间步的训练权重
        """
        idx = self.valid_train_indices[
            torch.randint(len(self.valid_train_indices), (1,)).item()
        ].item()
        sigma = self.sigmas[idx].item()
        t = self.timesteps_schedule[idx]
        training_weight = self.training_weights[idx].item()
        return sigma, t, training_weight


# ---------------------------------------------------------------------------
# 模型加载与参数冻结
# ---------------------------------------------------------------------------

def load_model(ckpt_dir: str, device: torch.device) -> "WanModelWithMemory":
    """从预训练 lingbot-world 权重加载 WanModelWithMemory。

    加载步骤：
        1. WanModel.from_pretrained(ckpt_dir, subfolder="low_noise_model", control_type="act")
        2. WanModelWithMemory.from_wan_model(base_model)  — 新增参数随机初始化

    Args:
        ckpt_dir: lingbot-world 预训练权重目录
        device:   目标设备（accelerator.device）

    Returns:
        WanModelWithMemory 实例（bfloat16，train 模式）
    """
    from wan.modules.model import WanModel
    from memory_module.model_with_memory import WanModelWithMemory

    logger.info("加载 low_noise_model（control_type=act）from %s ...", ckpt_dir)
    base_model = WanModel.from_pretrained(
        ckpt_dir,
        subfolder="low_noise_model",
        torch_dtype=torch.bfloat16,
        control_type="act",
    )

    logger.info("转换为 WanModelWithMemory ...")
    model = WanModelWithMemory.from_wan_model(base_model)
    model.train()

    # 验证控制信号嵌入维度（act mode 应为 Linear(1792, 5120)）
    wancamctrl = model.patch_embedding_wancamctrl
    logger.info(
        "patch_embedding_wancamctrl: Linear(%d, %d)（期望 1792, 5120）",
        wancamctrl.in_features, wancamctrl.out_features,
    )

    del base_model
    gc.collect()
    return model


def freeze_for_stage1(model: nn.Module) -> List[nn.Parameter]:
    """Stage1：冻结 DiT 所有参数，只训 memory 模块。

    冻结策略：
        1. model.requires_grad_(False)  — 冻结全部参数
        2. 遍历 model.memory_blocks（MemoryBlockWrapper）：
           解冻 memory_cross_attn + memory_norm
        3. 解冻 model.nfp_head

    MemoryBlockWrapper.block（原始 WanAttentionBlock）保持冻结，
    只有新增的 memory 参数参与梯度更新。

    Args:
        model: WanModelWithMemory 实例

    Returns:
        trainable_params: 可训参数列表（传给 optimizer）
    """
    # Step 1：全部冻结
    model.requires_grad_(False)

    # Step 2：解冻所有 MemoryBlockWrapper 中的 memory 组件
    num_unfrozen_blocks = 0
    for block in model.blocks:
        # model.blocks 中部分是 MemoryBlockWrapper，部分是原始 WanAttentionBlock
        if hasattr(block, 'memory_cross_attn'):
            block.memory_cross_attn.requires_grad_(True)
            block.memory_norm.requires_grad_(True)
            num_unfrozen_blocks += 1

    # Step 3：解冻 NFPHead
    model.nfp_head.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Stage1 冻结：解冻 %d 个 MemoryBlockWrapper memory 组件 + NFPHead | "
        "可训参数 %s / 总参数 %s（%.2f%%）",
        num_unfrozen_blocks,
        f"{trainable_count:,}", f"{total_count:,}",
        100.0 * trainable_count / max(total_count, 1),
    )
    return trainable_params


def setup_stage2_optimizer(
    model: nn.Module,
    lr_memory: float,
    lr_dit: float,
    weight_decay: float,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], torch.optim.Optimizer]:
    """Stage2：全参数解冻，DiT 参数使用较低 lr 防止遗忘。

    # PENDING[D-03]: Stage2 的起点权重尚未确定（D-03 OPEN）。
    # 当前假设：加载 Stage1 训练完成的 checkpoint（memory 模块已训练），
    #            原始 DiT 参数继续使用 lingbot-world 预训练权重（或对方提供的 CSGO-DiT 权重）。
    # D-03 解除后需要修改：
    #   - 选项A（原始 lingbot-world）：本函数无需改动，在 main() 中直接加载 Stage1 checkpoint
    #   - 选项B（对方 CSGO-DiT 权重）：在 main() 的 Stage2 模型加载处，
    #     用对方权重替换 base_model，再调用 from_wan_model()，
    #     然后加载 Stage1 checkpoint 覆盖 memory 模块参数
    # 待修改位置：main() 中标注了 "# PENDING[D-03]" 的代码段

    Args:
        model:        WanModelWithMemory（所有参数将被解冻）
        lr_memory:    记忆模块参数学习率（args.lr）
        lr_dit:       DiT 参数学习率（args.lr_dit，通常 lr/10）
        weight_decay: AdamW weight decay

    Returns:
        (memory_params, dit_params, optimizer)
    """
    # 解冻全部参数
    model.requires_grad_(True)

    # 区分参数组：memory 模块 vs 原始 DiT
    memory_param_ids = set()
    for block in model.blocks:
        if hasattr(block, 'memory_cross_attn'):
            for p in block.memory_cross_attn.parameters():
                memory_param_ids.add(id(p))
            for p in block.memory_norm.parameters():
                memory_param_ids.add(id(p))
    for p in model.nfp_head.parameters():
        memory_param_ids.add(id(p))

    memory_params = [p for p in model.parameters() if id(p) in memory_param_ids]
    dit_params = [p for p in model.parameters() if id(p) not in memory_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": memory_params, "lr": lr_memory},
            {"params": dit_params, "lr": lr_dit},
        ],
        weight_decay=weight_decay,
    )

    mem_count = sum(p.numel() for p in memory_params)
    dit_count = sum(p.numel() for p in dit_params)
    logger.info(
        "Stage2 全参数解冻：memory 模块 %s 参数（lr=%.2e），DiT %s 参数（lr=%.2e）",
        f"{mem_count:,}", lr_memory, f"{dit_count:,}", lr_dit,
    )
    return memory_params, dit_params, optimizer


# ---------------------------------------------------------------------------
# 梯度检查点（monkey-patch，参考 train_lingbot_csgo.py）
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module) -> int:
    """对 WanModelWithMemory 的每个 block（含 MemoryBlockWrapper）启用梯度检查点。

    WanModel 不支持 diffusers 标准接口，需要 monkey-patch 每个 block 的 forward 方法。
    与 train_lingbot_csgo.py 逻辑相同，但同时处理 MemoryBlockWrapper（也是 WanAttentionBlock）。

    WanAttentionBlock.forward 签名：
        forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None)

    Args:
        model: WanModelWithMemory 实例

    Returns:
        patched: 成功 patch 的 block 数量
    """
    patched = 0

    # 找到主 DiT block 容器（WanModelWithMemory.blocks）
    block_container = None
    for attr in ['blocks', 'layers', 'transformer_blocks']:
        if hasattr(model, attr):
            block_container = getattr(model, attr)
            break
    if block_container is None:
        # Fallback：找最大 ModuleList
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 10:
                block_container = mod
                break

    if block_container is None:
        logger.warning("梯度检查点：未找到 DiT block 容器，跳过 patch")
        return 0

    for block in block_container:
        if not any(p.requires_grad for p in block.parameters()):
            # 完全冻结的 block 无需 checkpoint（节省计算）
            continue
        orig_forward = block.forward

        def _make_ckpt_fn(fn):
            @wraps(fn)
            def _ckpt_forward(
                x, e, seq_lens, grid_sizes, freqs,
                context, context_lens, dit_cond_dict=None,
            ):
                # use_reentrant=False：不能用 kwargs，dit_cond_dict 作为位置参数传入
                # （code_standards.md §1：gradient_checkpointing use_reentrant=False）
                return torch_checkpoint(
                    fn, x, e, seq_lens, grid_sizes, freqs,
                    context, context_lens, dit_cond_dict,
                    use_reentrant=False,
                )
            return _ckpt_forward

        block.forward = _make_ckpt_fn(orig_forward)
        patched += 1

    logger.info("梯度检查点：patch 了 %d 个 block（含 MemoryBlockWrapper）", patched)
    return patched


# ---------------------------------------------------------------------------
# VAE Encode 辅助
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_video_with_vae(
    vae,
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """读取 video.mp4 并用 VAE encode 为 latent。

    复现 train_lingbot_csgo.py encode_video 逻辑：
        - OpenCV 读帧 → resize → normalize [-1, 1] → [3, F, H, W]
        - VAE.encode([tensor]) → latent [16, lat_f, lat_h, lat_w]

    Args:
        vae:        Wan2_1_VAE 实例
        video_path: video.mp4 绝对路径
        height:     目标高度（480）
        width:      目标宽度（832）
        num_frames: 目标帧数（81）
        device:     GPU 设备

    Returns:
        latent: Tensor [16, lat_f, lat_h, lat_w]，bfloat16，在 device 上
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
        frames.append(frame_t)
    cap.release()

    # 末帧填充（防止视频帧数不足）
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    video_tensor = torch.stack(frames, dim=1)  # [3, F, H, W]

    latent = vae.encode([video_tensor.to(device)])[0]
    torch.cuda.empty_cache()
    return latent


@torch.no_grad()
def encode_first_frame_condition(
    vae,
    image_path: str,
    video_latent: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """编码第一帧条件输入 y（mask + first-frame VAE latent）。

    完全复现 train_lingbot_csgo.py prepare_y：
        y = concat([mask, vae.encode([first_frame + zeros])])
        形状：[20, lat_f, lat_h, lat_w]（4ch mask + 16ch VAE latent）

    Args:
        vae:          Wan2_1_VAE 实例
        image_path:   image.jpg 绝对路径（或从 video_path 取第一帧）
        video_latent: [16, lat_f, lat_h, lat_w]（仅用于获取 lat 维度）
        num_frames:   原始视频帧数（81）
        height:       视频高度（480）
        width:        视频宽度（832）
        device:       GPU 设备

    Returns:
        y: Tensor [20, lat_f, lat_h, lat_w]，bfloat16，在 device 上
    """
    lat_f = video_latent.shape[1]
    lat_h = video_latent.shape[2]
    lat_w = video_latent.shape[3]

    # 加载第一帧图像
    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    first_frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    first_frame = first_frame.unsqueeze(1)  # [3, 1, H, W]

    zeros = torch.zeros(3, num_frames - 1, height, width, device=device)
    vae_input = torch.cat([first_frame.to(device), zeros], dim=1)  # [3, F, H, W]
    y_latent = vae.encode([vae_input])[0]  # [16, lat_f, lat_h, lat_w]

    # 构建 mask（只有第一帧为 1）—— 与 image2video.py 完全对齐
    msk = torch.ones(1, num_frames, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:],
    ], dim=1)  # [1, num_frames+3, lat_h, lat_w]
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]  # [4, lat_f, lat_h, lat_w]

    y = torch.cat([msk, y_latent], dim=0)  # [20, lat_f, lat_h, lat_w]
    return y


@torch.no_grad()
def encode_text(t5, prompt: str, device: torch.device) -> List[torch.Tensor]:
    """用 T5 编码 text prompt，encode 完 offload T5 到 CPU 节省显存。

    Args:
        t5:     T5EncoderModel 实例
        prompt: 文字描述
        device: GPU 设备

    Returns:
        context: list of Tensor，文本嵌入（传给 model.forward() 的 context 参数）
    """
    t5.model.to(device)
    context = t5([prompt], device)
    t5.model.cpu()
    torch.cuda.empty_cache()
    return [c.to(device) for c in context]


# ---------------------------------------------------------------------------
# 核心训练 step
# ---------------------------------------------------------------------------

def training_step(
    model: nn.Module,
    batch: Dict,
    vae,
    t5,
    schedule: FlowMatchingSchedule,
    device: torch.device,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    nfp_loss_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """单个训练步骤。

    training_step 接收已经过 accelerator.unwrap_model() 的模型，
    在 main() 的训练循环中以 accelerator.unwrap_model(model) 调用。

    训练循环核心步骤：
        1. VAE encode video → latent [16, lat_f, lat_h, lat_w]
        2. encode 第一帧 → y [20, lat_f, lat_h, lat_w]（条件输入）
        3. encode text prompt → context
        4. 采样时间步 t，计算 sigma 和训练权重
        5. 计算 noisy_latent（flow matching 插值）
        6. 注册 forward hook 到最后一个 transformer block，捕获 hidden_states
        7. model.forward(memory_states=None)
        8. 移除 hook，取 hidden_states → nfp_head.forward() → pred_latent [B, z_dim]
        9. L_diffusion = MSE(pred[:, 1:], target[:, 1:]) * training_weight
        10. L_nfp：nfp_head.compute_loss(pred_latent, actual_latent)
        11. L_total = L_diffusion + nfp_loss_weight * L_nfp['total']

    Flow Matching 推导（Wan/LingBot 约定）：
        noisy_latent = (1 - sigma) * x_0 + sigma * noise
        target = noise - x_0          (velocity prediction)
        pred 近似 target，用 MSE 训练

    NFP Loss 推导（B-01 修复）：
        通过 PyTorch forward hook 捕获最后一个 transformer block 的输出 hidden_states，
        形状为 [B, L, dim]（B=1，L=lat_f*h_p*w_p，dim=5120）。
        hook 在 try/finally 中确保被移除，不影响其他调用。
        nfp_head.forward(hidden_states) → pred_latent [B, z_dim]
        actual_latent = video_latent.mean(dim=[-3,-2,-1]) → [16]，unsqueeze(0) → [1, 16]
        nfp_head.compute_loss(pred_latent, actual_latent, mse_weight=1.0, cosine_weight=1.0)
        nfp_loss_weight 在外层统一控制整体权重，compute_loss 内部不再额外加权（W-02 修复）。

    注意：训练时 memory_states=None，MemoryCrossAttention 不参与计算；
          memory bank 是推理时机制，训练时每个 clip 独立，无跨 clip 记忆。

    Args:
        model:            WanModelWithMemory（accelerator.unwrap_model 后的版本）
        batch:            collate_fn 返回的 batch dict
        vae:              Wan2_1_VAE 实例（encode 完自动 offload 到 CPU）
        t5:               T5EncoderModel 实例（encode 完自动 offload 到 CPU）
        schedule:         FlowMatchingSchedule 实例
        device:           当前 GPU 设备
        height:           视频高度（默认 480）
        width:            视频宽度（默认 832）
        num_frames:       每 clip 帧数（默认 81，由 args.num_frames 传入）
        nfp_loss_weight:  NFP loss 权重（L_total = L_diffusion + w * L_nfp）

    Returns:
        dict with keys: 'total', 'diffusion', 'nfp_total', 'nfp_mse', 'nfp_cosine'
    """
    # 每个 batch 含一个样本（batch_size=1 默认，与 train_lingbot_csgo.py 一致）
    video_paths = batch["video_paths"]
    image_paths = batch["first_frames"]
    prompts = batch["prompts"]
    dit_cond_dict = batch["dit_cond_dict"]

    # ---- Step 1: VAE encode video ----
    # decode + VAE encode 在训练 step 中完成，encode 完释放 video tensors
    video_latent = encode_video_with_vae(
        vae, video_paths[0], height, width, num_frames, device
    )  # [16, lat_f, lat_h, lat_w]，VAE encode 完后已 empty_cache

    lat_f = video_latent.shape[1]
    lat_h = video_latent.shape[2]
    lat_w = video_latent.shape[3]

    # seq_len：WanModel.forward() 接受的序列长度（patch_size=(1,2,2) 折叠后）
    # seq_len = lat_f * (lat_h // p_h) * (lat_w // p_w) = 21 * 30 * 52 = 32760
    patch_h, patch_w = 2, 2
    seq_len = lat_f * (lat_h // patch_h) * (lat_w // patch_w)

    # ---- Step 2: 编码第一帧条件输入 ----
    y = encode_first_frame_condition(
        vae, image_paths[0], video_latent, num_frames, height, width, device
    )  # [20, lat_f, lat_h, lat_w]，encode 完后已 empty_cache

    # ---- Step 3: 编码 text prompt（T5 encode 完自动 offload 到 CPU）----
    context = encode_text(t5, prompts[0], device)

    # ---- Step 4: 采样时间步 ----
    sigma, t_val, training_weight = schedule.sample_timestep()
    t = t_val.to(device).unsqueeze(0)  # [1]

    # ---- Step 5: Flow Matching 加噪 ----
    # noisy_latent = (1 - sigma) * x_0 + sigma * noise（线性插值）
    noise = torch.randn_like(video_latent)
    noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
    target = noise - video_latent  # velocity target（flow matching 约定）

    # ---- Step 6: 注册 forward hook 捕获最后一个 transformer block 的 hidden_states ----
    # B-01 修复：使用 hook 捕获真正的 transformer hidden_states [B, L, dim]，
    # 而非 velocity prediction pred 的 mean，确保 NFPHead 接收语义正确的输入。
    # model.blocks 是 WanModelWithMemory 的 transformer block 列表（ModuleList）。
    # 最后一个 block（model.blocks[-1]）输出 [B, L, dim]（MemoryBlockWrapper 或 WanAttentionBlock）。
    hidden_states_capture = []

    def _nfp_hook(module, input, output):
        # MemoryBlockWrapper.forward 返回 Tensor；WanAttentionBlock.forward 也返回 Tensor
        hidden_states_capture.append(output[0] if isinstance(output, tuple) else output)

    last_block = model.blocks[-1]
    hook_handle = last_block.register_forward_hook(_nfp_hook)

    # ---- Step 7: model forward（memory_states=None，不使用 memory bank）----
    # 训练时不注入 memory bank：每个 clip 独立采样，memory 是推理时跨 clip 的机制
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                [noisy_latent],
                t=t,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond_dict,
                memory_states=None,  # 训练时不注入 memory
            )[0]  # [16, lat_f, lat_h, lat_w]
    finally:
        # ---- Step 8: 移除 hook，确保不影响后续调用 ----
        hook_handle.remove()

    # ---- Step 8 (cont): 从 hook 捕获的 hidden_states 计算 NFP pred_latent ----
    # hidden_states shape: [B, L, dim]，其中 B=1，L=lat_f*h_p*w_p，dim=5120
    # NFPHead.forward 期望 [B, L, dim]，mean-pool over L → [B, z_dim]
    hidden_states = hidden_states_capture[0].float()  # [B, L, dim]

    # model 已经是 unwrap_model 的结果，直接访问 nfp_head
    pred_latent = model.nfp_head.forward(hidden_states)  # [B, z_dim]，z_dim=16

    # actual_latent：video latent 在 spatial 维度均值池化 → [B, z_dim]
    # video_latent: [16, lat_f, lat_h, lat_w]（无 batch 维，单样本）
    # mean over [-3,-2,-1] (lat_f, lat_h, lat_w) → [16]，unsqueeze(0) → [1, 16]
    actual_latent = video_latent.float().mean(dim=[-3, -2, -1]).unsqueeze(0)  # [1, z_dim]

    # ---- Step 9: Diffusion loss（排除第一帧，第一帧是条件输入）----
    # pred[:, 0] 对应条件帧（由 y 提供），不计入 loss（与 train_lingbot_csgo.py 一致）
    pred_rest = pred[:, 1:]      # [16, lat_f-1, lat_h, lat_w]
    target_rest = target[:, 1:]  # [16, lat_f-1, lat_h, lat_w]
    diffusion_loss = F.mse_loss(pred_rest.float(), target_rest.float()) * training_weight

    # ---- Step 10: NFP loss ----
    # W-02 修复：compute_loss 内部 mse_weight=1.0, cosine_weight=1.0，
    # 避免双重加权（外层已由 nfp_loss_weight 统一控制整体权重）。
    nfp_loss_dict = model.nfp_head.compute_loss(
        pred_latent=pred_latent,
        target_latent=actual_latent,
        mse_weight=1.0,    # W-02: 不在 compute_loss 内部额外缩放，由外层 nfp_loss_weight 控制
        cosine_weight=1.0, # W-02: 同上
    )
    nfp_loss = nfp_loss_dict['total']

    # ---- Step 11: 合并 loss ----
    total_loss = diffusion_loss + nfp_loss_weight * nfp_loss

    return {
        'total': total_loss,
        'diffusion': diffusion_loss.detach(),
        'nfp_total': nfp_loss.detach(),
        'nfp_mse': nfp_loss_dict['mse'].detach(),
        'nfp_cosine': nfp_loss_dict['cosine'].detach(),
    }


# ---------------------------------------------------------------------------
# Checkpoint 保存
# ---------------------------------------------------------------------------

def save_checkpoint(
    accelerator,
    model: nn.Module,
    output_dir: str,
    step: int,
) -> None:
    """保存 accelerate state + model state_dict（兼容非 DeepSpeed 推理）。

    双重保存策略：
        1. accelerator.save_state(checkpoint_dir) — 含 optimizer / lr_scheduler 状态（断点续训）
        2. model.save_16bit_model() 或 state_dict 保存 — 供推理加载（不依赖 DeepSpeed）

    Args:
        accelerator: Accelerator 实例
        model:       训练模型（accelerator.prepare 后的版本）
        output_dir:  checkpoint 根目录
        step:        当前训练步数
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("保存 checkpoint → %s", checkpoint_dir)

    accelerator.wait_for_everyone()

    # 1. accelerator.save_state（含 optimizer / lr_scheduler，用于断点续训）
    accelerator.save_state(checkpoint_dir)

    # 2. 保存 model state_dict（ZeRO-3 需要 save_16bit_model；否则用 unwrap）
    # 检测是否在 DeepSpeed 模式下
    if hasattr(model, 'save_16bit_model'):
        # ZeRO-3：所有 rank 参与，rank 0 写文件
        model.save_16bit_model(checkpoint_dir, "model_weights.bin")
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if hasattr(unwrapped, 'save_config'):
                unwrapped.save_config(checkpoint_dir)
            # 验证保存完整性
            saved_path = os.path.join(checkpoint_dir, "model_weights.bin")
            if os.path.exists(saved_path):
                sd = torch.load(saved_path, map_location="cpu", weights_only=True)
                n_empty = sum(1 for v in sd.values() if v.numel() == 0)
                logger.info(
                    "checkpoint 保存完成：%d 参数，%d 空张量 → %s",
                    len(sd), n_empty, checkpoint_dir,
                )
                if n_empty > 0:
                    logger.error("警告：%d 个参数为空张量！", n_empty)
                del sd
    else:
        # 非 DeepSpeed 或 ZeRO-1/2：main process 保存
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            sd = unwrapped.state_dict()
            torch.save(sd, os.path.join(checkpoint_dir, "model_weights.bin"))
            logger.info(
                "checkpoint 保存完成：%d 参数 → %s", len(sd), checkpoint_dir
            )
            del sd

    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# 验证 loop（可选，有 metadata_val 时启用）
# ---------------------------------------------------------------------------

def eval_loop(
    model: nn.Module,
    val_dataloader,
    vae,
    t5,
    schedule: FlowMatchingSchedule,
    accelerator,
    epoch: int,
    height: int,
    width: int,
    num_frames: int,
    nfp_loss_weight: float,
) -> float:
    """在验证集上跑一遍，返回平均 total loss。

    Args:
        model:            训练模型
        val_dataloader:   验证集 DataLoader
        vae, t5:          编码器
        schedule:         FlowMatchingSchedule
        accelerator:      Accelerator 实例
        epoch:            当前 epoch（用于日志）
        height, width, num_frames, nfp_loss_weight: 同 training_step

    Returns:
        avg_loss: float，验证集平均 total loss
    """
    model.eval()
    total_loss_sum = 0.0
    num_batches = 0
    device = accelerator.device

    with torch.no_grad():
        for batch in val_dataloader:
            try:
                loss_dict = training_step(
                    model=accelerator.unwrap_model(model),
                    batch=batch,
                    vae=vae,
                    t5=t5,
                    schedule=schedule,
                    device=device,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    nfp_loss_weight=nfp_loss_weight,
                )
                total_loss_sum += loss_dict['total'].item()
                num_batches += 1
            except Exception as e:
                logger.warning("eval step 异常跳过: %s", e)
                continue

    avg_loss = total_loss_sum / max(num_batches, 1)
    if accelerator.is_main_process:
        logger.info("Epoch %d eval: avg_total_loss=%.4f（%d batches）", epoch + 1, avg_loss, num_batches)

    model.train()
    return avg_loss


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    """训练入口函数。

    训练循环结构：
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    1. VAE encode video
                    2. 编码第一帧图像（条件输入）
                    3. 编码 text prompt（T5）
                    4. 采样时间步 t
                    5. 计算 noisy latent
                    6. model forward（memory_states=None）
                    7. L_diffusion + L_nfp
                    8. backward + optimizer step
            if metadata_val: eval_loop
            save checkpoint
    """
    args = parse_args()

    # ---- Accelerator 初始化 ----
    import accelerate
    from accelerate import Accelerator
    from accelerate.utils import DataLoaderConfiguration

    accelerator_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    if args.deepspeed_config:
        from accelerate.utils import DeepSpeedPlugin
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config)
        accelerator_kwargs["deepspeed_plugin"] = deepspeed_plugin

    accelerator = Accelerator(**accelerator_kwargs)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("=" * 60)
        logger.info("LingBot-World Memory Enhancement 训练")
        logger.info("Stage: %d | Epochs: %d | LR: %.2e", args.stage, args.num_epochs, args.lr)
        logger.info("输出目录: %s", args.output_dir)
        logger.info("=" * 60)

    # ---- 加载辅助模型（VAE + T5）----
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.modules.t5 import T5EncoderModel

    logger.info("加载 VAE ...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"),
        device=accelerator.device,
    )

    logger.info("加载 T5 文本编码器 ...")
    t5 = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=accelerator.device,
        checkpoint_path=os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.ckpt_dir, "google", "umt5-xxl"),
    )

    # ---- 加载训练模型 ----
    model = load_model(args.ckpt_dir, accelerator.device)

    # ---- 参数冻结与 Optimizer ----
    if args.stage == 1:
        trainable_params = freeze_for_stage1(model)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    elif args.stage == 2:
        # PENDING[D-03]: Stage2 起点权重尚未确定（decisions.md D-03 OPEN）。
        # 当前假设：用 lingbot-world 预训练权重作为 Stage2 起点（选项A）。
        # 若 D-03 解除为选项B（对方的 CSGO-DiT 权重），需要在此处：
        #   1. 将下面的 load_model(args.ckpt_dir, ...) 替换为加载对方提供的 CSGO-DiT 权重
        #   2. 然后 load Stage1 checkpoint 来恢复 memory 模块参数
        # 具体操作示例（D-03 解除后填入）：
        #   csgo_dit_ckpt = "<对方提供的 CSGO-DiT checkpoint 路径>"
        #   base_model = WanModel.from_pretrained(csgo_dit_ckpt, ...)
        #   model = WanModelWithMemory.from_wan_model(base_model)
        #   stage1_sd = torch.load(args.resume_from + "/model_weights.bin", ...)
        #   model.load_state_dict(stage1_sd, strict=False)  # 只覆盖 memory 参数
        # 当前实现：直接对已加载的 lingbot-world 模型做全参数解冻
        _, _, optimizer = setup_stage2_optimizer(
            model=model,
            lr_memory=args.lr,
            lr_dit=args.lr_dit,
            weight_decay=args.weight_decay,
        )

    # ---- 梯度检查点 ----
    enable_gradient_checkpointing(model)

    # ---- Flow Matching Schedule ----
    schedule = FlowMatchingSchedule(
        num_train_timesteps=1000,
        shift=10.0,     # LingBot/Wan schedule shift（来自 train_lingbot_csgo.py）
        boundary=0.947, # low_noise_model 有效训练范围上界
    )

    # ---- DataLoader ----
    # 延迟导入（dataloader.py 依赖 lingbot-world cam_utils，需要 sys.path 就绪）
    # train.py 和 dataloader.py 同在 pipeline/ 包下，使用相对导入
    from dataloader import build_dataloader

    train_dataloader = build_dataloader(
        metadata_csv=args.metadata_train,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        height=args.height,
        width=args.width,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = None
    if args.metadata_val is not None:
        val_dataloader = build_dataloader(
            metadata_csv=args.metadata_val,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            height=args.height,
            width=args.width,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("验证集已加载，每个 epoch 结束后跑 eval loop")

    # ---- LR Scheduler ----
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * num_update_steps_per_epoch,
        eta_min=1e-6,
    )

    # ---- Accelerator prepare ----
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # ---- 恢复训练（断点续训）----
    global_step = 0
    start_epoch = 0
    if args.resume_from is not None:
        logger.info("从 checkpoint 恢复训练: %s", args.resume_from)
        accelerator.load_state(args.resume_from)
        # 估算已训练 step 数（从目录名解析）
        resume_tag = os.path.basename(args.resume_from)
        if resume_tag.startswith("checkpoint-"):
            try:
                global_step = int(resume_tag.split("-")[1])
                start_epoch = global_step // max(len(train_dataloader), 1)
                logger.info("恢复训练：global_step=%d，start_epoch=%d", global_step, start_epoch)
            except (ValueError, IndexError):
                logger.warning("无法从目录名解析 global_step，从 step=0 开始计数")

    # ---- 训练循环 ----
    logger.info(
        "开始训练：epochs=%d，steps/epoch=%d，gradient_accumulation=%d",
        args.num_epochs, len(train_dataloader), args.gradient_accumulation_steps,
    )

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_diffusion_loss_sum = 0.0
        epoch_nfp_loss_sum = 0.0
        num_batches = 0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                try:
                    loss_dict = training_step(
                        model=accelerator.unwrap_model(model),
                        batch=batch,
                        vae=vae,
                        t5=t5,
                        schedule=schedule,
                        device=accelerator.device,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,  # B-02: 使用 args.num_frames 替换硬编码 81
                        nfp_loss_weight=args.nfp_loss_weight,
                    )
                    total_loss = loss_dict['total']

                    # ---- OOM 防御（code_standards.md §2）----
                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
                        accelerator.clip_grad_norm_(trainable_params_list, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning(
                        "OOM at step %d (epoch=%d, batch=%d)，batch_size=%d。"
                        "建议减小 batch_size 或 height/width。",
                        global_step, epoch, batch_idx, args.batch_size,
                    )
                    raise  # 不静默吞掉（code_standards.md §2）

            # ---- 累计统计 ----
            epoch_loss_sum += loss_dict['total'].item()
            epoch_diffusion_loss_sum += loss_dict['diffusion'].item()
            epoch_nfp_loss_sum += loss_dict['nfp_total'].item()
            num_batches += 1
            global_step += 1

            # ---- 定期打印 loss ----
            if accelerator.is_main_process and global_step % args.log_every == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                elapsed = time.time() - t_epoch_start
                logger.info(
                    "step=%d | epoch=%d/%d | "
                    "loss=%.4f | diffusion=%.4f | nfp=%.4f | "
                    "lr=%.2e | elapsed=%.0fs",
                    global_step, epoch + 1, args.num_epochs,
                    loss_dict['total'].item(),
                    loss_dict['diffusion'].item(),
                    loss_dict['nfp_total'].item(),
                    current_lr,
                    elapsed,
                )

            # ---- 定期保存 checkpoint ----
            if global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, args.output_dir, global_step)

            # ---- dry_run：只跑 2 steps 验证流程 ----
            if args.dry_run and global_step >= 2:
                logger.info("dry_run 模式：2 steps 完成，退出训练")
                break

        # ---- Epoch 结束日志 ----
        avg_loss = epoch_loss_sum / max(num_batches, 1)
        avg_diff = epoch_diffusion_loss_sum / max(num_batches, 1)
        avg_nfp = epoch_nfp_loss_sum / max(num_batches, 1)
        epoch_time = time.time() - t_epoch_start

        if accelerator.is_main_process:
            logger.info(
                "Epoch %d/%d 完成 | avg_loss=%.4f | avg_diffusion=%.4f | "
                "avg_nfp=%.4f | time=%.0fs",
                epoch + 1, args.num_epochs,
                avg_loss, avg_diff, avg_nfp, epoch_time,
            )

        # ---- 验证 loop（可选）----
        if val_dataloader is not None:
            eval_loop(
                model=model,
                val_dataloader=val_dataloader,
                vae=vae,
                t5=t5,
                schedule=schedule,
                accelerator=accelerator,
                epoch=epoch,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,  # B-02: 使用 args.num_frames 替换硬编码 81
                nfp_loss_weight=args.nfp_loss_weight,
            )

        # ---- Epoch checkpoint ----
        save_checkpoint(accelerator, model, args.output_dir, global_step)

        if args.dry_run:
            break

    # ---- 最终保存 ----
    save_checkpoint(accelerator, model, args.output_dir, global_step)
    if accelerator.is_main_process:
        logger.info("训练完成！最终 checkpoint 已保存 → %s", args.output_dir)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
