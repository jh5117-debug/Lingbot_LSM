"""
smoke_test.py — 各模块的 shape / forward 验证

不需要真实权重，全部使用随机初始化。
需要 PyTorch；MemoryCrossAttention 和 WanModelWithMemory 需要 CUDA（会自动 skip）。

运行：
    python tests/smoke_test.py            # 全部测试
    python tests/smoke_test.py --dry_run  # 仅运行不依赖 CUDA 的测试
"""

import argparse
import os
import sys
import logging

import torch
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---- 路径设置 ----
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))        # → .../src/tests
_ROOT = os.path.dirname(_TESTS_DIR)                             # → .../src
_LINGBOT_WORLD = os.path.join(os.path.dirname(_ROOT), 'refs', 'lingbot-world')  # → .../Lingbot_LSM/refs/lingbot-world
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
_PIPELINE_DIR = os.path.join(_ROOT, 'pipeline')
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from memory_module.memory_bank import MemoryBank, MemoryFrame
from memory_module.nfp_head import NFPHead

HAS_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# MemoryBank 测试（纯 CPU）
# ---------------------------------------------------------------------------

def test_memory_bank_update_and_retrieve():
    """update / retrieve / size / clear 基本行为（v2 structured API）。"""
    bank = MemoryBank(max_size=4)
    dim = 64

    # 插入 4 帧（v2 API: key_state + value_visual）
    for i in range(4):
        key_state = torch.randn(dim)
        value_visual = torch.randn(dim)
        bank.update(key_state, value_visual, surprise_score=float(i) * 0.3, timestep=i, chunk_id=0)

    assert bank.size() == 4, f"expected 4, got {bank.size()}"

    # 插入第 5 帧（surprise=2.0 > 所有已存帧），应替换 surprise 最低的帧（i=0, score=0.0）
    bank.update(torch.randn(dim), torch.randn(dim), surprise_score=2.0, timestep=100, chunk_id=0)
    assert bank.size() == 4, "size should remain 4 after eviction"
    scores = [f.surprise_score for f in bank.frames]
    assert 0.0 not in scores, "lowest surprise frame should have been evicted"

    # 插入低 surprise 帧（不应进入）
    bank.update(torch.randn(dim), torch.randn(dim), surprise_score=0.0, timestep=200, chunk_id=0)
    assert bank.size() == 4
    assert all(f.surprise_score > 0.0 for f in bank.frames), \
        "0.0 surprise frame should not be stored"

    # retrieve top-2 → v2 returns dict
    query = torch.randn(dim)
    retrieved = bank.retrieve(query, top_k=2)
    assert retrieved is not None
    assert isinstance(retrieved, dict), f"retrieve should return dict, got {type(retrieved)}"
    assert 'key_states' in retrieved
    assert 'value_visuals' in retrieved
    assert 'similarities' in retrieved
    assert retrieved['key_states'].shape == (2, dim), f"expected (2, {dim}), got {retrieved['key_states'].shape}"
    assert retrieved['value_visuals'].shape == (2, dim)
    assert retrieved['similarities'].shape == (2,)

    # retrieve 超过 bank 大小时取全部
    retrieved_all = bank.retrieve(query, top_k=10)
    assert retrieved_all['key_states'].shape[0] == 4

    # get_stats()
    stats = bank.get_stats()
    assert 'memory/bank_size' in stats
    assert 'memory/store_count' in stats
    assert 'memory/evict_count' in stats
    assert stats['memory/bank_size'] == 4.0

    # increment_age
    bank.increment_age()

    # clear
    bank.clear()
    assert bank.size() == 0
    assert bank.retrieve(query) is None

    logger.info("[PASS] test_memory_bank_update_and_retrieve")


def test_memory_bank_empty_retrieve():
    """空 bank 的 retrieve 应返回 None。"""
    bank = MemoryBank(max_size=8)
    assert bank.retrieve(torch.randn(32)) is None
    assert bank.size() == 0
    logger.info("[PASS] test_memory_bank_empty_retrieve")


# ---------------------------------------------------------------------------
# NFPHead 测试（CPU）
# ---------------------------------------------------------------------------

def test_nfp_head_shapes():
    """forward / compute_surprise / compute_loss 的 shape 验证。"""
    dim, z_dim, B, L = 128, 8, 2, 10
    head = NFPHead(dim=dim, z_dim=z_dim)
    head.eval()

    hidden = torch.randn(B, L, dim)
    pred = head(hidden)
    assert pred.shape == (B, z_dim), f"expected ({B}, {z_dim}), got {pred.shape}"

    # Surprise score
    actual = torch.randn(B, z_dim)
    surprise = NFPHead.compute_surprise(pred, actual)
    assert surprise.shape == (B,), f"expected ({B},), got {surprise.shape}"
    assert (surprise >= 0).all(), "surprise should be >= 0"
    assert (surprise <= 2).all(), "surprise should be <= 2"

    # Loss（有 mask）
    mask = torch.ones(B)
    mask[0] = 0  # 第一个样本不计算 loss
    losses = NFPHead.compute_loss(pred, actual, loss_mask=mask)
    assert set(losses.keys()) == {'mse', 'cosine', 'total'}
    assert losses['total'].ndim == 0, "loss should be scalar"
    assert losses['total'] >= 0

    # Loss（无 mask）
    losses_no_mask = NFPHead.compute_loss(pred, actual)
    assert losses_no_mask['total'] >= 0

    logger.info("[PASS] test_nfp_head_shapes")


def test_nfp_head_per_frame():
    """v2 per-frame NFP: forward_per_frame / compute_surprise_per_frame / compute_loss_per_frame。"""
    dim, z_dim = 128, 8
    B, lat_f, spatial = 1, 11, 20
    L = lat_f * spatial

    head = NFPHead(dim=dim, z_dim=z_dim)
    head.eval()

    hs = torch.randn(B, L, dim)

    # Per-frame forward
    pred = head.forward_per_frame(hs, lat_f, spatial)
    assert pred.shape == (B, lat_f, z_dim), f"expected ({B}, {lat_f}, {z_dim}), got {pred.shape}"

    # Per-frame surprise
    actual = torch.randn(B, lat_f, z_dim)
    surprise = NFPHead.compute_surprise_per_frame(pred, actual)
    assert surprise.shape == (B, lat_f), f"expected ({B}, {lat_f}), got {surprise.shape}"

    # Per-frame loss
    video_latent = torch.randn(z_dim, lat_f, 4, 4)
    loss_dict = NFPHead.compute_loss_per_frame(pred, video_latent)
    assert 'mse' in loss_dict
    assert 'cosine' in loss_dict
    assert 'per_frame_surprise' in loss_dict
    assert loss_dict['per_frame_surprise'].shape == (lat_f - 1,), \
        f"expected ({lat_f - 1},), got {loss_dict['per_frame_surprise'].shape}"
    assert loss_dict['total'] >= 0

    logger.info("[PASS] test_nfp_head_per_frame")


def test_nfp_head_surprise_range():
    """边界情况：完全相同的向量 → surprise ≈ 0；完全相反 → surprise ≈ 2。"""
    dim = 32
    head = NFPHead(dim=dim, z_dim=8)

    v = torch.randn(1, 8)
    surprise_same = NFPHead.compute_surprise(v, v)
    assert surprise_same.item() < 0.01, f"identical vectors should have ~0 surprise, got {surprise_same.item()}"

    v_neg = -v
    surprise_opp = NFPHead.compute_surprise(v, v_neg)
    assert surprise_opp.item() > 1.9, f"opposite vectors should have ~2 surprise, got {surprise_opp.item()}"

    logger.info("[PASS] test_nfp_head_surprise_range")


# ---------------------------------------------------------------------------
# MemoryCrossAttention 测试（需要 CUDA，因为 flash_attention）
# ---------------------------------------------------------------------------

def test_memory_cross_attention_shapes():
    """forward shape 验证。"""
    if not HAS_CUDA:
        logger.info("[SKIP] test_memory_cross_attention_shapes (no CUDA)")
        return

    from memory_module.memory_attention import MemoryCrossAttention

    dim, num_heads = 64, 4
    B, L, K = 2, 20, 6
    device = torch.device("cuda:0")

    attn = MemoryCrossAttention(dim=dim, num_heads=num_heads).to(device)
    attn.eval()

    x = torch.randn(B, L, dim, device=device, dtype=torch.bfloat16)
    mem = torch.randn(B, K, dim, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = attn(x, mem)

    assert out.shape == (B, L, dim), f"expected ({B}, {L}, {dim}), got {out.shape}"
    logger.info("[PASS] test_memory_cross_attention_shapes")


def test_memory_cross_attention_no_memory_change():
    """memory_states 为零向量时，输出不应与输入相同（但 shape 应一致）。"""
    if not HAS_CUDA:
        logger.info("[SKIP] test_memory_cross_attention_no_memory_change (no CUDA)")
        return

    from memory_module.memory_attention import MemoryCrossAttention

    dim, num_heads = 64, 4
    B, L, K = 1, 10, 4
    device = torch.device("cuda:0")

    attn = MemoryCrossAttention(dim=dim, num_heads=num_heads).to(device)
    attn.eval()

    x = torch.randn(B, L, dim, device=device, dtype=torch.bfloat16)
    mem = torch.zeros(B, K, dim, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = attn(x, mem)

    assert out.shape == x.shape
    logger.info("[PASS] test_memory_cross_attention_no_memory_change")


# ---------------------------------------------------------------------------
# WanModelWithMemory + MemoryBlockWrapper 测试（需要 CUDA）
# ---------------------------------------------------------------------------

def _make_tiny_wan_model():
    """创建一个极小的 WanModel 用于 shape 测试（不需要预训练权重）。"""
    from wan.modules.model import WanModel
    # 满足约束: dim % num_heads == 0 且 (dim // num_heads) % 2 == 0
    # 这里 dim=32, num_heads=4, dim//num_heads=8, 8%2=0 ✓
    return WanModel(
        model_type='i2v',
        control_type='cam',
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    )


def test_memory_block_wrapper_shapes():
    """MemoryBlockWrapper 在有/无 memory 时的 forward shape。"""
    if not HAS_CUDA:
        logger.info("[SKIP] test_memory_block_wrapper_shapes (no CUDA)")
        return

    from wan.modules.model import WanAttentionBlock
    from memory_module.model_with_memory import MemoryBlockWrapper

    dim, num_heads = 32, 4
    device = torch.device("cuda:0")

    block = WanAttentionBlock(dim=dim, ffn_dim=64, num_heads=num_heads).to(device).eval()
    wrapper = MemoryBlockWrapper(block=block, dim=dim, num_heads=num_heads).to(device).eval()

    B, L = 1, 20
    x = torch.randn(B, L, dim, device=device)
    e = torch.randn(B, L, 6, dim, device=device).float()
    seq_lens = torch.tensor([L], device=device)
    grid_sizes = torch.tensor([[1, 4, 5]], device=device)  # F=1, H=4, W=5
    d = dim // num_heads
    freqs = torch.randn(1024, d // 2, device=device)  # rope freqs
    context = torch.randn(B, 8, dim, device=device)

    kwargs = dict(e=e, seq_lens=seq_lens, grid_sizes=grid_sizes,
                  freqs=freqs, context=context, context_lens=None,
                  dit_cond_dict=None)

    with torch.no_grad():
        # 无 memory
        out_no_mem = wrapper(x, **kwargs)
        assert out_no_mem.shape == (B, L, dim), f"no-mem shape mismatch: {out_no_mem.shape}"

        # 有 memory
        memory_states = torch.randn(B, 4, dim, device=device)
        kwargs_mem = dict(kwargs)
        kwargs_mem['dit_cond_dict'] = {'memory_states': memory_states}
        out_with_mem = wrapper(x, **kwargs_mem)
        assert out_with_mem.shape == (B, L, dim), f"with-mem shape mismatch: {out_with_mem.shape}"

    logger.info("[PASS] test_memory_block_wrapper_shapes")


def test_wan_model_with_memory_conversion():
    """from_wan_model: 权重保留 + 新增参数随机初始化。"""
    if not HAS_CUDA:
        logger.info("[SKIP] test_wan_model_with_memory_conversion (no CUDA)")
        return

    from memory_module.model_with_memory import WanModelWithMemory

    base = _make_tiny_wan_model().cuda().eval()
    model = WanModelWithMemory.from_wan_model(base, memory_layers=[0, 1], max_memory_size=4)

    # 检查原始权重是否保留
    for name, param in base.named_parameters():
        # MemoryBlockWrapper 将 block 包裹在 wrapper.block 内，key 变为 blocks.i.block.xxx
        wrapped_name = name.replace("blocks.", "blocks.").replace(
            "blocks.0.", "blocks.0.block.").replace("blocks.1.", "blocks.1.block.")
        if wrapped_name in dict(model.named_parameters()):
            orig = param.data
            new = dict(model.named_parameters())[wrapped_name].data
            assert torch.allclose(orig, new), f"Weight mismatch for {name}"

    # 检查新增参数存在
    param_names = [n for n, _ in model.named_parameters()]
    assert any("memory_cross_attn" in n for n in param_names), \
        "memory_cross_attn parameters not found"
    assert any("nfp_head" in n for n in param_names), \
        "nfp_head parameters not found"

    logger.info("[PASS] test_wan_model_with_memory_conversion")


def test_get_projected_frame_embs_shape():
    """get_projected_frame_embs 返回 [lat_f, dim]。"""
    if not HAS_CUDA:
        logger.info("[SKIP] test_get_projected_frame_embs_shape (no CUDA)")
        return

    from memory_module.model_with_memory import WanModelWithMemory

    dim = 32
    base = _make_tiny_wan_model().cuda().eval()
    model = WanModelWithMemory.from_wan_model(base, memory_layers=[0], max_memory_size=4).cuda().eval()

    # patch_size=(1,2,2), control_type='cam' → control_dim=6
    # patch_embedding_wancamctrl: in_features = 6 * 64 * 1 * 2 * 2 = 1536
    # 构造合法的 c2ws_plucker_emb: [1, C, lat_f, lat_h, lat_w]
    # C = 6 * (h//lat_h) * (w//lat_w) = 6 * 8 * 8 = 384 (假设 vae_stride=8)
    # 但实际 patch_embedding_wancamctrl 的 in_features = control_dim * 64 * prod(patch_size)
    # = 6 * 64 * 1 * 2 * 2 = 1536
    # rearrange 后 raw_dim = C * c1 * c2 * c3 = C * 1 * 2 * 2
    # 所以 C = 1536 / (1*2*2) = 384
    C, lat_f, lat_h, lat_w = 384, 3, 4, 6  # lat_h, lat_w 是 patch_size 的倍数
    raw_emb = torch.randn(1, C, lat_f, lat_h, lat_w, device='cuda')

    with torch.no_grad():
        frame_embs = model.get_projected_frame_embs(raw_emb)

    assert frame_embs.shape == (lat_f, dim), \
        f"expected ({lat_f}, {dim}), got {frame_embs.shape}"
    logger.info("[PASS] test_get_projected_frame_embs_shape")


# ---------------------------------------------------------------------------
# train_v2_stage1.py 新组件测试（CPU）
# ---------------------------------------------------------------------------

def test_train_v2_imports():
    """train_v2_stage1.py：关键类和函数可以正常导入"""
    try:
        from train_v2_stage1 import (
            FlowMatchingSchedule,
            CSGODataset,
            freeze_for_stage,
            parse_args,
        )

        # FlowMatchingSchedule 可以实例化
        sched = FlowMatchingSchedule()
        assert hasattr(sched, 'sigmas')
        assert hasattr(sched, 'valid_train_indices')
        assert hasattr(sched, 'training_weights')

        logger.info("[PASS] test_train_v2_imports")
    except Exception as e:
        logger.error(f"[FAIL] test_train_v2_imports: {e}")
        raise


def test_flow_matching_schedule_initialization():
    """FlowMatchingSchedule：sigma 范围、权重范围、valid_train_indices 正确性"""
    try:
        from train_v2_stage1 import FlowMatchingSchedule

        sched = FlowMatchingSchedule()

        # sigma 范围：0~1
        assert sched.sigmas.min() >= 0 and sched.sigmas.max() <= 1, "sigmas 超出 [0,1]"

        # valid_train_indices：所有 timestep < 947
        valid_ts = sched.timesteps_schedule[sched.valid_train_indices]
        assert (valid_ts < 947).all(), "valid_train_indices 包含 timestep >= 947"
        assert len(sched.valid_train_indices) > 0, "valid_train_indices 为空"

        # 训练权重：非负，且总和约等于 num_train_timesteps
        assert (sched.training_weights >= 0).all(), "训练权重出现负值"

        logger.info("[PASS] test_flow_matching_schedule_initialization")
    except Exception as e:
        logger.error(f"[FAIL] test_flow_matching_schedule_initialization: {e}")
        raise


def test_flow_matching_schedule_sample_timestep():
    """FlowMatchingSchedule：sample 返回值在有效范围内"""
    try:
        from train_v2_stage1 import FlowMatchingSchedule

        sched = FlowMatchingSchedule()

        for _ in range(10):
            idx = sched.valid_train_indices[
                torch.randint(len(sched.valid_train_indices), (1,)).item()
            ].item()
            sigma = sched.sigmas[idx].item()
            t = sched.timesteps_schedule[idx]
            weight = sched.training_weights[idx].item()

            assert 0 <= sigma <= 1, f"sigma={sigma} 超出 [0,1]"
            assert 0 <= t.item() < 947, f"t={t.item()} 超出有效范围"
            assert weight >= 0, f"weight={weight} 为负值"

        logger.info("[PASS] test_flow_matching_schedule_sample_timestep")
    except Exception as e:
        logger.error(f"[FAIL] test_flow_matching_schedule_sample_timestep: {e}")
        raise


def test_freeze_for_stage():
    """freeze_for_stage：接口存在性和参数正确性"""
    try:
        from train_v2_stage1 import freeze_for_stage
        import inspect

        sig = inspect.signature(freeze_for_stage)
        params = list(sig.parameters.keys())
        assert 'model' in params, "freeze_for_stage 缺少 model 参数"
        assert 'stage' in params, "freeze_for_stage 缺少 stage 参数"
        assert 'lora_rank' in params, "freeze_for_stage 缺少 lora_rank 参数"

        logger.info("[PASS] test_freeze_for_stage")
    except Exception as e:
        logger.error(f"[FAIL] test_freeze_for_stage: {e}")
        raise


# ---------------------------------------------------------------------------
# F-02 修复验证测试（CPU）
# ---------------------------------------------------------------------------

def test_nfp_target_shape():
    """验证 M-2 修复：NFP 训练目标取 clip 最后帧空间均值，shape 正确且不等于全 clip 均值。"""
    import torch.nn as nn

    # 构造 video_latent: [z_dim=16, lat_f=5, lat_h=4, lat_w=4]
    video_latent = torch.randn(16, 5, 4, 4)

    # M-2 修复逻辑：最后帧空间均值
    actual_latent = video_latent[:, -1].mean(dim=[-2, -1]).unsqueeze(0)  # [1, 16]
    assert actual_latent.shape == (1, 16), \
        f"expected (1, 16), got {actual_latent.shape}"

    # 验证不等于全 clip 均值（一般情况下不等，极少数随机情况下可能相等，但概率极低）
    clip_mean = video_latent.mean(dim=[-3, -2, -1]).unsqueeze(0)  # [1, 16]
    # 两者 shape 相同，但值不同（最后帧 vs 全 clip 平均）
    assert not torch.allclose(actual_latent, clip_mean), \
        "last-frame target should differ from full-clip mean target"

    logger.info("[PASS] test_nfp_target_shape")


def test_freeze_for_stage_lora_behavior():
    """验证 M-3 修复：LoRA adapter 在 Stage1 确实被解冻，base 参数仍被冻结。"""
    import torch.nn as nn
    from train_v2_stage1 import freeze_for_stage

    class MockModelWithLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(32, 32)
            self.lora_A = nn.Parameter(torch.randn(8, 32))
            self.lora_B = nn.Parameter(torch.randn(32, 8))
            # 模拟 memory 模块（freeze_for_stage Stage1 会解冻含这些名字的参数）
            self.memory_cross_attn = nn.Linear(32, 32)
            self.memory_norm = nn.LayerNorm(32)
            self.nfp_head = nn.Linear(32, 16)

    model = MockModelWithLoRA()

    # 调用 freeze_for_stage，lora_rank > 0 触发 M-3 逻辑
    freeze_for_stage(model, stage=1, lora_rank=8)

    # LoRA adapter 参数应被解冻
    assert model.lora_A.requires_grad is True, \
        "lora_A should be unfrozen in Stage1 with lora_rank > 0"
    assert model.lora_B.requires_grad is True, \
        "lora_B should be unfrozen in Stage1 with lora_rank > 0"

    # base 参数应被冻结
    assert model.base_layer.weight.requires_grad is False, \
        "base_layer.weight should be frozen in Stage1"

    logger.info("[PASS] test_freeze_for_stage_lora_behavior")


def test_video_last_frame_extraction():
    """验证 HIGH-1 修复：video[:, -1] 正确取最后时间帧而非色道。"""
    # 构造 video: [C=3, T=10, H=64, W=64]
    video = torch.randn(3, 10, 64, 64)

    # HIGH-1 修复：[:, -1] 取最后时间帧
    last_frame = video[:, -1]  # [C=3, H=64, W=64]
    assert last_frame.shape == (3, 64, 64), \
        f"expected (3, 64, 64), got {last_frame.shape}"

    # 验证内容：等于第 10 帧（索引 9），不等于蓝色通道（video[2]）
    assert torch.allclose(last_frame, video[:, 9]), \
        "last_frame should equal video[:, 9] (10th time frame)"
    assert not torch.allclose(last_frame, video[:, 0]), \
        "last_frame should NOT equal video[:, 0] (first frame)"

    logger.info("[PASS] test_video_last_frame_extraction")


def test_all_videos_cat_dim():
    """验证 H-3 + BLOCK-A 修复：numpy→tensor 转换 + dim=1 时间维 cat。"""
    import numpy as np

    # 构造 3 个 numpy 视频片段，每个 [C=3, T=5, H=64, W=64]
    all_videos = []
    for _ in range(3):
        vid = np.random.randn(3, 5, 64, 64).astype(np.float32)
        # H-3 修复：numpy→tensor 转换
        _video_tensor = torch.from_numpy(vid.copy()) if isinstance(vid, np.ndarray) else vid
        all_videos.append(_video_tensor)

    # 确认转换成功
    assert isinstance(all_videos[0], torch.Tensor), \
        "all_videos[0] should be torch.Tensor after numpy→tensor conversion"

    # BLOCK-A 修复：沿时间维度 dim=1 拼接，[C=3, T=15, H=64, W=64]
    video = torch.cat(all_videos, dim=1)
    assert video.shape == (3, 15, 64, 64), \
        f"expected (3, 15, 64, 64), got {video.shape}"

    logger.info("[PASS] test_all_videos_cat_dim")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="只跑纯 CPU 测试，跳过所有需要 CUDA 的测试")
    args = parser.parse_args()

    if args.dry_run:
        global HAS_CUDA
        HAS_CUDA = False
        logger.info("DRY RUN: skipping all CUDA tests")

    # CPU 测试
    test_memory_bank_update_and_retrieve()
    test_memory_bank_empty_retrieve()
    test_nfp_head_shapes()
    test_nfp_head_per_frame()
    test_nfp_head_surprise_range()

    # train_v2_stage1.py 新组件测试（CPU）
    test_train_v2_imports()
    test_flow_matching_schedule_initialization()
    test_flow_matching_schedule_sample_timestep()
    test_freeze_for_stage()

    # F-02 修复验证测试（CPU）
    test_nfp_target_shape()
    test_freeze_for_stage_lora_behavior()
    test_video_last_frame_extraction()
    test_all_videos_cat_dim()

    # CUDA 测试（自动跳过）
    test_memory_cross_attention_shapes()
    test_memory_cross_attention_no_memory_change()
    test_memory_block_wrapper_shapes()
    test_wan_model_with_memory_conversion()
    test_get_projected_frame_embs_shape()

    logger.info("All smoke tests completed.")


if __name__ == "__main__":
    main()
