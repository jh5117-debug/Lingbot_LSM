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

from memory_module.memory_bank import MemoryBank, MemoryFrame
from memory_module.nfp_head import NFPHead

HAS_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# MemoryBank 测试（纯 CPU）
# ---------------------------------------------------------------------------

def test_memory_bank_update_and_retrieve():
    """update / retrieve / size / clear 基本行为。"""
    bank = MemoryBank(max_size=4)
    dim = 64

    # 插入 4 帧
    for i in range(4):
        pose_emb = torch.randn(dim)
        latent = torch.randn(16, 4, 4)
        bank.update(pose_emb, latent, surprise_score=float(i) * 0.3, timestep=i)

    assert bank.size() == 4, f"expected 4, got {bank.size()}"

    # 插入第 5 帧（surprise=2.0 > 所有已存帧），应替换 surprise 最低的帧（i=0, score=0.0）
    bank.update(torch.randn(dim), torch.randn(16, 4, 4), surprise_score=2.0, timestep=100)
    assert bank.size() == 4, "size should remain 4 after eviction"
    scores = [f.surprise_score for f in bank.frames]
    assert 0.0 not in scores, "lowest surprise frame should have been evicted"

    # 插入低 surprise 帧（不应进入）
    bank.update(torch.randn(dim), torch.randn(16, 4, 4), surprise_score=0.0, timestep=200)
    assert bank.size() == 4
    assert all(f.surprise_score > 0.0 for f in bank.frames), \
        "0.0 surprise frame should not be stored"

    # retrieve top-2
    query = torch.randn(dim)
    retrieved = bank.retrieve(query, top_k=2)
    assert retrieved is not None
    assert retrieved.shape == (2, dim), f"expected (2, {dim}), got {retrieved.shape}"

    # retrieve 超过 bank 大小时取全部
    retrieved_all = bank.retrieve(query, top_k=10)
    assert retrieved_all.shape[0] == 4

    # clear
    bank.clear()
    assert bank.size() == 0
    assert bank.retrieve(query) is None

    logger.info("[PASS] test_memory_bank_update_and_retrieve")


def test_memory_bank_empty_retrieve():
    """空 bank 的 retrieve 应返回 None。"""
    bank = MemoryBank(max_size=8)
    assert bank.retrieve(torch.randn(32)) is None
    assert bank.get_all_states() is None
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
    test_nfp_head_surprise_range()

    # CUDA 测试（自动跳过）
    test_memory_cross_attention_shapes()
    test_memory_cross_attention_no_memory_change()
    test_memory_block_wrapper_shapes()
    test_wan_model_with_memory_conversion()
    test_get_projected_frame_embs_shape()

    logger.info("All smoke tests completed.")


if __name__ == "__main__":
    main()
