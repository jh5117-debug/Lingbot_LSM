# LingBot-World Memory Enhancement

在 LingBot-World 上引入 Surprise-Driven Memory Bank，提升长视频时序一致性。

---

## 目录结构

```
Lingbot_LSM/
├── README.md
├── setup_env.sh
├── refs/
│   ├── lingbot-world/    # 基础模型（wan 包来源）
│   ├── WorldMem/         # 参考实现
│   └── cambrian-s/       # 参考实现
└── src/
    ├── infer.py          # 推理入口
    ├── configs/          # 训练/推理配置（预留）
    ├── memory_module/    # 核心记忆模块
    │   ├── memory_bank.py
    │   ├── memory_attention.py
    │   ├── nfp_head.py
    │   └── model_with_memory.py
    └── tests/
        └── smoke_test.py
```

---

## 快速开始：验证环境

运行冒烟测试（CPU 模式，无需 CUDA）：

```bash
cd /u/lwu9/Memory/projects/Lingbot_LSM
python src/tests/smoke_test.py --dry_run
```

测试覆盖：
- `MemoryBank`：update / retrieve 接口
- `NFPHead`：forward + compute_surprise
- `MemoryCrossAttention`、`MemoryBlockWrapper`、`WanModelWithMemory`：需要 CUDA，无 GPU 时自动 skip

---

## 快速推理：infer.py 最简用法

```bash
python src/infer.py \
    --ckpt_dir /path/to/lingbot-world-ckpt \
    --image /path/to/first_frame.png \
    --action_path /path/to/actions.json \
    --dry_run
```

参数说明：
- `--ckpt_dir`：LingBot-World 预训练权重目录
- `--image`：起始帧图片路径
- `--action_path`：动作序列 JSON 文件
- `--num_clips N`：生成 N 个 clip（默认 1），多 clip 时自回归衔接
- `--dry_run`：跳过实际模型推理，仅验证数据流和接口

---

## 参考库说明

| 目录 | 来源 | 取用范围 |
|------|------|---------|
| `lingbot-world/` | LingBot-World（基于 Wan2.2 的 action-conditioned world model） | baseline 架构、预训练权重格式、camera/action 控制信号注入方式（只读，不修改） |
| `WorldMem/` | WorldMem（arxiv: 2504.12369）| Memory Bank 数据结构（VAE latent + C2W pose）、FOV overlap confidence 检索机制、cross-attention 注入方式（只读） |
| `cambrian-s/` | Cambrian-S（arxiv: 2511.04670）| NFP head 结构、Surprise score 计算方式（cosine distance）、三级 KV cache 记忆管理逻辑（只读） |
