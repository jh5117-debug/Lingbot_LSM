# LingBot-World Memory Enhancement

## 背景

LingBot-World 是基于 WanModel（DiT 架构）预训练的视频世界模型，能够根据相机姿态和动作序列生成第一人称视角视频（CS:GO 场景）。该模型在短视频生成上表现出色，但受限于固定的 context window，在生成超长视频时存在长期遗忘问题：一旦历史帧超出 context，模型无法维持场景一致性，导致画面崩溃。

本项目在 LingBot-World 预训练权重基础上，通过引入外部记忆模块，解决超长视频生成中的长期时序一致性问题，同时不破坏短视频生成质量，不显著增加显存开销。

---

## 问题描述

视频世界模型的长期遗忘问题体现在以下三个方面：

1. **连续性断裂**：相邻 chunk 之间缺少明确的衔接信息，生成帧与上一段末尾帧可能出现跳变
2. **动态事件遗忘**：高 surprise 的关键事件（如场景突变）超出 context 后不可恢复
3. **场景重访不一致**：相机离开某区域后再次回来，模型无法回忆该区域的视觉状态，导致地图外观不一致

现有方法（LingBot-World baseline）的局限：依赖 context window，显存随帧数线性增长，context 外历史信息完全丢失。

---

## 方案概述

本工作引入三个新增模块，在冻结预训练权重的前提下插入记忆机制：

1. **NFP Head**（Next Frame Prediction Head）：2 层 MLP，在 VAE latent 空间计算每帧的 Surprise score（= 1 - cosine_sim(预测 latent, 真实 latent)），量化"意外程度"，驱动记忆写入决策
2. **三层 Memory Bank**（Three-Tier Memory Bank）：固定容量的外部记忆存储，将历史帧按照连续性、意外性、稳定性分层管理，容量恒定不随视频长度增长
3. **Memory Cross-Attention**：插入 DiT 部分层，将从 Memory Bank 检索到的历史帧信息注入当前 chunk 的生成过程，K/V 分离路由（K 来自 pose_emb，V 来自 visual_emb）

训练策略采用两阶段微调：Stage 1 冻结 DiT 只训记忆模块，Stage 2 解冻全部参数联合优化。

---

## 系统架构

### 三层 Memory Bank 设计

三层各自负责不同时间尺度和语义类型的历史信息：

| 层 | 名称 | 容量 | 写入条件 | 检索策略 | 目的 |
|----|------|------|---------|---------|------|
| 短期 | ShortTermBank | 2 帧（FIFO） | 无条件，每个 chunk 结束后写入最近 2 帧 | 强制全部返回（2 帧） | 保证 chunk 间衔接连续性 |
| 中期 | MediumTermBank | 8 帧 | surprise > SURPRISE_THRESHOLD（"意外"帧） | pose_emb cosine similarity，top-3 | 记录近期动态事件 |
| 长期 | LongTermBank | 8–16 帧 | stable（低 surprise）AND novel（语义不重复）| semantic_key cosine similarity，top-2 | 支持场景重访一致性 |

### 各层职责

**ShortTermBank**：FIFO 队列，容量 2，无淘汰策略。每个 chunk 处理完成后强制写入最近 2 帧，覆盖最旧记录。检索时全部注入，不做相似度筛选。

**MediumTermBank**：容量 8，按有效分数淘汰：
- `effective_score = surprise × (0.5 ^ (age / half_life))`
- 每个 chunk 结束后对 medium 帧执行 `age += 1`
- 检索时按 pose_emb cosine similarity 返回 top-3

**LongTermBank**：容量 8–16，双重写入门槛：
- stable：`surprise < STABILITY_THRESHOLD`
- novel：`max(cosine_sim(semantic_key, 已存帧的 semantic_key)) < NOVELTY_THRESHOLD`
- 淘汰语义冗余度最高的帧（与其他帧 semantic_key 相似度最大者）
- 检索时按 semantic_key cosine similarity 返回 top-2

### 混合检索预算

三层固定预算合计 7 帧传入 MemoryCrossAttention：
- Short：2 帧（强制注入）
- Medium：top-3（pose_emb cosine sim）
- Long：top-2（semantic_key cosine sim）

### K/V 分离路由

Memory Cross-Attention 的 K 来自 pose_emb（相机视野路由，用于检索），V 来自 visual_emb（历史视觉内容，注入生成）。两路解耦后检索语义更精准，latent 视觉信息得以真正注入生成过程。

---

## 创新点

1. **首次将 Surprise-Driven Memory 引入视频生成 World Model**
   Cambrian-S 的 Surprise 机制仅在 MLLM VQA 任务验证；本工作将其迁移到 VAE latent 空间，通过 NFP Head（2 层 MLP）计算 per-frame Surprise score，驱动记忆写入决策。

2. **K/V 分离路由**
   Memory Cross-Attention 的 K 来自 pose_emb（相机视野路由），V 来自 visual_emb（历史视觉内容）。两路解耦后检索语义更精准，latent 视觉信息得以真正注入生成过程。

3. **三层 Memory Bank 架构（Three-Tier Memory Bank）**
   单一 surprise 驱动无法同时满足"保证连续性"、"记录意外事件"、"支持场景重访"三个需求。新设计将三个目标拆分为三个独立子银行（Short/Medium/Long），各自拥有独立的写入条件、容量和检索策略，互不干扰。

4. **Semantic Key：基于 K 投影空间的语义相似度**
   借鉴 HyDRA 中"用 attention 投影层作为特征提取器"的思路，设计专用于 pose 空间的语义距离度量：`semantic_key = cross_attn.norm_k(cross_attn.k(pose_emb)).detach()`（K·K 相似度，两侧都是 pose_emb 的 K 投影）。相比原始 pose_emb cosine similarity，在学到的检索特征空间中衡量场景语义距离，区分力更强，用于 LongTermBank 的 novelty check 和语义检索。

5. **混合检索预算（Hybrid Retrieval Budget）**
   三层各司其职，检索预算固定 7 帧（Short 2 + Medium 3 + Long 2）。相比当前 top-k=4 检索，语义更结构化：Short 保连续性，Medium 找相似视野的意外帧，Long 找同一地图区域的稳定帧。固定预算设计使 cross-attention 的 key/value 维度恒定，不随视频长度增长。

---

## 与 HyDRA 的关系

HyDRA（Out of Sight but Not Out of Mind）是处理视频中动态主体离场后重新出现一致性问题的工作，其核心是 DynamicRetrievalAttention，使用 Q·K 特征空间（video hidden state）计算时序相关性。本工作借鉴了 HyDRA "用 attention 投影层作为特征提取器"的思路，但设计方向不同：本工作针对静态场景的相机重访一致性，使用 K·K 相似度（pose_emb 投影）衡量场景语义距离，并结合三层 Memory Bank 按 surprise/stable/novel 分类存储，与 HyDRA 的单一 context memory 压缩方案有本质区别。

---

## 代码结构

```
Lingbot_LSM/
├── setup_env.sh
├── refs/                            # 参考代码库（只读）
└── src/
    ├── configs/
    │   ├── accelerate_stage1.yaml
    │   └── accelerate_stage2.yaml
    ├── memory_module/               # 模型组件
    │   ├── memory_bank.py           # 当前：单层 MemoryBank；下一步：ThreeTierMemoryBank
    │   ├── memory_attention.py      # Memory Cross-Attention（K/V 分离）
    │   ├── nfp_head.py              # NFP Head（Surprise score 计算）
    │   └── model_with_memory.py     # WanModelWithMemory（封装 DiT + 记忆模块）
    ├── pipeline/                    # 训练/推理入口
    │   ├── dataloader.py
    │   ├── train_v2_stage1.py       # v2 数据 + Stage1 + 单模型（已实现）
    │   ├── train_v2_stage1_dual.py  # v2 数据 + Stage1 + 双模型（已实现）
    │   ├── train_v2_stage2.py       # v2 数据 + Stage2 + 单模型（stub，PENDING D-03）
    │   ├── train_v2_stage2_dual.py  # v2 数据 + Stage2 + 双模型（stub，PENDING D-03）
    │   ├── train_v3_stage1_single.py# v3 数据 + Stage1 + 单模型（stub，等 v3 数据）
    │   ├── train_v3_stage1_dual.py  # v3 数据 + Stage1 + 双模型（stub，等 v3 数据）
    │   ├── train_v3_stage2_single.py# v3 数据 + Stage2 + 单模型（stub）
    │   ├── train_v3_stage2_dual.py  # v3 数据 + Stage2 + 双模型（stub，最终目标）
    │   ├── infer_v2.py              # v2 推理（已实现，支持 LoRA/全参/Memory Bank）
    │   └── infer_v3.py              # v3 推理（stub）
    ├── scripts/                     # 运行脚本
    │   ├── run_train_v2.sh
    │   ├── run_train_v2_dual.sh
    │   ├── run_train_v3.sh
    │   ├── run_infer_v2.sh
    │   └── run_infer_v3.sh
    └── tests/
        └── smoke_test.py
```

---

## 当前实现状态

### v2 已完成（2026-04-08）

| 组件 | 状态 | 说明 |
|------|------|------|
| memory_bank.py | 完成 | 单层 MemoryBank，max_size=8，Surprise 驱动 + pose_emb 检索 |
| memory_attention.py | 完成 | Memory Cross-Attention，K/V 分离（K←pose_emb，V←visual_emb） |
| model_with_memory.py | 完成 | WanModelWithMemory，封装 DiT + 记忆模块接口 |
| nfp_head.py | 完成 | 2 层 MLP，计算 per-frame Surprise score |
| dataloader.py | 完成 | metadata.csv + per-clip 目录结构，81 帧 clip 规格 |
| train_v2_stage1.py | 完成 | v2 数据 + Stage1 + 单模型 |
| train_v2_stage1_dual.py | 完成 | v2 数据 + Stage1 + 双模型（两模型均含 Memory Bank） |
| infer_v2.py | 完成 | v2 推理，支持 LoRA/全参/Memory Bank |
| smoke_test.py | 完成 | 含 FlowMatchingSchedule/freeze_for_stage 测试 |

### 下一步：三层 Memory Bank（待实现）

当前 memory_bank.py 为单层设计，待实现 `ThreeTierMemoryBank`，主要变更点：

- 新增 `ThreeTierMemoryBank` 类（包含 ShortTermBank / MediumTermBank / LongTermBank 三个子对象）
- 新增 `MemoryFrame.semantic_key [5120]` 字段（= `cross_attn.norm_k(cross_attn.k(pose_emb)).detach()`）
- `model_with_memory.py` 新增 `get_semantic_key()` 方法
- `infer_v2.py` / 训练脚本中更新 `bank.update()` 和 `bank.retrieve()` 调用签名
- `MemoryCrossAttention.forward()` 接口不变（已兼容 7 帧混合输入）

超参数（待调参）：

| 超参数 | 建议值 | 说明 |
|-------|-------|------|
| short_cap | 2 | 固定，不建议修改 |
| medium_cap | 8 | MediumTermBank 容量 |
| long_cap | 8–16 | LongTermBank 容量 |
| SURPRISE_THRESHOLD | 0.3–0.6 | Medium 写入下限 |
| STABILITY_THRESHOLD | 0.1–0.3 | Long 写入上限 |
| NOVELTY_THRESHOLD | 0.5–0.8 | Long 写入语义相似度上限 |
| half_life | 5–15 chunks | Medium age decay 半衰期 |

---

## 训练配置

### 两阶段训练策略

**Stage 1（冻结 DiT，只训记忆模块）**
- 训练参数：MemoryCrossAttention + NFPHead + memory_norm
- 学习率：1e-4
- 目标：验证记忆模块能正常工作，不破坏 DiT 已有生成能力

**Stage 2（解冻全部参数，联合微调）**
- DiT 学习率：1e-5，记忆模块学习率：1e-4
- 目标：全参数联合优化，提升长程一致性

### Loss 函数

```
L_total = L_diffusion + 0.1 × L_nfp
```

- `L_diffusion`：Flow Matching MSE，排除第一帧（`pred[:, 1:]` vs `target[:, 1:]`）；第一帧作为条件输入，不计入 loss
- `L_nfp`：NFP Head 预测 loss，权重 0.1（量级约 0.01，不主导训练）

### 训练配置矩阵（2×2×2=8 个配置）

| 编号 | 数据 | Stage | 方式 | 状态 |
|------|------|-------|------|------|
| ① | v2（4ch WASD） | Stage1 | 单模型 | 已实现 |
| ② | v2（4ch WASD） | Stage1 | 双模型 | 已实现 |
| ③ | v2（4ch WASD） | Stage2 | 单模型 | stub（PENDING D-03）|
| ④ | v2（4ch WASD） | Stage2 | 双模型 | stub（PENDING D-03）|
| ⑤ | v3（8ch，含 jump/crouch/fire/walk） | Stage1 | 单模型 | stub（等 v3 数据）|
| ⑥ | v3 | Stage1 | 双模型 | stub（等 v3 数据）|
| ⑦ | v3 | Stage2 | 单模型 | stub |
| ⑧ | v3 | Stage2 | 双模型 | stub（最终目标）|

**近期优先级**：先完成 ①②（v2 数据，Stage1），对比单模型 vs 双模型，验证 Memory Bank 有效性。

### 其他训练配置

- Flow Matching sigma schedule：shift=10.0，有效范围 max_t=0.947
- Optimizer：AdamW（lr=1e-4，weight_decay=0.01）
- LR Scheduler：CosineAnnealingLR
- 梯度裁剪：max_grad_norm=1.0；梯度累积：gradient_accumulation_steps=4
- 分布式：DeepSpeed ZeRO-3 + Accelerate（4/8 GPU）
- 资源：8x H20，FSDP + Ulysses

---

## 数据格式

```
{dataset_dir}/
├── metadata_train.csv    # 列：prompt, video, clip_path, map, episode_id, stem, num_frames
├── metadata_val.csv
└── train/clips/{stem}_clip{NNNN}/
    ├── video.mp4          # 16fps，480×832，81 帧
    ├── image.jpg          # 第一帧静态图（条件输入）
    ├── poses.npy          # [81, 4, 4]，float32，camera-to-world 矩阵
    ├── action.npy         # [81, 4]，int32，WASD 动作（0/1）
    ├── intrinsics.npy     # [81, 4]，float32，[fx, fy, cx, cy]
    └── prompt.txt
```

Clip 规格：81 帧（满足 VAE 约束 4n+1，n=20），480×832，16 FPS；latent 空间 21 帧 × 60 × 104。
