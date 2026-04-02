# LingBot 超时空记忆修复评审与 W&B 训练观测方案

## 1. 结论摘要

当前代码还没有正确实现你想要的 `Cambrian-S Surprise + WorldMem Memory Bank` 闭环。

核心原因不是“模块没写”，而是“闭环没有打通”：

- 推理没有真正的跨 `chunk` 读写循环
- 训练阶段没有让 `memory_cross_attn` 真正参与 forward 并获得梯度
- 检索没有使用真实几何对齐的 `pose/state`
- Memory Bank 还没有把“视觉内容”和“状态”同时作为可读回的记忆
- `LFP + Surprise` 训练目标和推理写入策略没有闭环

所以，当前版本更接近：

- 已经搭好了 `MemoryBank / MemoryCrossAttention / NFPHead / WanModelWithMemory`
- 但是还没有变成一套可以稳定训练、稳定推理、稳定对比实验的完整系统

---

## 2. 当前阻塞问题

### P0. 推理没有形成跨 chunk 闭环

当前 `infer_v2.py` 只会：

1. 创建空的 `MemoryBank`
2. 在生成前尝试检索
3. 调用一次 `generate()`
4. 生成完成后更新 bank
5. 退出

这意味着：

- 第一段生成时 bank 为空
- 写入发生在最后
- 没有“下一段生成前读取上一段记忆”的过程

相关文件：

- `src/pipeline/infer_v2.py`
- `src/pipeline/infer_v3.py`

### P0. Stage1 实际没有训练到 memory attention

`train_v2_stage1.py` 里虽然解冻了：

- `memory_cross_attn`
- `memory_norm`
- `nfp_head`

但是训练 forward 明确传了：

```python
memory_states=None
```

而 `MemoryBlockWrapper` 里又是：

- 没有 `memory_states` 就直接跳过 memory 分支

所以 Stage1 实际上：

- `memory_cross_attn` 没参与计算图
- `memory_norm` 没参与计算图
- 真正能学到的基本只剩 `nfp_head`

相关文件：

- `src/pipeline/train_v2_stage1.py`
- `src/memory_module/model_with_memory.py`

### P0. 当前检索向量维度和语义都不对

`infer_v2.py` 现在写入 bank 的 `pose_emb` 是：

- `latent[:, t].mean(dim=(-2, -1))`

这其实是：

- `z_dim=16` 的 latent 通道均值

但 `MemoryCrossAttention` 需要的 `memory_states` 是：

- 模型隐藏维 `dim`

而且检索 query 当前还是全零向量，不是真实相机位姿或几何条件。

因此当前检索存在两个问题：

- 维度上不符合最终设计
- 语义上也不是“位姿对齐检索”

相关文件：

- `src/pipeline/infer_v2.py`
- `src/memory_module/memory_attention.py`
- `src/memory_module/model_with_memory.py`

### P1. Memory Bank 还不是 WorldMem 风格

现在 `MemoryFrame` 虽然存了：

- `pose_emb`
- `latent`
- `surprise_score`
- `timestep`

但 `retrieve()` 只返回：

- `pose_emb`

也就是说当前 memory attention 实际读回的是：

- 一个历史位姿代理向量

而不是：

- 历史视觉内容 + 历史状态

这和你想要的 WorldMem 设计仍然有差距。

相关文件：

- `src/memory_module/memory_bank.py`
- `src/memory_module/memory_attention.py`

### P1. Surprise 训练目标和推理写入策略未闭环

当前 `NFPHead` 的训练方式是：

- 把整段 hidden states 做 mean pool
- 预测整段视频 latent 的全局均值

但推理写入 memory 时使用的是：

- 相邻 latent 帧之间的 cosine distance

所以现在存在两个不一致：

- 训练学的是 clip-level proxy
- 推理用的是 frame-level heuristic

这并不能严格实现你想要的：

- “由 LFP 决定哪些帧值得写入”

相关文件：

- `src/memory_module/nfp_head.py`
- `src/pipeline/train_v2_stage1.py`
- `src/pipeline/infer_v2.py`

### P2. Cambrian-S 风格的事件分段和压缩没有实现

目前 Memory Bank 的策略只是：

- 容量满后替换最低 surprise

还没有：

- 低 surprise 片段压缩
- 段级 merge
- 惊奇驱动的事件分段
- 基于 segment 的写入预算

相关文件：

- `src/memory_module/memory_bank.py`

---

## 3. 修复优先级

## 3.1 P0：必须先修

### 任务 A：把推理改成真正的 chunked memory rollout

目标：

- 第一段生成后写入 memory
- 第二段生成前读取 memory
- 第三段继续沿用更新后的 memory
- 直到整段 episode 结束

建议：

- 不再只调用一次 `generate()`
- 引入显式 `chunk loop`
- 每个 chunk 都执行：
  - 准备当前 chunk 的相机条件
  - 计算 query pose/state
  - 从 bank 做 top-K 检索
  - 注入 `memory_states`
  - 生成当前 chunk
  - 计算该 chunk 的 surprise
  - 更新 bank

最低验收标准：

- 第 2 个 chunk 开始，`memory_bank.size() > 0`
- 第 2 个 chunk 开始，`memory_states is not None`
- W&B 和文本日志里能看到每个 chunk 的 bank size、retrieved K、retrieval similarity

### 任务 B：训练时必须让 memory attention 真进图

目标：

- Stage1 不只是训 `nfp_head`
- 至少要让 `memory_cross_attn` 在训练时参与 forward

建议方案：

- 训练中构造 teacher-forced memory bank
- 当前样本的前若干帧作为 memory source
- 当前样本后若干帧作为 query/target
- 用历史帧的 `pose_emb + visual content` 构建 `memory_states`

最低验收标准：

- `memory_cross_attn.q/k/v/o` 的 grad norm 非零
- `memory_norm` 的 grad norm 非零
- W&B 中可以看到单独的 `grad_norm/memory_cross_attn/*`

### 任务 C：把 pose query / memory key 统一到模型空间

目标：

- 写入和检索都使用同一种“模型空间中的位姿/状态嵌入”

建议：

- 不再使用 `latent.mean(...)` 作为 `pose_emb`
- 统一使用 `WanModelWithMemory.get_projected_frame_embs()`
- 写入时保存每帧 `frame_embs[t]`
- 检索时用当前 chunk 的 query frame emb 做 top-K

最低验收标准：

- `memory_states.shape[-1] == model.dim`
- `retrieve()` 的输入输出维度稳定
- 不再依赖零向量 query

## 3.2 P1：随后修

### 任务 D：把 Memory Bank 改成真正的 visual content + state

建议把单条 memory 拆成：

- `state_key`
  - pose embedding
  - timestamp
  - chunk index
  - optional action summary
- `value_visual`
  - 历史视觉 latent / visual tokens / pooled feature
- `meta`
  - surprise
  - age
  - source clip id

建议接口：

```python
bank.update(
    key_state=...,
    value_visual=...,
    surprise_score=...,
    timestep=...,
    chunk_id=...,
)

retrieved = bank.retrieve(...)
# 返回:
# {
#   "key_states": ...,
#   "value_visuals": ...,
#   "timesteps": ...,
#   "surprises": ...,
# }
```

### 任务 E：让 Surprise 变成真正的 per-frame 写入信号

建议：

- `NFPHead` 改成 per-frame 或 per-latent-frame 预测
- 目标不再是整段 clip 的 global mean latent
- 改成预测“下一 latent frame 的 pooled visual target”
- 再用 `1 - cosine_similarity` 作为 per-frame surprise

推荐最小实现：

- 从 `hidden_states` 重排回 `[B, lat_f, spatial_tokens, dim]`
- 对每个时间步单独 spatial pool
- 输出 `[B, lat_f, z_dim]`
- 用下一帧 latent pooled target 做监督

### 任务 F：加入段级事件分段与压缩

建议的最小版本：

- 连续低 surprise 帧先缓存在 short-term buffer
- 当遇到高 surprise 或段落结束时：
  - 将低 surprise 段压缩成 1 个 summary memory
  - 将高 surprise 关键帧单独写入 bank
- 满容量时不只是丢掉最低 surprise
  - 优先合并相邻且相似的低价值 memory

---

## 4. 逐文件修改建议

## 4.1 `src/pipeline/infer_v2.py`

当前问题：

- 只有单次生成
- query 是零向量
- 写入用 `latent.mean` 冒充 pose emb

建议修改：

- 新增 `generate_chunked_episode(...)`
- 按 chunk 组织 episode
- 如果 `action_path` 可提供 `poses/intrinsics`
  - 先计算 raw `c2ws_plucker_emb`
  - 再调用 `get_projected_frame_embs()`
- 用真实 query emb 做检索
- 将每个 chunk 的：
  - bank size
  - stored frames
  - evicted frames
  - retrieval top-k similarity
  - average surprise
  记录到日志和 W&B

建议新增函数：

- `_compute_chunk_query_emb(...)`
- `_update_memory_bank_from_projected_embs(...)`
- `generate_chunked_episode(...)`

## 4.2 `src/pipeline/infer_v3.py`

当前状态：

- `NotImplementedError`

建议：

- 先不要继续扩双模型版本
- 优先让 `infer_v2.py` 把 memory 闭环跑通
- 等单低噪声模型版本闭环稳定，再实现 v3

建议文档中明确：

- `v3` 暂不作为主训练/主推理入口
- 当前主线以 `v2 stage1/stage2 + chunked infer_v2` 为准

## 4.3 `src/pipeline/train_v2_stage1.py`

当前问题：

- `memory_states=None`
- `memory_cross_attn` 没有实际训练

建议修改：

- 引入训练期 `memory curriculum`
- 最开始可以做最小 teacher-forcing：
  - 前半段帧做 memory
  - 后半段帧做 query + diffusion target
- 分离损失项：
  - `loss/diffusion`
  - `loss/nfp_total`
  - `loss/nfp_mse`
  - `loss/nfp_cosine`
  - `loss/total`

进一步建议：

- 增加 `args.memory_train_top_k`
- 增加 `args.memory_train_burnin_frames`
- 增加 `args.enable_memory_training`

## 4.4 `src/memory_module/memory_bank.py`

当前问题：

- 只返回 `pose_emb`
- 没有 visual value
- 没有段级压缩

建议修改：

- `MemoryFrame` 扩展字段：
  - `key_state`
  - `value_visual`
  - `surprise_score`
  - `timestep`
  - `chunk_id`
  - `age`
- `retrieve()` 返回结构化结果而不是裸 tensor
- 增加：
  - `eviction_count`
  - `merge_count`
  - `store_count`
  - `reject_count`
- 将这些统计持续输出到日志/W&B

## 4.5 `src/memory_module/memory_attention.py`

建议修改：

- 支持 `key_states` 和 `value_visuals` 分离
- 不要再默认 `K=V=pose_emb`
- 增加可选 gate：

```python
x = x + gate * memory_cross_attn(...)
```

建议记录：

- attention output norm
- gate value mean/std
- memory attention active ratio

## 4.6 `src/memory_module/nfp_head.py`

建议修改：

- 从 clip-level 改成 frame-level
- 输出 `[B, lat_f, z_dim]`
- 添加：
  - `compute_surprise_per_frame`
  - `aggregate_surprise_to_segment`

建议记录：

- per-frame surprise 的 mean/std/max/min
- store threshold 以上的帧占比
- 被压缩帧占比

## 4.7 `src/memory_module/model_with_memory.py`

建议修改：

- `forward()` 支持结构化 memory dict
- `get_projected_frame_embs()` 作为唯一合法 pose emb 来源
- 增加断言：
  - `memory_states.shape[-1] == self.dim`
  - 如果传入了 memory dict，必须包含所需字段

## 4.8 `src/tests/smoke_test.py`

当前测试覆盖不够。

建议新增：

- `test_memory_states_dim_matches_model_dim`
- `test_train_step_memory_branch_has_grad`
- `test_chunked_infer_second_chunk_uses_non_empty_memory`
- `test_retrieve_returns_visual_value_and_state`
- `test_per_frame_surprise_shape`
- `test_zero_query_not_used_in_real_infer`

---

## 5. W&B 设计目标与修复要求

这一节是必须落实的，不是“可选增强”。

你的训练很可能会经历多轮失败、恢复、调参和重跑，所以：

- W&B 不是锦上添花
- 它必须是主观测系统

最终状态应满足以下要求。

### 5.1 最早初始化

GPU 任务一启动就尽快创建 W&B run。

要求：

- 在最早的训练入口初始化 run
- 早于模型大规模加载
- 早于 dataset 全量 warmup
- 早于第一个 optimizer step

原因：

- 这样才能记录启动前后的日志
- 如果模型加载阶段就 OOM/认证失败/挂死，也能在同一个 run 里看到上下文

建议：

- 在 `main()` 里 `parse_args()` 和 `accelerator` 初始化之后、`load_models()` 之前就创建 run
- 只允许主进程创建 run

### 5.2 W&B 不只出图表，也要同步文本日志

这里“W&B 输出所有 log”的准确理解应该是：

- 标量指标进入 charts
- 训练期文本日志同步到 run
- 异常、警告、环境信息、配置和 checkpoint 事件都能回查

建议同时保留三种输出：

1. 终端 stdout/stderr
2. 本地结构化日志文件
3. W&B run 内的同步文本日志

### 5.3 Crash 时把完整 SLURM 日志补回同一个 run

要求：

- 训练异常退出时，不要让 run 只剩半截 scalar
- 要把该次作业的完整 `slurm-*.out` 或 stdout/stderr 文件补传到同一个 run

建议：

- 训练脚本在 `try/except/finally` 中处理 run
- `finally` 阶段尝试：
  - flush 本地日志
  - `wandb.save()` 或 artifact 上传当前 job 的文本日志
- 如果检测到异常：
  - 给 run 打 `status=crashed`
  - 记录 traceback
  - 上传 SLURM 输出文件

### 5.4 认证与缓存路径必须显式修复

这一点要在脚本和文档里明确写成“最终状态要求”。

SLURM 脚本应统一把以下环境变量指到项目目录内：

- `WANDB_DIR`
- `WANDB_CACHE_DIR`
- `WANDB_DATA_DIR`
- `WANDB_CONFIG_DIR`

目的：

- 避免把缓存和 staging 文件写爆 home 配额

同时要求：

- 如果环境里没有现成的 `WANDB_API_KEY`
- 脚本自动尝试从 `~/.netrc` 解析并导出

目的：

- 避免只改了 `WANDB_CONFIG_DIR` 之后，节点端拿不到认证信息

这一点请在最终 SLURM 启动脚本中显式落实。

### 5.5 推荐的实现位置

建议新增一个独立模块：

- `src/scripts/wandb_utils.py`

至少包含：

- `init_wandb_run(...)`
- `log_text_file_to_wandb(...)`
- `log_crash_to_wandb(...)`
- `finish_wandb_run(...)`

同时新增一个统一启动脚本：

- `src/scripts/run_train_v2_slurm.sh`

负责：

- 设置 W&B 环境变量
- 解析/导出 `WANDB_API_KEY`
- 创建日志目录
- 调用 `accelerate launch`
- 在崩溃时补传日志

---

## 6. W&B 必须输出的图表与日志

这一节是为了满足你的实际需求：

- 一次大概率跑不成功
- 所以后续要靠图表和日志快速判断问题出在哪

下面这些内容建议全部保留。

## 6.1 训练主指标

必须记录：

- `loss/total`
- `loss/diffusion`
- `loss/nfp_total`
- `loss/nfp_mse`
- `loss/nfp_cosine`
- `train/lr_main`
- `train/lr_dit`
- `train/global_step`
- `train/epoch`

## 6.2 Memory / Surprise 专项指标

必须记录：

- `memory/bank_size`
- `memory/store_count`
- `memory/reject_count`
- `memory/evict_count`
- `memory/merge_count`
- `memory/retrieved_k`
- `memory/retrieval_sim_top1`
- `memory/retrieval_sim_topk_mean`
- `memory/retrieval_sim_topk_std`
- `memory/retrieved_age_mean`
- `memory/retrieved_timestep_gap_mean`
- `memory/query_norm`
- `memory/key_norm_mean`
- `memory/value_norm_mean`
- `memory/attn_out_norm`
- `memory/gate_mean`
- `memory/gate_std`

必须记录 surprise 分布：

- `surprise/frame_mean`
- `surprise/frame_std`
- `surprise/frame_min`
- `surprise/frame_max`
- `surprise/frame_p50`
- `surprise/frame_p90`
- `surprise/frame_p95`
- `surprise/store_threshold`
- `surprise/store_ratio`
- `surprise/compress_ratio`

如果做 segment 机制，还要记录：

- `segment/count_per_clip`
- `segment/avg_length`
- `segment/high_surprise_count`
- `segment/low_surprise_count`

## 6.3 梯度与参数健康度

必须记录：

- `grad/global_norm`
- `grad/memory_cross_attn_q`
- `grad/memory_cross_attn_k`
- `grad/memory_cross_attn_v`
- `grad/memory_cross_attn_o`
- `grad/memory_norm`
- `grad/nfp_head`
- `grad/dit_main`

建议记录：

- `param_norm/memory_cross_attn`
- `param_norm/nfp_head`
- `update_norm/memory_cross_attn`
- `update_norm/nfp_head`

目的：

- 判断是不是 memory 分支根本没在学
- 判断是不是某组参数梯度爆了或消失了

## 6.4 数值稳定性

必须记录：

- `train/has_nan`
- `train/has_inf`
- `train/skipped_steps`
- `train/overflow_count`
- `train/grad_scale`  
  如果使用 AMP scaler

建议遇到数值问题时立即打印：

- 当前 batch id
- 当前 sample 路径
- 当前 timestep
- 当前 sigma
- 当前 loss 分解

## 6.5 性能与资源

必须记录：

- `perf/step_time`
- `perf/data_time`
- `perf/forward_time`
- `perf/backward_time`
- `perf/optimizer_time`
- `perf/samples_per_sec`
- `perf/frames_per_sec`
- `perf/tokens_per_sec`

GPU/系统资源建议记录：

- `gpu/mem_allocated_mb`
- `gpu/mem_reserved_mb`
- `gpu/max_mem_allocated_mb`
- `gpu/max_mem_reserved_mb`
- `gpu/utilization`  
  如果方便采集
- `cpu/rss_gb`

## 6.6 数据健康度

必须记录：

- `data/frame_num`
- `data/lat_f`
- `data/lat_h`
- `data/lat_w`
- `data/prompt_length`
- `data/valid_pose_ratio`
- `data/valid_action_ratio`
- `data/invalid_sample_count`

建议记录输入分布：

- `data/action_hist/*`
- `data/pose_translation_norm_mean`
- `data/pose_rotation_norm_mean`

## 6.7 可视化样本

建议定期上传：

- 输入首帧图像
- 当前 chunk 的生成视频
- 回访同一 pose 的对比帧
- 检索到的 top-K memory 缩略图
- 高 surprise 帧网格图
- 被压缩段的 summary 可视化

如果成本允许，建议每 `N` 个 step 上传：

- `media/train_samples/...`
- `media/retrieval_debug/...`

## 6.8 文本日志

W&B 中应保存以下文本内容：

- 完整训练参数
- `git commit`
- 启动命令
- `accelerate` 配置
- 数据集路径与 split
- 权重路径
- 每个 checkpoint 保存事件
- 每次恢复训练事件
- OOM / NaN / 异常 traceback
- SLURM job id / node / gpu 信息

建议同时输出到本地文件：

- `logs/train.log`
- `logs/metrics.jsonl`
- `logs/env.txt`
- `logs/git_state.txt`

---

## 7. 训练脚本与 SLURM 启动脚本的建议改造

## 7.1 训练脚本

建议修改：

- `src/pipeline/train_v2_stage1.py`
- 后续若有 `train_v2_stage2.py`，保持同一套 logging 接口

建议新增参数：

- `--wandb_project`
- `--wandb_entity`
- `--wandb_run_name`
- `--wandb_mode`
- `--wandb_tags`
- `--wandb_notes`
- `--log_every_steps`
- `--media_log_every_steps`
- `--save_retrieval_debug_every_steps`

## 7.2 SLURM 启动脚本

建议新增：

- `src/scripts/run_train_v2_slurm.sh`

必须包含：

- 项目内 `wandb` 目录创建
- `WANDB_DIR/WANDB_CACHE_DIR/WANDB_DATA_DIR/WANDB_CONFIG_DIR` 导出
- 从 `~/.netrc` 自动提取 `WANDB_API_KEY`
- 训练 stdout/stderr 落盘
- 崩溃后把 SLURM 输出文件补传 W&B

建议额外记录：

- `SLURM_JOB_ID`
- `SLURM_NODELIST`
- `CUDA_VISIBLE_DEVICES`
- `HOSTNAME`
- `nvidia-smi`

---

## 8. 复现稳定性的配置要求

每个 run 必须固化以下信息：

- `git commit`
- 工作区是否 dirty
- 训练脚本路径
- 数据集版本
- 模型权重版本
- 所有 CLI 参数
- `accelerate` 配置文件内容
- 随机种子
- 机器与 GPU 信息

建议把这些内容：

- 同时写进 `config.yaml`
- 同时写进 W&B config

---

## 9. 建议的验收标准

修复完成后，至少要满足以下条件：

### 功能验收

- `infer_v2.py` 能按 chunk 连续生成并跨 chunk 使用 memory
- Stage1 训练时 `memory_cross_attn` 有非零梯度
- 写入和检索都使用真实的模型空间 `pose/state`
- Memory Bank 能同时返回 state 和 visual value
- `NFPHead` 输出 per-frame surprise

### 观测验收

- GPU 任务启动后尽快创建 W&B run
- 文本日志与 scalar 同步到同一个 run
- Crash 后完整 SLURM 日志能补回同一个 run
- W&B 缓存/配置目录统一写入项目目录内
- 节点端没有预设 `WANDB_API_KEY` 时也能自动认证

### 调试验收

- 可以从图表中直接判断：
  - loss 是不是爆了
  - memory 分支是不是没梯度
  - retrieval 是不是失效
  - surprise 是否塌缩
  - bank 是否只存了重复帧
  - step time / GPU memory 是否异常

---

## 10. 推荐执行顺序

建议严格按下面顺序推进：

1. 先补 W&B + 日志基础设施  
   原因：否则后面每次试验都很难定位问题

2. 再修 `infer_v2.py` 的 chunked memory 闭环  
   原因：先让推理逻辑符合设计

3. 再修 `train_v2_stage1.py`，让 memory 分支实际参与训练  
   原因：否则你训不到想要的模块

4. 再升级 `MemoryBank` 为 WorldMem 风格  
   原因：这一步会影响接口，适合在闭环跑通后再做

5. 最后再加入 segment / compression / advanced retrieval  
   原因：这属于增强项，不适合在基础闭环没通时就叠复杂度

---

## 11. 最终建议

如果只做最小可行修复，优先做这四件事：

- 给训练脚本加完整 W&B/SLURM 观测
- 把 `infer_v2.py` 改成真正的跨 chunk 推理
- 训练时不要再传 `memory_states=None`
- 用 `get_projected_frame_embs()` 替换掉当前的退化 `latent.mean` 检索

如果这四件事做完，你的系统才开始接近：

- “Surprise 决定记什么”
- “位姿对齐决定从哪里读”
- “世界模型跨 chunk 持续使用历史记忆”

在这之前，建议不要急着上更多高级设计，因为现在最缺的不是更多模块，而是：

- 一个真正可训练、可推理、可观测、可复现的基础闭环

