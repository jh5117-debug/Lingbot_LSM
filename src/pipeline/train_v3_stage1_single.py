"""
train_v3_stage1_single.py — LingBot-World Memory Enhancement 训练脚本
v3 数据 + Stage1 + 单模型（对应实验配置 ⑤）

数据：v3（8ch action：WASD + jump / crouch / fire / walk）
训练阶段：Stage1 — 冻结 DiT 所有参数，只训练记忆模块
  训练参数：MemoryCrossAttention + NFPHead + memory_norm（同 train_v2_stage1.py）
  冻结参数：DiT blocks、VAE、T5

单模型架构：
  - 仅训练 low_noise_model（WanModelWithMemory，t < 0.947）
  - high_noise_model 不参与本配置的训练
  - FlowMatchingSchedule sigma 范围：[0, 0.947)

v3 vs v2 关键差异：
  - action.npy shape：[81, 8]（v2 为 [81, 4]）
  - 动作通道：[forward, back, left, right, jump, crouch, fire, walk]
  - dataloader 需适配 8ch action 编码

状态：PENDING — 等待 v3 数据（8ch action 格式）就绪后实现
参考：
  - train_v2_stage1.py（结构模板，仅需适配 8ch action 维度）
  - refs/csgo-finetune-v3/train_lingbot_csgo.py（v3 数据处理参考）
"""

raise NotImplementedError(
    "train_v3_stage1_single.py（单模型 Stage1）尚未实现。\n"
    "等待 v3 数据（8ch action）就绪后，基于 train_v2_stage1.py 适配 8ch action 维度。\n"
    "双模型版本见 train_v3_stage1.py（将重命名为 train_v3_stage1_dual.py，实验配置 ⑥）。"
)
