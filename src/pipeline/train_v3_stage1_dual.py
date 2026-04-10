"""
train_v3_stage1_dual.py — LingBot-World Memory Enhancement 训练脚本
v3 数据 + Stage1 + 双模型（对应实验配置 ⑥）

数据：v3（8ch action：WASD + jump / crouch / fire / walk）
训练阶段：Stage1 — 冻结 DiT 所有参数，只训练记忆模块
  训练参数：MemoryCrossAttention + NFPHead + memory_norm（两个模型各自的记忆模块）
  冻结参数：DiT blocks（low_noise_model + high_noise_model 均冻结）、VAE、T5

双模型架构（与 train_v2_stage1_dual.py 一致）：
  - low_noise_model：WanModelWithMemory（t < 0.947）
  - high_noise_model：WanModelWithMemory（t >= 0.947）
  注：两个模型均使用 WanModelWithMemory（含记忆模块），与 train_v2_stage1_dual.py 完全一致。
  - 两个模型在同一 forward 中同时参与训练（非交替 epoch 策略）
  - 与 train_v2_stage1_dual.py 结构一致，区别仅在 action 维度（4ch → 8ch）

v3 vs v2 关键差异：
  - action.npy shape：[81, 8]（v2 为 [81, 4]）
  - 动作通道：[forward, back, left, right, jump, crouch, fire, walk]
  - dataloader 需适配 8ch action 编码（需更新 plucker/pose 条件拼接）

状态：PENDING — 等待 v3 数据（8ch action 格式）就绪后实现
参考：
  - train_v2_stage1_dual.py（结构模板）
  - refs/csgo-finetune-v3/train_lingbot_csgo.py（v3 数据处理参考）
"""

raise NotImplementedError(
    "train_v3_stage1_dual.py（双模型 Stage1）尚未实现。\n"
    "等待 v3 数据（8ch action）就绪后，基于 train_v2_stage1_dual.py 适配 8ch action 维度。\n"
    "单模型版本见 train_v3_stage1_single.py（实验配置 ⑤）。"
)
