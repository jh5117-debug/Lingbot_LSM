"""
train_v3.py — LingBot-World Memory Enhancement 训练脚本（双模型交替训练版本）

基于 csgo-finetune-v3，同时训练 low_noise_model（WanModelWithMemory）和
high_noise_model（WanModel），交替 epoch 策略：
  - 偶数 epoch：训练 low_noise_model（t < 947）
  - 奇数 epoch：训练 high_noise_model（t >= 947）

关键改进（相比 train_v2.py）：
  - 双 optimizer + 双 CosineAnnealingLR scheduler
  - FlowMatchingSchedule 新增 low_noise_indices / high_noise_indices 分组
  - 训练权重各组内部归一化（防止高噪声侧被低权重忽视）
  - Memory 模块只加在 low_noise_model（WanModelWithMemory），high_noise_model 为原始 WanModel

状态：PENDING — 等待更完整的 csgo-finetune-v3 代码后实现
参考：/u/lwu9/Memory/projects/Lingbot_LSM/refs/csgo-finetune-v3/train_lingbot_csgo.py
"""

raise NotImplementedError(
    "train_v3.py 尚未实现。\n"
    "等待更完整的 csgo-finetune-v3 代码后，基于其实现 Memory Enhancement 双模型训练版本。\n"
    "当前请使用 train_v2.py（单低噪声模型版本）。"
)
