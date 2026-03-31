"""
infer_v3.py — LingBot-World Memory Enhancement 推理脚本（双模型版本）

对应 train_v3.py 训练的 checkpoint，推理时交替使用：
  - low_noise_model（WanModelWithMemory，含 Memory Bank）
  - high_noise_model（WanModel，无 Memory 模块）

状态：PENDING — 等待 train_v3.py 实现完成后配套实现
参考：/u/lwu9/Memory/projects/Lingbot_LSM/refs/csgo-finetune-v3/inference_csgo.py
"""

raise NotImplementedError(
    "infer_v3.py 尚未实现，等待 train_v3.py 实现完成后配套开发。\n"
    "当前请使用 infer_v2.py。"
)
