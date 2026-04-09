"""
Inference script for CSGO fine-tuned LingBot-World model.
Supports dual-model checkpoint (both high_noise_model + low_noise_model fine-tuned).

Usage:
    # Dual-model full FT checkpoint:
    torchrun --nproc_per_node=8 inference_csgo.py \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act/ \
        --ft_ckpt_dir /home/nvme02/lingbot-world/output/csgo_dual_ft/epoch_2/ \
        --image <clip>/image.jpg \
        --action_path <clip>/ \
        --prompt "First-person view of CS:GO gameplay on de_dust2" \
        --size 480*832 --frame_num 81

    # LoRA inference:
    torchrun --nproc_per_node=8 inference_csgo.py \
        --ckpt_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act/ \
        --lora_path <output>/final/low_noise_model/lora_weights.pth \
        --image <clip>/image.jpg \
        --action_path <clip>/ \
        --size 480*832 --frame_num 81
"""

import argparse
import logging
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Base model directory (with VAE, T5, etc.)")
    parser.add_argument("--lingbot_code_dir", type=str,
                        default="/home/nvme02/lingbot-world/code/lingbot-world")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights .pth file")
    parser.add_argument("--ft_ckpt_dir", type=str, default=None,
                        help="Path to fine-tuned checkpoint dir (contains low_noise_model/ and high_noise_model/)")
    # Keep old arg name for backward compatibility
    parser.add_argument("--ft_model_dir", type=str, default=None,
                        help="(Deprecated) Path to single fine-tuned low_noise_model directory")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--action_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="First-person view of CS:GO competitive gameplay")
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--save_file", type=str, default="output_csgo.mp4")
    parser.add_argument("--sample_steps", type=int, default=70)
    parser.add_argument("--sample_shift", type=float, default=10.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    args = parser.parse_args()

    # Add lingbot code to path
    sys.path.insert(0, args.lingbot_code_dir)

    effective_ckpt_dir = args.ckpt_dir

    if args.lora_path:
        # LoRA inference: load base model, inject LoRA, merge, save temp
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
        logging.info("Loading base model + LoRA weights for inference...")

        from wan.modules.model import WanModel
        from peft import LoraConfig, inject_adapter_in_model

        model = WanModel.from_pretrained(
            args.ckpt_dir, subfolder="low_noise_model",
            torch_dtype=torch.bfloat16, control_type="act",
        )

        lora_state = torch.load(args.lora_path, map_location="cpu", weights_only=True)
        target_modules = set()
        for key in lora_state.keys():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ("lora_A", "lora_B"):
                    module_name = ".".join(parts[:i])
                    target_modules.add(module_name)
                    break

        target_modules = sorted(list(target_modules))
        logging.info(f"Detected {len(target_modules)} LoRA target modules")

        for key, val in lora_state.items():
            if "lora_A" in key:
                lora_rank = val.shape[0]
                break

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)

        mapped_state = {}
        for key, val in lora_state.items():
            if "lora_A.weight" in key and "default" not in key:
                key = key.replace("lora_A.weight", "lora_A.default.weight")
            if "lora_B.weight" in key and "default" not in key:
                key = key.replace("lora_B.weight", "lora_B.default.weight")
            mapped_state[key] = val

        result = model.load_state_dict(mapped_state, strict=False)
        logging.info(f"Loaded LoRA weights: {len(mapped_state)} keys, "
                     f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='act_')
        merged_ckpt = os.path.join(tmp_dir, "low_noise_model")
        import peft.tuners.lora as lora_module
        for _name, _module in model.named_modules():
            if isinstance(_module, lora_module.Linear):
                _module.merge()
        model.save_pretrained(merged_ckpt)
        logging.info(f"Saved merged model to {merged_ckpt}")

        for item in ["high_noise_model", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                      "google", "configuration.json"]:
            src = os.path.join(args.ckpt_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        effective_ckpt_dir = tmp_dir
        del model, lora_state, mapped_state
        torch.cuda.empty_cache()

    elif args.ft_ckpt_dir:
        # Dual-model fine-tuned checkpoint: contains both low_noise_model/ and high_noise_model/
        # Create temp dir with symlinks: FT models + base model's VAE/T5/etc.
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='act_')

        # Symlink fine-tuned models
        for model_name in ["low_noise_model", "high_noise_model"]:
            src = os.path.join(args.ft_ckpt_dir, model_name)
            dst = os.path.join(tmp_dir, model_name)
            if os.path.exists(src):
                os.symlink(src, dst)
                logging.info(f"Using fine-tuned {model_name} from {src}")
            else:
                # Fallback to base model if this model wasn't fine-tuned
                base_src = os.path.join(args.ckpt_dir, model_name)
                os.symlink(base_src, dst)
                logging.info(f"Using base {model_name} from {base_src} (not found in ft_ckpt_dir)")

        # Symlink shared components from base model
        for item in ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                      "google", "configuration.json"]:
            src = os.path.join(args.ckpt_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        effective_ckpt_dir = tmp_dir

    elif args.ft_model_dir:
        # (Deprecated) Old single-model full FT: only low_noise_model was fine-tuned
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='act_')
        os.symlink(args.ft_model_dir, os.path.join(tmp_dir, "low_noise_model"))
        for item in ["high_noise_model", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                      "google", "configuration.json"]:
            src = os.path.join(args.ckpt_dir, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
        effective_ckpt_dir = tmp_dir

    # Standard LingBot inference
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    from wan.distributed.util import init_distributed_group
    from PIL import Image
    import torch.distributed as dist

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if args.ulysses_size > 1:
        init_distributed_group()

    cfg = WAN_CONFIGS["i2v-A14B"]

    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=effective_ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
    )

    img = Image.open(args.image).convert("RGB")

    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver="unipc",
        sampling_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        seed=42,
        offload_model=False,
    )

    if rank == 0 and video is not None:
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info(f"Saved video -> {args.save_file}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
