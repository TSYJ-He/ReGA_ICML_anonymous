import argparse
import math
import os
from dataclasses import dataclass

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from rega.common import ensure_dir, init_wandb, print_rank0, set_seed, wandb_config_from_env
from rega.data import LlavaSFTCollator, build_training_dataset
from rega.modeling import LoraHyperParams, apply_lora, load_vlm


@dataclass
class Phase1Config:
    base_model: str
    output_dir: str
    total_samples: int
    vqa_samples: int
    per_device_batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    logging_steps: int
    run_name: str
    vqa_manifest: str
    ocr_manifest: str
    allow_streaming_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ReGA Phase1 LoRA training")
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_samples", type=int, default=650_000)
    parser.add_argument("--vqa_samples", type=int, default=443_757)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--vqa_manifest", type=str, default="")
    parser.add_argument("--ocr_manifest", type=str, default="")
    parser.add_argument("--disable_streaming_fallback", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> Phase1Config:
    return Phase1Config(
        base_model=args.base_model,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        vqa_samples=args.vqa_samples,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        vqa_manifest=args.vqa_manifest,
        ocr_manifest=args.ocr_manifest,
        allow_streaming_fallback=(not args.disable_streaming_fallback),
    )


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(cfg.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process

    if is_main:
        ensure_dir(cfg.output_dir)
        wandb_cfg = wandb_config_from_env(
            run_name=cfg.run_name,
            job_type="train_phase1",
            tags="rega,phase1,lora",
        )
        init_wandb(wandb_cfg, config_payload=cfg.__dict__)

    model, processor = load_vlm(
        base_model=cfg.base_model,
        dtype_name="bf16",
        device_map=None,
    )
    lora_hp = LoraHyperParams(
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = apply_lora(model, lora_hp)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()
    # Ensure grad is allowed
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()
    model.train()

    dataset = build_training_dataset(
        total_samples=cfg.total_samples,
        vqa_samples=cfg.vqa_samples,
        seed=cfg.seed,
        vqa_manifest=cfg.vqa_manifest or None,
        ocr_manifest=cfg.ocr_manifest or None,
        allow_streaming_fallback=cfg.allow_streaming_fallback,
    )
    if "qwen" in cfg.base_model.lower():
        from rega.data import Qwen2VLSFTCollator
        collator = Qwen2VLSFTCollator(processor=processor, max_length=cfg.max_length)
    elif "internvl" in cfg.base_model.lower():
        from rega.data import InternVLSFTCollator
        collator = InternVLSFTCollator(processor=processor, max_length=cfg.max_length)
    else:
        collator = LlavaSFTCollator(processor=processor, max_length=cfg.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )

    global_batch = cfg.per_device_batch_size * cfg.grad_accum_steps * accelerator.num_processes
    total_steps = math.ceil(cfg.total_samples / global_batch)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    progress = tqdm(total=total_steps, disable=not is_main, desc="Phase1")
    optimizer.zero_grad(set_to_none=True)

    step = 0
    micro_step = 0
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss / cfg.grad_accum_steps)
        micro_step += 1

        if micro_step % cfg.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)

            if is_main and (step % cfg.logging_steps == 0 or step == 1):
                import wandb

                wandb.log(
                    {
                        "train/loss": float(loss.detach().cpu().item()),
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/step": step,
                    },
                    step=step,
                )
                print_rank0(f"[phase1] step={step}/{total_steps} loss={float(loss.detach().cpu().item()):.4f}")

            if step >= total_steps:
                break

    progress.close()
    accelerator.wait_for_everyone()

    if is_main:
        save_dir = os.path.join(cfg.output_dir, "phase1_adapter")
        ensure_dir(save_dir)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        
        # Avoid saving processor for internvl as it fails
        if "internvl" not in cfg.base_model.lower():
            processor.save_pretrained(save_dir)
            
        print_rank0(f"[phase1] saved adapter to {save_dir}")

        import wandb

        wandb.log({"train/phase1_done": 1})
        wandb.finish()


if __name__ == "__main__":
    main()
