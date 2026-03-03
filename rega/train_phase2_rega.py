import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rega.common import ensure_dir, init_wandb, print_rank0, set_seed, wandb_config_from_env
from rega.data import LlavaSFTCollator, build_training_dataset
from rega.modeling import load_vlm, named_trainable_params


@dataclass
class Phase2Config:
    base_model: str
    phase1_adapter: str
    output_dir: str
    total_samples: int
    vqa_samples: int
    phase2_fraction: float
    per_device_batch_size: int
    grad_accum_steps: int
    lr: float
    seed: int
    max_length: int
    lambda_prox: float
    beta_lmc: float
    lmc_k: int
    logging_steps: int
    run_name: str
    vqa_manifest: str
    ocr_manifest: str
    allow_streaming_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ReGA Phase2 geometric alignment")
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--phase1_adapter", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_samples", type=int, default=650_000)
    parser.add_argument("--vqa_samples", type=int, default=443_757)
    parser.add_argument("--phase2_fraction", type=float, default=0.2)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lambda_prox", type=float, default=0.1)
    parser.add_argument("--beta_lmc", type=float, default=1.0)
    parser.add_argument("--lmc_k", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--vqa_manifest", type=str, default="")
    parser.add_argument("--ocr_manifest", type=str, default="")
    parser.add_argument("--disable_streaming_fallback", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> Phase2Config:
    return Phase2Config(
        base_model=args.base_model,
        phase1_adapter=args.phase1_adapter,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        vqa_samples=args.vqa_samples,
        phase2_fraction=args.phase2_fraction,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        seed=args.seed,
        max_length=args.max_length,
        lambda_prox=args.lambda_prox,
        beta_lmc=args.beta_lmc,
        lmc_k=args.lmc_k,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        vqa_manifest=args.vqa_manifest,
        ocr_manifest=args.ocr_manifest,
        allow_streaming_fallback=(not args.disable_streaming_fallback),
    )


def _clone_grads(param_dict: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, p in param_dict.items():
        if p.grad is None:
            out[name] = torch.zeros_like(p, dtype=torch.float32)
        else:
            out[name] = p.grad.detach().to(torch.float32).clone()
    return out


def _zero_param_grads(param_dict: Dict[str, torch.nn.Parameter]) -> None:
    for p in param_dict.values():
        p.grad = None


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
            job_type="train_phase2",
            tags="rega,phase2,lmc,proximal",
        )
        init_wandb(wandb_cfg, config_payload=cfg.__dict__)

    base_model, processor = load_vlm(
        base_model=cfg.base_model,
        dtype_name="bf16",
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, cfg.phase1_adapter, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()
    model.train()

    phase2_total = int(cfg.total_samples * cfg.phase2_fraction)
    phase2_vqa = min(cfg.vqa_samples, phase2_total)
    dataset = build_training_dataset(
        total_samples=phase2_total,
        vqa_samples=phase2_vqa,
        seed=cfg.seed + 17,
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

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

    global_batch = cfg.per_device_batch_size * cfg.grad_accum_steps * accelerator.num_processes
    total_steps = math.ceil(phase2_total / global_batch)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    trainable = named_trainable_params(model)
    # Keep anchor params on each rank's device to avoid CPU/CUDA mismatch during interpolation.
    anchor_params = {n: p.detach().to(torch.float32).clone() for n, p in trainable.items()}
    progress = tqdm(total=total_steps, disable=not is_main, desc="Phase2-ReGA")

    step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        # Snapshot current trainable weights.
        with torch.no_grad():
            current_params = {n: p.detach().clone() for n, p in trainable.items()}

        # 1) Standard task gradient at current point.
        outputs = model(**batch)
        task_loss = outputs.loss
        accelerator.backward(task_loss)
        task_grads = _clone_grads(trainable)
        _zero_param_grads(trainable)

        # 2) LMC Monte Carlo gradients at interpolated points.
        lmc_grads = {n: torch.zeros_like(g) for n, g in task_grads.items()}
        lmc_losses = []
        for _ in range(cfg.lmc_k):
            alpha = random.random()
            with torch.no_grad():
                for name, p in trainable.items():
                    anchor = anchor_params[name]
                    current = current_params[name].to(torch.float32)
                    interp = anchor + alpha * (current - anchor)
                    p.copy_(interp.to(dtype=p.dtype))

            out_k = model(**batch)
            loss_k = out_k.loss
            lmc_losses.append(float(loss_k.detach().cpu().item()))
            accelerator.backward(loss_k)
            grads_k = _clone_grads(trainable)
            _zero_param_grads(trainable)
            for name in lmc_grads:
                lmc_grads[name] += alpha * grads_k[name]

        # 3) Restore current weights and compose final gradient.
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(current_params[name])

        for name, p in trainable.items():
            prox_grad = 2.0 * cfg.lambda_prox * (p.detach().to(torch.float32) - anchor_params[name])
            g = task_grads[name] + (cfg.beta_lmc / cfg.lmc_k) * lmc_grads[name] + prox_grad
            g = g / cfg.grad_accum_steps
            if p.grad is None:
                p.grad = g.to(p.dtype)
            else:
                p.grad = p.grad + g.to(p.dtype)

        micro_step += 1
        if micro_step % cfg.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)

            if is_main and (step % cfg.logging_steps == 0 or step == 1):
                import wandb

                prox_norm = 0.0
                with torch.no_grad():
                    for name, p in trainable.items():
                        prox_norm += torch.sum((p.detach().to(torch.float32) - anchor_params[name]) ** 2).item()
                wandb.log(
                    {
                        "train/task_loss": float(task_loss.detach().cpu().item()),
                        "train/lmc_loss_mean": float(sum(lmc_losses) / max(1, len(lmc_losses))),
                        "train/prox_norm": float(prox_norm),
                        "train/step": step,
                    },
                    step=step,
                )
                print_rank0(
                    f"[phase2] step={step}/{total_steps} task_loss={float(task_loss.detach().cpu().item()):.4f} lmc={float(sum(lmc_losses)/max(1,len(lmc_losses))):.4f}"
                )

            if step >= total_steps:
                break

    progress.close()
    accelerator.wait_for_everyone()

    if is_main:
        save_dir = os.path.join(cfg.output_dir, "phase2_rega_adapter")
        ensure_dir(save_dir)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print_rank0(f"[phase2] saved adapter to {save_dir}")

        import wandb

        wandb.log({"train/phase2_done": 1})
        wandb.finish()


if __name__ == "__main__":
    main()

