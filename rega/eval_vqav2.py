import argparse
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

from rega.common import init_wandb, wandb_config_from_env


def _normalize_answer(ans: str) -> str:
    return str(ans).replace("\n", " ").replace("\t", " ").strip().lower()


def vqav2_accuracy(pred: str, gt_answers: List[str]) -> float:
    pred = _normalize_answer(pred)
    answers = [{"answer": _normalize_answer(a)} for a in gt_answers]
    if not answers:
        return 0.0
    gt_acc = []
    for gt in answers:
        others = [x for x in answers if x is not gt]
        matches = [x for x in others if x["answer"] == pred]
        gt_acc.append(min(1.0, len(matches) / 3.0))
    return float(statistics.mean(gt_acc))


def find_image(coco_root: str, image_id: int) -> Optional[str]:
    val_name = f"COCO_val2014_{image_id:012d}.jpg"
    train_name = f"COCO_train2014_{image_id:012d}.jpg"
    p1 = os.path.join(coco_root, val_name)
    p2 = os.path.join(coco_root, train_name)
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None


@dataclass
class EvalConfig:
    model_path: str
    coco_root: str
    split: str
    max_samples: int
    max_new_tokens: int
    run_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate VQAv2 with local COCO images")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True, help="Directory containing val2014/train2014")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--run_name", type=str, required=True)
    return parser.parse_args()


def load_model(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if "llava" in model_path.lower():
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_answer(model, processor, image: Image.Image, question: str, max_new_tokens: int) -> str:
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, model.dtype)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    answer_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    out = processor.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
    return out.strip()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        model_path=args.model_path,
        coco_root=args.coco_root,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        run_name=args.run_name,
    )

    wb = wandb_config_from_env(
        run_name=cfg.run_name,
        job_type="eval_vqav2",
        tags="rega,eval,vqav2",
    )
    run = init_wandb(wb, config_payload=cfg.__dict__)

    model, processor = load_model(cfg.model_path)
    ds = load_dataset("lmms-lab/VQAv2", split=cfg.split)
    n_eval = len(ds) if cfg.max_samples < 0 else min(len(ds), cfg.max_samples)

    total = 0.0
    seen = 0
    missing = 0
    pbar = tqdm(total=n_eval, desc="VQAv2-eval")
    for row in ds:
        if seen >= n_eval:
            break
        image = row.get("image", None)
        if image is None:
            missing += 1
            continue
        image = image.convert("RGB")
        pred = generate_answer(model, processor, image, row["question"], cfg.max_new_tokens)
        score = vqav2_accuracy(pred, row.get("answers", []))
        total += score
        seen += 1
        pbar.update(1)
        if seen % 50 == 0:
            import wandb

            wandb.log({"eval/progress_acc": total / max(1, seen), "eval/seen": seen, "eval/missing": missing})
    pbar.close()

    final_acc = total / max(1, seen)
    import wandb

    wandb.log(
        {
            "eval/vqav2_acc": final_acc,
            "eval/seen": seen,
            "eval/missing_images": missing,
            "eval/num_target": n_eval,
        }
    )
    run.finish()
    print(f"VQAv2 accuracy: {final_acc:.4f} over {seen} samples (missing={missing})", flush=True)


if __name__ == "__main__":
    main()

