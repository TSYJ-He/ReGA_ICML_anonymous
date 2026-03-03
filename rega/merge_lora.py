import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge LoRA adapter into base VLM")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def dtype_of(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=dtype_of(args.dtype),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
    processor.save_pretrained(args.output_dir)
    print(f"Merged model saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

