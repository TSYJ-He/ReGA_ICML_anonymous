import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm


def majority_answer_from_raw(answers):
    c = Counter()
    if isinstance(answers, list):
        for x in answers:
            if isinstance(x, dict):
                a = str(x.get("answer", "")).strip()
            else:
                a = str(x).strip()
            if a:
                c[a] += 1
    if not c:
        return ""
    return c.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser("Build VQAv2 manifest from HF dataset with inline images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=443_757)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Multimodal-Fatima/VQAv2_train", split="train", streaming=True)
    if args.shuffle_buffer > 1:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    n = 0
    with open(args.output_manifest, "w", encoding="utf-8") as wf:
        pbar = tqdm(total=args.max_samples, desc="vqav2_pairs")
        for row in ds:
            if n >= args.max_samples:
                break
            q = str(row.get("question", "")).strip()
            a = majority_answer_from_raw(row.get("answers", []))
            image = row.get("image", None)
            if not q or not a or image is None:
                continue

            qid = row.get("question_id", n)
            img_name = f"{qid}.jpg"
            img_path = img_dir / img_name
            if not img_path.exists():
                image.convert("RGB").save(img_path, format="JPEG", quality=95)

            rec = {
                "image_path": str(img_path),
                "question": q,
                "answer": a,
                "source": "vqa_v2",
                "question_id": int(qid) if str(qid).isdigit() else n,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            pbar.update(1)
        pbar.close()

    print(f"Manifest written: {args.output_manifest} ({n} samples)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

