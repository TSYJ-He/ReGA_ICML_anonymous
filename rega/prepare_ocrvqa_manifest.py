import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser("Build OCR-VQA local images + JSONL manifest")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=206_243)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("howard-hou/OCR-VQA", split="train", streaming=True)

    n_pairs = 0
    n_rows = 0
    n_saved_img = 0
    image_path_cache = {}

    mode = "a" if args.start_index > 0 else "w"
    with open(args.output_manifest, mode, encoding="utf-8") as wf:
        pbar = tqdm(total=args.max_pairs, desc="ocrvqa_pairs")
        for row in ds:
            n_rows += 1
            if n_rows <= args.start_index:
                continue

            image = row.get("image", None)
            image_id = str(row.get("image_id", f"row{n_rows}"))
            if image is None:
                continue

            if image_id not in image_path_cache:
                image_path = img_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    image.convert("RGB").save(image_path, format="JPEG", quality=95)
                    n_saved_img += 1
                image_path_cache[image_id] = str(image_path)

            questions = row.get("questions", []) or []
            answers = row.get("answers", []) or []
            for q, a in zip(questions, answers):
                q = str(q).strip()
                a = str(a).strip()
                if not q or not a:
                    continue
                rec = {
                    "image_path": image_path_cache[image_id],
                    "question": q,
                    "answer": a,
                    "source": "ocr_vqa",
                    "image_id": image_id,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
                pbar.update(1)
                if n_pairs >= args.max_pairs:
                    pbar.close()
                    print(
                        f"Done OCR-VQA manifest: pairs={n_pairs}, rows={n_rows}, images_saved={n_saved_img}",
                        flush=True,
                    )
                    sys.stdout.flush()
                    os._exit(0)
        pbar.close()

    print(
        f"Finished stream end: pairs={n_pairs}, rows={n_rows}, images_saved={n_saved_img}",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

