import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm


def majority_answer(answers: List[Dict]) -> str:
    c = Counter()
    for a in answers:
        ans = str(a.get("answer", "")).strip()
        if ans:
            c[ans] += 1
    if not c:
        return ""
    return c.most_common(1)[0][0]


def image_path(coco_root: str, split: str, image_id: int) -> str:
    if split == "train":
        name = f"COCO_train2014_{image_id:012d}.jpg"
        return str(Path(coco_root) / "train2014" / name)
    if split == "val":
        name = f"COCO_val2014_{image_id:012d}.jpg"
        return str(Path(coco_root) / "val2014" / name)
    raise ValueError(f"Unsupported split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser("Build VQAv2 JSONL manifest for ReGA training/eval")
    parser.add_argument("--questions_json", type=str, required=True)
    parser.add_argument("--annotations_json", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.questions_json, "r", encoding="utf-8") as f:
        q_data = json.load(f)["questions"]
    with open(args.annotations_json, "r", encoding="utf-8") as f:
        a_data = json.load(f)["annotations"]

    q_map = {int(x["question_id"]): str(x["question"]).strip() for x in q_data}
    rows = []
    for ann in a_data:
        qid = int(ann["question_id"])
        question = q_map.get(qid, "").strip()
        answer = majority_answer(ann.get("answers", []))
        if not question or not answer:
            continue
        img = image_path(args.coco_root, args.split, int(ann["image_id"]))
        if not os.path.exists(img):
            continue
        rows.append(
            {
                "image_path": img,
                "question": question,
                "answer": answer,
                "source": "vqa_v2",
                "question_id": qid,
                "image_id": int(ann["image_id"]),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in tqdm(rows, desc="write_manifest"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Manifest written: {out_path} ({len(rows)} samples)", flush=True)


if __name__ == "__main__":
    main()

