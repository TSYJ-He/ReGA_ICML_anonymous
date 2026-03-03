import argparse
import os
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm


def download(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"[skip] exists: {out_path}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def extract(zip_path: Path, dest: Path) -> None:
    marker = dest / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"[skip] extracted: {zip_path}", flush=True)
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser("Download VQAv2 evaluation image assets")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--with_train_images", action="store_true")
    parser.add_argument("--with_train_qa", action="store_true")
    parser.add_argument("--with_val_qa", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    zip_dir = root / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    val_zip = zip_dir / "val2014.zip"
    download("http://images.cocodataset.org/zips/val2014.zip", val_zip)
    extract(val_zip, root)

    if args.with_train_images:
        train_zip = zip_dir / "train2014.zip"
        download("http://images.cocodataset.org/zips/train2014.zip", train_zip)
        extract(train_zip, root)

    if args.with_train_qa:
        q_train = zip_dir / "v2_OpenEnded_mscoco_train2014_questions.zip"
        a_train = zip_dir / "v2_mscoco_train2014_annotations.zip"
        download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip", q_train)
        download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip", a_train)
        extract(q_train, root)
        extract(a_train, root)

    if args.with_val_qa:
        q_val = zip_dir / "v2_OpenEnded_mscoco_val2014_questions.zip"
        a_val = zip_dir / "v2_mscoco_val2014_annotations.zip"
        download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip", q_val)
        download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip", a_val)
        extract(q_val, root)
        extract(a_val, root)

    print(f"Done. Assets in: {root}", flush=True)


if __name__ == "__main__":
    main()

