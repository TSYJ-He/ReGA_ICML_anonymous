import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import wandb

from rega.common import ensure_dir, init_wandb, wandb_config_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run VLMEval benchmarks with W&B logging")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--local_model_path", type=str, default="")
    parser.add_argument("--model_family", type=str, default="", choices=["", "llava", "qwen2vl", "internvl"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "infer", "eval"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--judge", type=str, default="")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def resolve_dataset_class(dataset_name: str) -> str:
    from vlmeval.dataset import DATASET_CLASSES

    for cls in DATASET_CLASSES:
        try:
            if dataset_name in cls.supported_datasets():
                return cls.__name__
        except Exception:
            continue
    raise ValueError(f"Could not resolve dataset class for dataset: {dataset_name}")


def build_local_model_config(local_model_path: str, model_family: str) -> Dict:
    if model_family == "llava":
        return {"class": "LLaVA_HF", "model_path": local_model_path}
    if model_family == "qwen2vl":
        return {
            "class": "Qwen2VLChat",
            "model_path": local_model_path,
            "min_pixels": 1280 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
            "use_custom_prompt": False,
        }
    if model_family == "internvl":
        return {"class": "InternVLChat", "model_path": local_model_path, "version": "V2.0"}
    raise ValueError(f"Unsupported model_family: {model_family}")


def maybe_generate_config(args: argparse.Namespace) -> str:
    if not args.local_model_path:
        return args.config_json
    if not args.datasets:
        raise ValueError("--local_model_path requires --datasets")
    if not args.model_family:
        raise ValueError("--local_model_path requires --model_family")

    ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
    model_name = f"{args.model_family}_{Path(args.local_model_path).name}"
    cfg = {
        "model": {
            model_name: build_local_model_config(
                local_model_path=args.local_model_path,
                model_family=args.model_family,
            )
        },
        "data": {},
    }
    for d in ds:
        cfg["data"][d] = {"class": resolve_dataset_class(d), "dataset": d}

    out_path = Path(args.work_dir) / f"{args.run_name}.config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return str(out_path)


def collect_result_files(work_dir: Path, start_ts: float) -> List[Path]:
    out = []
    for ext in ("*.csv", "*.xlsx"):
        out.extend([p for p in work_dir.rglob(ext) if p.stat().st_mtime >= start_ts])
    return sorted(out)


def log_metrics_from_file(path: Path) -> None:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception:
        return
    if df.empty:
        return
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return
    row = df.iloc[-1]
    payload = {}
    for c in numeric_cols:
        val = row[c]
        if pd.notna(val):
            payload[f"result/{path.stem}/{c}"] = float(val)
    if payload:
        wandb.log(payload)


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)
    cfg_json = maybe_generate_config(args)

    wb = wandb_config_from_env(
        run_name=args.run_name,
        job_type="eval_benchmark",
        tags="rega,eval,vlmeval",
    )
    run = init_wandb(wb, config_payload=vars(args))

    cmd = [sys.executable, "-m", "vlmeval", "--work-dir", args.work_dir, "--mode", args.mode, "--nproc", str(args.nproc)]
    if cfg_json:
        cmd.extend(["--config", cfg_json])
    else:
        if not args.datasets or not args.models:
            raise ValueError("Either --config_json or both --datasets and --models must be provided.")
        ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
        ms = [x.strip() for x in args.models.split(",") if x.strip()]
        cmd.extend(["--data", *ds, "--model", *ms])

    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.judge:
        cmd.extend(["--judge", args.judge])
    if args.reuse:
        cmd.append("--reuse")

    env = os.environ.copy()
    env.setdefault("USE_TF", "0")
    env.setdefault(
        "PYTHONPATH",
        "/mnt/data/zsk/ReGA_ICML/.deps_wandb:/mnt/data/zsk/ReGA_ICML/.deps_tf452:/mnt/data/zsk/ReGA_ICML/.deps_antlr413:/mnt/data/zsk/ReGA_ICML/.deps",
    )

    log_file = Path(args.work_dir) / f"{args.run_name}.vlmeval.log"
    start_ts = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        exit_code = process.returncode

    wandb.log({"eval/exit_code": exit_code, "eval/log_file": str(log_file), "eval/config_json": cfg_json or ""})
    if exit_code != 0:
        run.finish(exit_code=exit_code)
        raise SystemExit(exit_code)

    result_files = collect_result_files(Path(args.work_dir), start_ts)
    wandb.log({"eval/result_files_count": len(result_files)})
    for p in result_files:
        log_metrics_from_file(p)
    run.finish()


if __name__ == "__main__":
    main()

