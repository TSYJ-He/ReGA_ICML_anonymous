import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb


@dataclass
class WandbConfig:
    project: str
    entity: str
    run_name: str
    job_type: str
    tags: Optional[str] = None
    notes: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_wandb(cfg: WandbConfig, config_payload: Dict[str, Any]) -> wandb.sdk.wandb_run.Run:
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        job_type=cfg.job_type,
        tags=(cfg.tags.split(",") if cfg.tags else None),
        notes=cfg.notes,
        config=config_payload,
        reinit=True,
    )
    return run


def wandb_config_from_env(run_name: str, job_type: str, tags: Optional[str] = None) -> WandbConfig:
    project = os.getenv("WANDB_PROJECT", "rega-icml-repro")
    entity = os.getenv("WANDB_ENTITY", "")
    if not entity:
        raise ValueError("WANDB_ENTITY is empty. Please export WANDB_ENTITY before running experiments.")
    return WandbConfig(
        project=project,
        entity=entity,
        run_name=run_name,
        job_type=job_type,
        tags=tags,
    )


def print_rank0(msg: str) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(msg, flush=True)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object is not a dataclass: {type(obj)}")

