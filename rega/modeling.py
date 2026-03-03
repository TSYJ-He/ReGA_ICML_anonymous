from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass
class LoraHyperParams:
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"


def _dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype name: {name}")
    return table[name]


def load_vlm(base_model: str, dtype_name: str = "bf16", device_map: str = "auto"):
    dtype = _dtype_from_name(dtype_name)
    
    if "internvl" in base_model.lower():
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        config.vision_config.use_flash_attn = False
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        processor = tokenizer # Use tokenizer as processor for InternVL
    else:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        if "llava" in base_model.lower():
            model = LlavaForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map=device_map,
            )
        elif "qwen2.5-vl" in base_model.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            # Fallback path for models supporting AutoModelForVision2Seq.
            model = AutoModelForVision2Seq.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
    return model, processor


def apply_lora(model, hp: LoraHyperParams):
    cfg = LoraConfig(
        r=hp.r,
        lora_alpha=hp.alpha,
        lora_dropout=hp.dropout,
        target_modules=hp.target_modules,
        bias=hp.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # InternVL uses custom forward which doesn't accept inputs_embeds
    # We need to wrap it or just apply LoRA to language_model
    if hasattr(model, "language_model") and "internvl" in model.__class__.__name__.lower():
        model.language_model = get_peft_model(model.language_model, cfg)
    else:
        model = get_peft_model(model, cfg)

    return model


def named_trainable_params(model) -> Dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def clone_param_dict(param_dict: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in param_dict.items()}


def load_lora_only(model, adapter_path: str):
    # Keep this helper localized for future reuse.
    model.load_adapter(adapter_path)
    return model
