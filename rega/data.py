import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


def _safe_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _pick_vqav2_answer(row: Dict) -> str:
    # VQAv2 provides multiple answers. Use the dominant answer if possible.
    answers = row.get("answers", None)
    if isinstance(answers, list) and answers:
        freq: Dict[str, int] = {}
        for ans in answers:
            a = _safe_text(ans)
            if not a:
                continue
            freq[a] = freq.get(a, 0) + 1
        if freq:
            return max(freq.items(), key=lambda kv: kv[1])[0]
    return _safe_text(row.get("multiple_choice_answer", ""))


def iter_vqav2(limit: int, seed: int) -> Iterator[Dict]:
    ds = load_dataset("Multimodal-Fatima/VQAv2_train", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=256)
    n = 0
    for row in ds:
        if n >= limit:
            break
        answer = _pick_vqav2_answer(row)
        question = _safe_text(row.get("question", ""))
        image = row.get("image", None)
        if not question or not answer or image is None:
            continue
        n += 1
        yield {
            "image": image.convert("RGB"),
            "question": question,
            "answer": answer,
            "source": "vqa_v2",
        }


def iter_ocr_vqa(limit: int, seed: int) -> Iterator[Dict]:
    ds = load_dataset("howard-hou/OCR-VQA", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=128)
    n = 0
    for row in ds:
        image = row.get("image", None)
        questions = row.get("questions", []) or []
        answers = row.get("answers", []) or []
        if image is None:
            continue
        for q, a in zip(questions, answers):
            if n >= limit:
                return
            q = _safe_text(q)
            a = _safe_text(a)
            if not q or not a:
                continue
            n += 1
            yield {
                "image": image.convert("RGB"),
                "question": q,
                "answer": a,
                "source": "ocr_vqa",
            }


def interleave_streams(stream_a: Iterable[Dict], stream_b: Iterable[Dict]) -> Iterator[Dict]:
    # Deterministic alternating interleave.
    ita = iter(stream_a)
    itb = iter(stream_b)
    while True:
        try:
            yield next(ita)
        except StopIteration:
            ita = None
        try:
            yield next(itb)
        except StopIteration:
            itb = None
        if ita is None and itb is None:
            return


class ReGAMixedIterableDataset(IterableDataset):
    def __init__(self, total_samples: int = 650_000, vqa_samples: int = 443_757, seed: int = 42):
        super().__init__()
        self.total_samples = total_samples
        self.vqa_samples = min(vqa_samples, total_samples)
        self.ocr_samples = max(0, total_samples - self.vqa_samples)
        self.seed = seed

    def __iter__(self):
        vqa_stream = iter_vqav2(self.vqa_samples, self.seed)
        ocr_stream = iter_ocr_vqa(self.ocr_samples, self.seed + 1)
        for sample in interleave_streams(vqa_stream, ocr_stream):
            yield sample


def _read_manifest(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _sample_or_repeat(records: List[Dict], n: int, seed: int) -> List[Dict]:
    if n <= 0:
        return []
    if not records:
        return []
    rng = random.Random(seed)
    if len(records) >= n:
        return rng.sample(records, n)
    # If insufficient records, repeat with replacement.
    return [rng.choice(records) for _ in range(n)]


class ReGAManifestDataset(Dataset):
    def __init__(
        self,
        vqa_manifest: Optional[str],
        ocr_manifest: Optional[str],
        total_samples: int = 650_000,
        vqa_samples: int = 443_757,
        seed: int = 42,
    ):
        self.total_samples = total_samples
        self.vqa_samples = min(vqa_samples, total_samples)
        self.ocr_samples = max(0, total_samples - self.vqa_samples)
        self.seed = seed

        vqa_records = _read_manifest(vqa_manifest) if vqa_manifest else []
        ocr_records = _read_manifest(ocr_manifest) if ocr_manifest else []

        vqa_part = _sample_or_repeat(vqa_records, self.vqa_samples, seed)
        ocr_part = _sample_or_repeat(ocr_records, self.ocr_samples, seed + 1)

        merged: List[Dict] = []
        ia, ib = 0, 0
        while ia < len(vqa_part) or ib < len(ocr_part):
            if ia < len(vqa_part):
                merged.append(vqa_part[ia])
                ia += 1
            if ib < len(ocr_part):
                merged.append(ocr_part[ib])
                ib += 1

        rng = random.Random(seed + 7)
        rng.shuffle(merged)
        self.records = merged

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "question": _safe_text(row["question"]),
            "answer": _safe_text(row["answer"]),
            "source": row.get("source", "unknown"),
        }


def build_training_dataset(
    total_samples: int,
    vqa_samples: int,
    seed: int,
    vqa_manifest: Optional[str] = None,
    ocr_manifest: Optional[str] = None,
    allow_streaming_fallback: bool = True,
):
    has_manifest = bool(vqa_manifest) or bool(ocr_manifest)
    if has_manifest:
        return ReGAManifestDataset(
            vqa_manifest=vqa_manifest,
            ocr_manifest=ocr_manifest,
            total_samples=total_samples,
            vqa_samples=vqa_samples,
            seed=seed,
        )
    if not allow_streaming_fallback:
        raise ValueError("No manifests provided and streaming fallback is disabled.")
    return ReGAMixedIterableDataset(
        total_samples=total_samples,
        vqa_samples=vqa_samples,
        seed=seed,
    )


@dataclass
class LlavaBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor


class LlavaSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        prefix_texts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
        full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}" for q, a in zip(questions, answers)]

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask prompt region so loss only supervises answer tokens.
        for i, (prefix, image) in enumerate(zip(prefix_texts, images)):
            prefix_ids = self.processor(
                text=prefix,
                images=image,
                return_tensors="pt",
                truncation=False,
            )["input_ids"][0]
            prefix_len = min(prefix_ids.shape[0], labels.shape[1])
            labels[i, :prefix_len] = -100

        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels
        return model_inputs


class Qwen2VLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        prefix_texts = []
        full_texts = []
        for q, a in zip(questions, answers):
            msg_prefix = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }
            ]
            msg_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": a}
                    ]
                }
            ]
            prefix_texts.append(self.processor.apply_chat_template(msg_prefix, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(msg_full, add_generation_prompt=False))

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = model_inputs["input_ids"]

        prefix_inputs = self.processor(
            text=prefix_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_lens = [len(p[p != self.processor.tokenizer.pad_token_id]) for p in prefix_inputs["input_ids"]]

        labels = input_ids.clone()
        for i, plen in enumerate(prefix_lens):
            valid_len = len(input_ids[i][input_ids[i] != self.processor.tokenizer.pad_token_id])
            pad_len = len(input_ids[i]) - valid_len
            if self.processor.tokenizer.padding_side == "left":
                labels[i, : pad_len + plen] = -100
            else:
                labels[i, :plen] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Note: InternVL processor handles image processing differently.
        # It's better to process images and texts separately.
        # But for now, let's use a simplified approach or we need to mimic the test script.
        pass

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Note: InternVL processor handles image processing differently.
        # It's better to process images and texts separately.
        # But for now, let's use a simplified approach or we need to mimic the test script.
        pass

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Note: InternVL processor handles image processing differently.
        # It's better to process images and texts separately.
        # But for now, let's use a simplified approach or we need to mimic the test script.
        pass

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Note: InternVL processor handles image processing differently.
        # It's better to process images and texts separately.
        # But for now, let's use a simplified approach or we need to mimic the test script.
        pass

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        from rega.internvl_utils import load_image
        self.load_image = load_image

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        pixel_values_list = []
        num_patches_list = []
        for img in images:
            pixel_values = self.load_image(img, max_num=6)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        
        pixel_values = torch.cat(pixel_values_list, dim=0)

        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        full_texts = []
        for q, a, num_patches in zip(questions, answers, num_patches_list):
            msg_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": a}
                    ]
                }
            ]
            full_text = self.processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches + IMG_END_TOKEN
            full_text = full_text.replace('<image>\n', image_tokens + '\n', 1)
            full_texts.append(full_text)

        tokens = self.processor(full_texts, return_tensors='pt', padding=True, truncation=False)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        labels = input_ids.clone()
        
        # We need to mask out the prompt part.
        # For simplicity, we can mask out everything before the assistant token if needed,
        # but let's just mask padding and image tokens for now, or find the assistant start.
        # In InternVL, assistant starts with '<|im_start|>assistant\n'
        assistant_token_id = self.processor.convert_tokens_to_ids('<|im_start|>')
        
        for i in range(len(labels)):
            # find <|im_start|>assistant
            # Actually, standard way is to tokenize prefix and mask it.
            msg_prefix = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": questions[i]}
                    ]
                }
            ]
            prefix_text = self.processor.apply_chat_template(msg_prefix, tokenize=False, add_generation_prompt=True)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches_list[i] + IMG_END_TOKEN
            prefix_text = prefix_text.replace('<image>\n', image_tokens + '\n', 1)
            prefix_ids = self.processor(prefix_text, return_tensors='pt', truncation=False).input_ids[0]
            prefix_len = min(prefix_ids.shape[0], labels.shape[1])
            labels[i, :prefix_len] = -100

        labels[attention_mask == 0] = -100
        
        # image_flags
        image_flags = torch.ones(sum(num_patches_list), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "labels": labels
        }

class InternVLSFTCollator:
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        from rega.internvl_utils import load_image
        self.load_image = load_image

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        pixel_values_list = []
        num_patches_list = []
        for img in images:
            pixel_values = self.load_image(img, max_num=6)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        
        pixel_values = torch.cat(pixel_values_list, dim=0)

        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        full_texts = []
        for q, a, num_patches in zip(questions, answers, num_patches_list):
            msg_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": a}
                    ]
                }
            ]
            full_text = self.processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches + IMG_END_TOKEN
            full_text = full_text.replace('<image>\n', image_tokens + '\n', 1)
            full_texts.append(full_text)

        tokens = self.processor(full_texts, return_tensors='pt', padding=True, truncation=False)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        labels = input_ids.clone()
        
        # We need to mask out the prompt part.
        # For simplicity, we can mask out everything before the assistant token if needed,
        # but let's just mask padding and image tokens for now, or find the assistant start.
        # In InternVL, assistant starts with '<|im_start|>assistant\n'
        assistant_token_id = self.processor.convert_tokens_to_ids('<|im_start|>')
        
        for i in range(len(labels)):
            # find <|im_start|>assistant
            # Actually, standard way is to tokenize prefix and mask it.
            msg_prefix = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": questions[i]}
                    ]
                }
            ]
            prefix_text = self.processor.apply_chat_template(msg_prefix, tokenize=False, add_generation_prompt=True)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches_list[i] + IMG_END_TOKEN
            prefix_text = prefix_text.replace('<image>\n', image_tokens + '\n', 1)
            prefix_ids = self.processor(prefix_text, return_tensors='pt', truncation=False).input_ids[0]
            prefix_len = min(prefix_ids.shape[0], labels.shape[1])
            labels[i, :prefix_len] = -100

        labels[attention_mask == 0] = -100
        
        # image_flags
        image_flags = torch.ones(sum(num_patches_list), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "labels": labels
        }
