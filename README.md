# ReGA: Restoring Generalization in Fine-tuned MLLMs via Geometric Alignment

<img width="1858" height="688" alt="teaser_rega" src="https://github.com/user-attachments/assets/e7c8f560-4435-4a7e-bb1b-8952a53808e1" />


This repository provides the official implementation for reproducing the results of the paper **"Restoring Generalization in Fine-tuned Multimodal LLMs via Geometric Alignment (ReGA)"**.

ReGA is a two-phase post-tuning framework designed to mitigate the catastrophic forgetting of zero-shot generalization capabilities in Multimodal Large Language Models (MLLMs) during downstream fine-tuning.

## 🚀 Overview

ReGA aligns the fine-tuned model's weights with the pre-trained generalization prior using:
1. **Linear Mode Connectivity (LMC)**: Exploring the weight space between pre-trained and fine-tuned points.
2. **Proximal Generalization Prior ($R_{prox}$)**: Constraining the optimization to stay within the high-generalization region.






<img width="946" height="688" alt="ReGA_Para_Space" src="https://github.com/user-attachments/assets/62fedf10-3852-4ec1-8ce8-2936f76b2703" />

## 🛠️ Supported Models
- **LLaVA-1.5-7B**
- **Qwen2.5-VL-7B-Instruct**
- **InternVL-3.5-8B**

## 📊 Benchmarks
We evaluate ReGA across a wide range of multimodal benchmarks:
- **General Perception**: MME, MMMU, POPE
- **Document/OCR**: OCRBench, DocVQA, InfoVQA, TextVQA
- **VQA**: VQAv2
- 
<img width="940" height="438" alt="Visual Representation Fidelity Analysis" src="https://github.com/user-attachments/assets/04454cfe-feb2-404a-bf6f-4cdde973eca1" />



### 1. Phase 1: Downstream Fine-tuning (LoRA)
Fine-tune the base VLM on specific downstream tasks (e.g., VQAv2 + OCR-VQA).
```bash
bash scripts/run_llava_phase1.sh
# or for other models:
bash scripts/run_qwen_phase1.sh
bash scripts/run_internvl_phase1.sh
```

### 2. Phase 2: ReGA Geometric Alignment
Apply the ReGA algorithm to restore generalization while maintaining downstream performance.
```bash
bash scripts/run_llava_phase2.sh <phase1_adapter_path>
```




## 📜 License
This project is released under the MIT License.
