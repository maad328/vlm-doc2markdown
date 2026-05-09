<div align="center">

# 📄 VLM-Doc2Markdown

### Fine-Tuned Vision Language Model for Document-to-Markdown Conversion

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/simpHawk/qwen2-vl-document-ocr)
[![Model](https://img.shields.io/badge/🤗%20Model-qwen2--2b--vl--merged-orange)](https://huggingface.co/simpHawk/qwen2-2b-vl-merged)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A fine-tuned **Qwen2-VL-2B** vision-language model that converts document images into clean, structured Markdown text.*

[Live Demo](https://huggingface.co/spaces/simpHawk/qwen2-vl-document-ocr) • [Model Weights](https://huggingface.co/simpHawk/qwen2-2b-vl-merged) • [Report Issue](https://github.com/maad328/vlm-doc2markdown/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

**VLM-Doc2Markdown** is an end-to-end pipeline for converting document images (research papers, reports, articles) into structured Markdown using a fine-tuned Vision Language Model (VLM). The project leverages the **Qwen2-VL-2B** model, fine-tuned on the Nougat training dataset, to achieve high-quality document OCR with preserved formatting, mathematical notation, and hierarchical structure.

### Problem Statement
Traditional OCR tools extract raw text from documents but **lose formatting, structure, and mathematical expressions**. This project bridges that gap by using a vision-language model to understand document layout and generate properly formatted Markdown output.

### Solution
Fine-tune a compact yet powerful VLM (Qwen2-VL-2B) using **parameter-efficient techniques (LoRA + 4-bit quantization)** to generate Markdown from document images, preserving:
- Headings and section hierarchy
- Mathematical equations and formulas
- Tables and structured content
- Lists and bullet points
- Bold, italic, and other text formatting

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Vision-Language Model** | Fine-tuned Qwen2-VL-2B for document understanding |
| ⚡ **Efficient Training** | 4-bit quantization + LoRA via Unsloth for memory-efficient fine-tuning |
| 📐 **Structure Preservation** | Maintains headings, equations, tables, and lists |
| 🌐 **Live Demo** | Interactive Gradio app hosted on Hugging Face Spaces |
| 🔧 **End-to-End Pipeline** | From data preparation to model deployment |
| 📊 **Training Visualization** | Loss curves and prediction comparisons |

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Document Image  │────▶│   Qwen2-VL-2B    │────▶│  Markdown Text  │
│   (PNG/JPG)      │     │  (Fine-tuned)    │     │   (Structured)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   LoRA Adapters     │
                    │   (r=16, α=16)      │
                    │   4-bit Quantized   │
                    └────────────────────┘
```

**Base Model:** [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)  
**Fine-Tuning Method:** QLoRA (Quantized Low-Rank Adaptation)  
**Framework:** [Unsloth](https://github.com/unslothai/unsloth) + Hugging Face Transformers + TRL

---

## 📦 Dataset

The model is trained on the [Nougat Training Dataset](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example), which contains paired document images and their corresponding Markdown annotations.

| Split | Total Available | Used |
|-------|:-:|:-:|
| **Train** | 11,388 | 2,500 |
| **Validation** | 2,848 | 300 |

Each sample consists of:
- **Input:** Document page image (`.png`)
- **Target:** Corresponding Markdown text (`.mmd`)

---

## 🚀 Training Pipeline

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen2-VL-2B-Instruct` |
| Quantization | 4-bit (NF4) |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 16 |
| Learning Rate | 2e-4 |
| Batch Size | 2 (per device) |
| Gradient Accumulation | 4 steps |
| Epochs | 1 |
| Optimizer | AdamW (8-bit) |
| Hardware | NVIDIA Tesla T4 (Kaggle) |

### Training Workflow

```
1. Install dependencies (transformers, unsloth, peft, trl)
        │
2. Load & preprocess Nougat dataset (image-markdown pairs)
        │
3. Load Qwen2-VL-2B with 4-bit quantization
        │
4. Attach LoRA adapters to target modules
        │
5. Train with SFTTrainer (supervised fine-tuning)
        │
6. Merge LoRA weights → 16-bit model
        │
7. Push merged model to Hugging Face Hub
        │
8. Deploy Gradio app on Hugging Face Spaces
```

---

## 📊 Results

The fine-tuned model demonstrates strong performance in converting document images to Markdown, accurately preserving:

- ✅ Section headings and hierarchy
- ✅ Paragraph text and formatting
- ✅ Mathematical expressions
- ✅ Structured content layout

Training and validation loss curves, along with side-by-side comparisons of ground truth vs. predicted Markdown, are generated during training and saved as visualization artifacts.

---

## 🛠 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for training/inference)
- Kaggle account (for training on free GPU)
- Hugging Face account (for model hosting)

### Clone the Repository

```bash
git clone https://github.com/maad328/vlm-doc2markdown.git
cd vlm-doc2markdown
```

### Install Dependencies

```bash
pip install transformers datasets accelerate bitsandbytes peft trl
pip install qwen-vl-utils torchvision
pip install unsloth
```

### Run Inference

```python
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_id = "simpHawk/qwen2-2b-vl-merged"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load your document image
image = Image.open("your_document.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Convert this document into Markdown."}
    ]
}]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512)

markdown_text = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(markdown_text)
```

---

## 🌐 Live Demo

Try the model instantly without any setup:

🔗 **[Launch Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/simpHawk/qwen2-vl-document-ocr)**

Simply upload a document image and receive structured Markdown output in seconds.

---

## 📁 Project Structure

```
vlm-doc2markdown/
│
├── gen-a05.ipynb          # Main training notebook (Kaggle)
│                            ├── Library setup & installation
│                            ├── Dataset preparation & visualization
│                            ├── Model loading & LoRA configuration
│                            ├── Training with SFTTrainer
│                            ├── Loss curve visualization
│                            ├── Inference & evaluation
│                            ├── Ground truth vs. prediction comparison
│                            ├── Gradio local demo
│                            └── Model & Space deployment to Hugging Face
│
└── README.md              # Project documentation
```

---

## 🧰 Technologies Used

| Technology | Purpose |
|-----------|---------|
| [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Base vision-language model |
| [Unsloth](https://github.com/unslothai/unsloth) | Efficient fine-tuning framework |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Model loading & inference |
| [PEFT](https://github.com/huggingface/peft) | Parameter-efficient fine-tuning (LoRA) |
| [TRL](https://github.com/huggingface/trl) | Supervised fine-tuning trainer |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization |
| [Gradio](https://gradio.app/) | Interactive web demo |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Model deployment & hosting |
| [Kaggle](https://www.kaggle.com/) | Free GPU training environment |

---

## 🙏 Acknowledgments

- **[Qwen Team](https://github.com/QwenLM)** — for the Qwen2-VL-2B base model
- **[Unsloth AI](https://github.com/unslothai/unsloth)** — for the efficient fine-tuning framework
- **[Nougat (Meta AI)](https://github.com/facebookresearch/nougat)** — for inspiration and the training dataset
- **[Hugging Face](https://huggingface.co/)** — for model hosting and deployment infrastructure
- **[Kaggle](https://www.kaggle.com/)** — for free GPU compute resources

---

<div align="center">

**Made with ❤️ for document intelligence**

⭐ Star this repo if you find it useful!

</div>
