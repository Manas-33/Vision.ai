# Vision.ai

A multimodal AI framework that fuses a vision encoder with a LLaMA-based large language model, enabling rich image-and-text understanding and generation. The model can be trained from scratch, fine-tuned end-to-end, or adapted efficiently with LoRA—all backed by DeepSpeed ZeRO for large-scale distributed training.

---

## Architecture

```
Image ──► CLIP / EvaCLIP Vision Encoder
                  │
                  ▼
          Multimodal Projector (linear or MLP-GELU)
                  │
                  ▼
LLaMA Language Model  ◄──  Text tokens
                  │
                  ▼
             Response
```

| Component | Description |
|---|---|
| **Vision encoder** | CLIP or EvaCLIP tower; frozen by default, fine-tuneable |
| **Multimodal projector** | Linear layer or configurable depth MLP with GELU activations |
| **Language model** | `VisionLlamaForCausalLM` — LLaMA extended with vision input handling |
| **Conversation templates** | LLaMA-2 chat and plain formats |

---

## Repository Structure

```
Vision.ai/
├── examples/                      # Sample images for testing
│   ├── photo.png
│   └── breaking_bad.png
├── scripts/
│   ├── shell/
│   │   ├── finetune.py            # Fine-tuning entry point (full / LoRA)
│   │   └── finetune_lora.sh       # Example launch script (8-GPU, torchrun)
│   ├── zero2.json                 # DeepSpeed ZeRO-2 config
│   ├── zero3.json                 # DeepSpeed ZeRO-3 config
│   └── zero3_offload.json         # DeepSpeed ZeRO-3 + CPU offload config
└── visionModel/
    ├── conversation.py            # Conversation / prompt templates
    ├── mm_utils.py                # Image preprocessing & tokenisation helpers
    └── model/
        ├── builder.py             # Model + tokenizer loading (incl. LoRA merge)
        ├── consolidate.py         # Checkpoint utilities
        ├── utils.py
        ├── share4v_arch.py        # Core multimodal meta-model classes
        ├── language_model/
        │   └── vision_llama.py    # VisionLlamaForCausalLM
        ├── multimodal_encoder/
        │   ├── builder.py
        │   └── clip_encoder.py    # CLIP / EvaCLIP vision tower
        └── multimodal_projector/
            └── builder.py         # Linear / MLP-GELU projector factory
    └── train/
        ├── train_mem.py           # Training entry point with Flash Attention
        ├── share4v_trainer.py     # Custom Trainer subclass
        └── llama_flash_attn_monkey_patch.py
```

---

## Requirements

```
torch >= 2.0
transformers
peft
deepspeed
bitsandbytes
Pillow
accelerate
```

Install dependencies:

```bash
pip install torch transformers peft deepspeed bitsandbytes Pillow accelerate
```

---

## Fine-Tuning

### 1. Prepare your data

The trainer accepts either a single JSON file or a `.txt` file listing multiple JSON dataset paths (one per line, optionally followed by a sampling ratio):

```
/path/to/dataset_a.json 1.0
/path/to/dataset_b.json 0.5
```

### 2. Configure the launch script

Edit `scripts/shell/finetune_lora.sh` and set the `MODEL` and `DATA` variables:

```bash
export MODEL="path/to/your/llm"   # local path or HuggingFace model ID
export DATA="path/to/data.txt"
```

### 3. Launch training

```bash
bash scripts/shell/finetune_lora.sh
```

This runs `torchrun` across 8 GPUs with the ZeRO-2 DeepSpeed config. Key hyperparameters can be adjusted directly in the script:

| Flag | Default | Description |
|---|---|---|
| `--img_size` | 490 | Input image resolution |
| `--use_lora` | True | Enable LoRA adaptation |
| `--fix_vit` | True | Freeze the vision encoder |
| `--fix_sampler` | True | Freeze the vision projector |
| `--learning_rate` | 5e-5 | Peak learning rate |
| `--num_train_epochs` | 1 | Number of training epochs |
| `--max_length` | 4096 | Maximum sequence length |
| `--deepspeed` | `ds_config_zero2.json` | DeepSpeed config to use |

### LoRA configuration

LoRA is configured via dataclass arguments. Defaults:

```
lora_r            = 64
lora_alpha        = 64
lora_dropout      = 0.05
lora_target_modules = [attention.wqkv, attention.wo,
                        feed_forward.w1/w2/w3]
```

---

## Inference / Loading a Model

Use `load_pretrained_model` from `visionModel/model/builder.py`:

```python
from visionModel.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="path/to/vision-model",
    model_base=None,          # provide base model path when loading LoRA weights
    model_name="vision",
    load_4bit=False,          # set True for 4-bit quantisation (bitsandbytes)
    load_8bit=False,          # set True for 8-bit quantisation
)
```

### Quantisation

| Flag | Precision | Library |
|---|---|---|
| `load_4bit=True` | NF4 (double quant) | bitsandbytes |
| `load_8bit=True` | INT8 | bitsandbytes |
| *(default)* | FP16 | — |

---

## Image Preprocessing

```python
from visionModel.mm_utils import process_images, tokenizer_image_token
from PIL import Image

image = Image.open("examples/photo.png").convert("RGB")
pixel_values = process_images([image], image_processor, model.config)
```

---

## DeepSpeed Configs

Three ready-to-use DeepSpeed configs are provided under `scripts/`:

| File | Stage | Notes |
|---|---|---|
| `zero2.json` | ZeRO-2 | Gradient/optimizer state sharding |
| `zero3.json` | ZeRO-3 | Full parameter sharding |
| `zero3_offload.json` | ZeRO-3 + offload | Offloads optimizer & parameters to CPU |

Pass the desired config with `--deepspeed <path>` when launching training.

---

## Conversation Templates

Two built-in conversation templates are available in `visionModel/conversation.py`:

- **`conv_vision_llama_2`** — LLaMA-2 chat format for vision-language dialogue.
- **`conv_llama_2`** — Standard LLaMA-2 chat format (text only).

---

## License

See [LICENSE](LICENSE) for details.
