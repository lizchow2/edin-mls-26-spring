# Save this as download_model.py
import torch
from transformers import AutoProcessor, AutoConfig, GlmAsrForConditionalGeneration

model_name = "zai-org/GLM-ASR-Nano-2512"

print(f"Downloading {model_name} to cache...")
# This fetches the config
AutoConfig.from_pretrained(model_name)
# This fetches the processor/tokenizer
AutoProcessor.from_pretrained(model_name)
# This fetches the actual weights (this may take a few minutes)
GlmAsrForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)

print("Download complete! Files are now in your ~/.cache folder.")
