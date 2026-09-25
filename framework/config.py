import os
import torch

# Hugging Face Configuration
# Local Hugging Face cache directory
HF_HOME = r"E:\benchmarking\hf_home"
os.environ["HF_HOME"] = HF_HOME

# Model Configuration
# Model Configuration
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "ministral/Ministral-3b-instruct",
]

NUM_RUNS = 3

# Quantization Configuration
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.float16
DOUBLE_QUANT = True

# Device Configuration
DEVICE_MAP = "auto"
