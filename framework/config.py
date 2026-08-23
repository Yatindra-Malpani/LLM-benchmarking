import os
import torch

# Hugging Face Configuration
# Local Hugging Face cache directory
HF_HOME = r"E:\benchmarking\hf_home"
os.environ["HF_HOME"] = HF_HOME

# Model Configuration
'''MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"'''

'''MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" '''

'''MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"'''

'''MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"'''

MODEL_NAME = "ministral/Ministral-3b-instruct"

# Quantization Configuration
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.float16
DOUBLE_QUANT = True

# Device Configuration
DEVICE_MAP = "auto"
