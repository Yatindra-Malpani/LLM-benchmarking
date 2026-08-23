import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from src.framework.config import (
    MODEL_NAME,
    LOAD_IN_4BIT,
    QUANT_TYPE,
    COMPUTE_DTYPE,
    DOUBLE_QUANT,
    DEVICE_MAP,
)
def create_bnb_config():
    """
    Create the BitsAndBytes quantization configuration.
    """
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=DOUBLE_QUANT,
    )
def load_model():
    """
    Load tokenizer and model.

    Returns:
        tokenizer
        model
        load_time (seconds)
    """
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=create_bnb_config(),
        device_map=DEVICE_MAP,
    )
    load_time = time.perf_counter() - start_time

    return tokenizer, model, load_time
