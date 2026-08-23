from src.framework.config import MODEL_NAME
from src.framework.loader import load_model
from src.framework.metrics import (
    reset_gpu_peak_memory,
    get_gpu_memory,
    get_peak_gpu_memory,
)

def main():
    print("=" * 60)
    print("Model Loading Test")
    print("=" * 60)

    print(f"\nModel: {MODEL_NAME}")

    # Reset peak memory statistics before loading
    reset_gpu_peak_memory()

    # Load model
    tokenizer, model, load_time = load_model()

    # GPU memory statistics
    allocated, reserved = get_gpu_memory()
    peak = get_peak_gpu_memory()

    print(f"\nLoad time: {load_time:.2f} seconds")

    print("\nModel device:")
    print(next(model.parameters()).device)

    print("\nGPU Memory:")
    print(f"Allocated:      {allocated:.2f} GB")
    print(f"Reserved:       {reserved:.2f} GB")
    print(f"Peak allocated: {peak:.2f} GB")

    print("\nModel loaded successfully.")

if __name__ == "__main__":
    main()
