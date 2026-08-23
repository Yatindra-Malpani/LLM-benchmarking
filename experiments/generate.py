from src.framework.config import MODEL_NAME
from src.framework.loader import load_model
from src.framework.inference import generate_response


def main():
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print("Inference Benchmark")
    print("=" * 60)


    tokenizer, model, load_time = load_model()

    messages = [
        {
            "role": "user",
            "content": "Explain what a transformer is in simple terms.",
        }
    ]

    response, metrics = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=100,
    )

    print("\nResponse:")
    print(response)

    print("\nInference Metrics:")
    print(f"Input tokens:       {metrics['input_tokens']}")
    print(f"Output tokens:      {metrics['output_tokens']}")
    print(f"TTFT:               {metrics['ttft']:.3f} seconds")
    print(f"Generation time:    {metrics['generation_time']:.3f} seconds")
    print(f"Tokens/second:      {metrics['tokens_per_second']:.2f}")
    print(f"Peak VRAM:          {metrics['peak_vram_gb']:.2f} GB")


if __name__ == "__main__":
    main()
