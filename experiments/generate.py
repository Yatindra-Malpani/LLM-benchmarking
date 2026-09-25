import gc
import torch

from src.framework.config import MODELS, NUM_RUNS
from src.framework.loader import load_model
from src.framework.inference import generate_response

def main():
    messages = [
        {
            "role": "user",
            "content": (
                "Explain what a transformer is in the context of "
                "neural networks. Explain it in simple terms."
            ),
        }
    ]

    print("=" * 70)
    print("LOCAL LLM BENCHMARK")
    print("=" * 70)
    print(f"Models:         {len(MODELS)}")
    print(f"Runs per model: {NUM_RUNS}")
    print("Max new tokens: 100")
    print("Temperature:    0.7")
    print("Sampling:       True")
    print("=" * 70)

    for model_index, model_name in enumerate(MODELS, start=1):
        print("\n" + "=" * 70)
        print(f"MODEL {model_index}/{len(MODELS)}")
        print(f"{model_name}")
        print("=" * 70)

        # Load model ONCE
        print("\nLoading model...")

        tokenizer, model, load_time = load_model(model_name)

        print(f"Load time: {load_time:.2f} seconds")
        print(f"Device: {next(model.parameters()).device}")

        run_metrics = []

        # Run inference 3 times
        for run in range(1, NUM_RUNS + 1):

            print("\n" + "-" * 70)
            print(f"RUN {run}/{NUM_RUNS}")
            print("-" * 70)

            response, metrics = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=100,
                temperature=0.7,
            )

            run_metrics.append(metrics)

            print("\nResponse:")
            print(response)

            print("\nMetrics:")
            print(f"Input tokens:       {metrics['input_tokens']}")
            print(f"Output tokens:      {metrics['output_tokens']}")
            print(f"TTFT:               {metrics['ttft']:.3f} seconds")
            print(
                f"Generation time:    "
                f"{metrics['generation_time']:.3f} seconds"
            )
            print(
                f"Tokens/second:      "
                f"{metrics['tokens_per_second']:.2f}"
            )
            print(
                f"Peak VRAM:          "
                f"{metrics['peak_vram_gb']:.2f} GB"
            )

        # Calculate averages
        avg_ttft = sum(
            m["ttft"] for m in run_metrics
        ) / NUM_RUNS

        avg_generation_time = sum(
            m["generation_time"] for m in run_metrics
        ) / NUM_RUNS

        avg_tokens_per_second = sum(
            m["tokens_per_second"] for m in run_metrics
        ) / NUM_RUNS

        avg_peak_vram = sum(
            m["peak_vram_gb"] for m in run_metrics
        ) / NUM_RUNS

        avg_output_tokens = sum(
            m["output_tokens"] for m in run_metrics
        ) / NUM_RUNS

        # Final model summary
        print("\n" + "=" * 70)
        print(f"AVERAGE RESULTS — {model_name}")
        print("=" * 70)

        print(f"Load time:          {load_time:.2f} seconds")
        print(f"Average output:     {avg_output_tokens:.2f} tokens")
        print(f"Average TTFT:       {avg_ttft:.3f} seconds")
        print(
            f"Average generation:"
            f" {avg_generation_time:.3f} seconds"
        )
        print(
            f"Average tokens/sec:"
            f" {avg_tokens_per_second:.2f}"
        )
        print(
            f"Average peak VRAM:"
            f" {avg_peak_vram:.2f} GB"
        )

        # Unload model
        print("\nUnloading model...")

        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("GPU memory cleared.")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
