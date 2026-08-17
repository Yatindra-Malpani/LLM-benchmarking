from src.framework.config import MODEL_NAME
from src.framework.loader import load_model
from src.framework.inference import generate_response

def main():
    print("=" * 60)
    print("Qwen 1.5B — Generation Test")
    print("=" * 60)

    print(f"\nModel: {MODEL_NAME}")

    tokenizer, model, load_time = load_model()

    messages = [
        {
            "role": "user",
            "content": "Explain what a transformer is in simple terms.",
        }
    ]

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=100,
    )

    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
