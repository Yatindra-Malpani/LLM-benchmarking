import torch
def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens=100,
    temperature=0.7,
):
    """
    Generate a response from a chat-based causal language model.
    """

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move all tokenizer outputs to the model's device.
    inputs = inputs.to(next(model.parameters()).device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response
