import time
import torch
from transformers import TextIteratorStreamer
from threading import Thread

def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens=100,
    temperature=0.7,
):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = inputs.to(next(model.parameters()).device)

    input_length = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "streamer": streamer,
    }

    start_time = time.perf_counter()

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )

    thread.start()

    first_token_time = None
    response_parts = []

    for text in streamer:
        if first_token_time is None:
            torch.cuda.synchronize()
            first_token_time = time.perf_counter()

        response_parts.append(text)

    thread.join()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    response = "".join(response_parts)

    ttft = first_token_time - start_time

    generation_time = end_time - start_time

    output_tokens = len(
        tokenizer.encode(
            response,
            add_special_tokens=False,
        )
    )
    tokens_per_second = (
        output_tokens / generation_time
        if generation_time > 0
        else 0
    )

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    metrics = {
        "input_tokens": input_length,
        "output_tokens": output_tokens,
        "ttft": ttft,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "peak_vram_gb": peak_vram,
    }
    return response, metrics
