"""
benchmark_memory.py - compares peak memory: unbounded cache vs streaming
(bounded) cache, over a long generation run where the difference should
actually become visible (short runs don't show it - see the per-token
cache size estimate that motivated this file).
"""

import torch
import time
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from baseline import load_config, load_model
from kv_cache import evict_streaming, StreamingPositionTracker

N_SINK = 4
N_WINDOW = 256  # larger window than the coherence test, since 32 was deliberately
                # aggressive to stress-test position tracking - a real deployment
                # would use something like this for better quality
GEN_LENGTH = 4000
SAMPLE_EVERY = 200  # record memory every N steps, not every step - avoids
                     # torch.mps sync overhead dominating the timing


def generate(model, tokenizer, prompt, num_steps, device, streaming, n_sink=N_SINK, n_window=N_WINDOW):
    cache = DynamicCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    tracker = StreamingPositionTracker(start=0) if streaming else None

    memory_samples = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for step in range(num_steps):
            if step == 0:
                if streaming:
                    position_ids = tracker.position_ids(generated.shape[1], device)
                    out = model(input_ids=generated, position_ids=position_ids, past_key_values=cache, use_cache=True)
                    tracker.advance(generated.shape[1])
                else:
                    out = model(input_ids=generated, past_key_values=cache, use_cache=True)
            else:
                if streaming:
                    position_ids = tracker.position_ids(1, device)
                    out = model(input_ids=generated[:, -1:], position_ids=position_ids, past_key_values=cache, use_cache=True)
                    tracker.advance(1)
                else:
                    out = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)

            cache = out.past_key_values
            if streaming:
                evict_streaming(cache, n_sink=n_sink, n_window=n_window)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if step % SAMPLE_EVERY == 0:
                torch.mps.synchronize()
                mem_mb = torch.mps.current_allocated_memory() / (1024**2)
                memory_samples.append((step, mem_mb))
                elapsed = time.perf_counter() - start_time
                print(f"    step {step}: mem={mem_mb:.1f}MB, elapsed={elapsed:.1f}s")

            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = time.perf_counter() - start_time
    return generated, memory_samples, total_time


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    prompt = "Write a long, detailed story about a journey across a vast continent, describing many places, characters, and events along the way."

    print(f"=== unbounded cache, {GEN_LENGTH} tokens ===")
    _, unbounded_samples, unbounded_time = generate(model, tokenizer, prompt, GEN_LENGTH, device, streaming=False)

    print(f"\n=== streaming cache (n_sink={N_SINK}, n_window={N_WINDOW}), {GEN_LENGTH} tokens ===")
    _, streaming_samples, streaming_time = generate(model, tokenizer, prompt, GEN_LENGTH, device, streaming=True)

    print(f"\n{'='*50}")
    print(f"unbounded: peak={max(m for _, m in unbounded_samples):.1f}MB, time={unbounded_time:.1f}s")
    print(f"streaming: peak={max(m for _, m in streaming_samples):.1f}MB, time={streaming_time:.1f}s")
    print(f"memory reduction: {(1 - max(m for _, m in streaming_samples) / max(m for _, m in unbounded_samples)) * 100:.1f}%")


if __name__ == "__main__":
    main()
