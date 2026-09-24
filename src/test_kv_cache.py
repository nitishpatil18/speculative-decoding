"""
test_kv_cache.py - correctness checks for evict_streaming(), before trusting it
in any generation or memory benchmark.

test 1: cache never exceeds n_sink + n_window after eviction kicks in.
test 2: does generation stay coherent past the window size - the real risk is
        RoPE position handling breaking after tokens are evicted from the
        middle of the sequence, producing fluent-looking but broken text.
"""

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from baseline import load_config, load_model
from kv_cache import evict_streaming

N_SINK = 4
N_WINDOW = 32


def test_shapes():
    print("=== test 1: cache shape bookkeeping ===")
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    cache = DynamicCache()
    prompt = "Once upon a time, in a land far away, there lived a wise old wizard who"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated = input_ids
        for step in range(80):
            if step == 0:
                out = model(input_ids=generated, past_key_values=cache, use_cache=True)
            else:
                out = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            evict_streaming(cache, n_sink=N_SINK, n_window=N_WINDOW)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            cache_len = cache.layers[0].get_seq_length()
            expected_max = N_SINK + N_WINDOW
            if cache_len > expected_max:
                print(f"  FAIL at step {step}: cache_len={cache_len} exceeds max={expected_max}")
                return False

    print(f"  cache never exceeded {expected_max} entries across 80 steps: PASS")
    return True


def test_coherence():
    print("\n=== test 2: generation coherence after eviction kicks in ===")
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    cache = DynamicCache()
    prompt = "Once upon a time, in a land far away, there lived a wise old wizard who"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated = input_ids
        for step in range(150):
            if step == 0:
                out = model(input_ids=generated, past_key_values=cache, use_cache=True)
            else:
                out = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            evict_streaming(cache, n_sink=N_SINK, n_window=N_WINDOW)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  generated text (150 tokens, well past window of {N_WINDOW}):\n\n{text}\n")
    print("  READ THIS YOURSELF - coherent English, or degrading into repetition/garbage")
    print("  after ~36 tokens in? that's the actual failure signature to watch for.")


if __name__ == "__main__":
    shapes_ok = test_shapes()
    if shapes_ok:
        test_coherence()
    else:
        print("\nskipping coherence test - fix shape bookkeeping first")
