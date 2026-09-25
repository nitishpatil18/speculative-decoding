"""
test_kv_cache.py - correctness checks for evict_streaming() + StreamingPositionTracker.

test 1: cache never exceeds n_sink + n_window.
test 2: does generation stay coherent past the window size, now that position_ids
        track TRUE elapsed steps instead of the capped cache length.
"""

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from baseline import load_config, load_model
from kv_cache import evict_streaming, StreamingPositionTracker

N_SINK = 4
N_WINDOW = 32


def generate_with_streaming(model, tokenizer, prompt, num_steps, device):
    cache = DynamicCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    tracker = StreamingPositionTracker(start=0)

    with torch.no_grad():
        for step in range(num_steps):
            if step == 0:
                position_ids = tracker.position_ids(generated.shape[1], device)
                out = model(input_ids=generated, position_ids=position_ids, past_key_values=cache, use_cache=True)
                tracker.advance(generated.shape[1])
            else:
                position_ids = tracker.position_ids(1, device)
                out = model(input_ids=generated[:, -1:], position_ids=position_ids, past_key_values=cache, use_cache=True)
                tracker.advance(1)

            cache = out.past_key_values
            evict_streaming(cache, n_sink=N_SINK, n_window=N_WINDOW)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            cache_len = cache.layers[0].get_seq_length()
            expected_max = N_SINK + N_WINDOW
            if cache_len > expected_max:
                print(f"  FAIL at step {step}: cache_len={cache_len} exceeds max={expected_max}")
                return generated, False

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated, True


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    prompt = "Once upon a time, in a land far away, there lived a wise old wizard who"

    print("=== test 1: cache shape bookkeeping (80 steps) ===")
    generated, shapes_ok = generate_with_streaming(model, tokenizer, prompt, num_steps=80, device=device)
    if shapes_ok:
        print(f"  cache never exceeded {N_SINK + N_WINDOW} entries: PASS")
    else:
        print("  FAIL - stopping before coherence test")
        return

    print("\n=== test 2: generation coherence (150 steps, well past window) ===")
    generated, _ = generate_with_streaming(model, tokenizer, prompt, num_steps=150, device=device)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  generated text:\n\n{text}\n")
    print("  READ THIS YOURSELF - coherent English past ~36 tokens, or still degrading?")


if __name__ == "__main__":
    main()
