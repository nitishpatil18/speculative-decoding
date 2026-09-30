"""
test_speculative_streaming.py - correctness check: speculative decoding +
streaming cache must match a streaming baseline (target model alone, same
eviction policy) - NOT the unbounded baseline, which is a different system
once eviction is involved.
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model
from speculative_streaming import run_streaming_baseline, speculative_decode_streaming

N_SINK = 4
N_WINDOW = 32
K = 2
MAX_NEW_TOKENS = 200
PROMPT = "Write a long, detailed story about a journey across a vast continent, describing many places, characters, and events along the way."


def main():
    config = load_config()
    device = config["device"]
    dtype = config["dtype"]

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], dtype, device)
    draft_model = load_model(config["draft_model"], dtype, device)

    print("warming up...")
    run_streaming_baseline(target_model, tokenizer, "Warmup.", 10, device, N_SINK, N_WINDOW)
    speculative_decode_streaming(draft_model, target_model, tokenizer, "Warmup.", 10, K, device, N_SINK, N_WINDOW)

    print(f"\nrunning streaming baseline ({MAX_NEW_TOKENS} tokens, window={N_WINDOW})...")
    baseline_result = run_streaming_baseline(target_model, tokenizer, PROMPT, MAX_NEW_TOKENS, device, N_SINK, N_WINDOW)

    print(f"running speculative + streaming (k={K})...")
    spec_result = speculative_decode_streaming(draft_model, target_model, tokenizer, PROMPT, MAX_NEW_TOKENS, K, device, N_SINK, N_WINDOW)

    match = baseline_result["generated_text"] == spec_result["generated_text"]
    speedup = spec_result["tokens_per_sec"] / baseline_result["tokens_per_sec"]

    print(f"\nstreaming baseline: {baseline_result['tokens_per_sec']:.2f} tok/s")
    print(f"spec + streaming:   {spec_result['tokens_per_sec']:.2f} tok/s ({speedup:.2f}x)")
    print(f"acceptance rate: {spec_result['acceptance_rate']:.2%}")
    print(f"outputs match: {match}")

    if not match:
        min_len = min(len(baseline_result["token_ids"]), len(spec_result["token_ids"]))
        for i in range(min_len):
            if baseline_result["token_ids"][i] != spec_result["token_ids"][i]:
                print(f"\nfirst divergence at position {i}")
                print(f"baseline: {tokenizer.decode([baseline_result['token_ids'][i]])!r}")
                print(f"spec:     {tokenizer.decode([spec_result['token_ids'][i]])!r}")
                break


if __name__ == "__main__":
    main()
