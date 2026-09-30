"""
debug_streaming_divergence.py - inspects the target model's ACTUAL logits at
the exact incremental step where speculative+streaming diverged from the
streaming baseline, using the real forward pass (not a reconstruction).
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model
from speculative_streaming import run_streaming_baseline_with_logit_capture

N_SINK = 4
N_WINDOW = 32
PROMPT = "Write a long, detailed story about a journey across a vast continent, describing many places, characters, and events along the way."

DIVERGE_AT_GENERATED_POSITION = 54
CAPTURE_STEP = DIVERGE_AT_GENERATED_POSITION - 1


def main():
    config = load_config()
    device = config["device"]
    dtype = config["dtype"]

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], dtype, device)

    tokens, captured_logits = run_streaming_baseline_with_logit_capture(
        target_model, tokenizer, PROMPT, max_new_tokens=60, device=device,
        n_sink=N_SINK, n_window=N_WINDOW, capture_at_step=CAPTURE_STEP,
    )

    prompt_len = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]
    actual_next_token = tokens[prompt_len + DIVERGE_AT_GENERATED_POSITION]
    print(f"actual next token generated at this position: {actual_next_token} -> {tokenizer.decode([actual_next_token])!r}")

    top5 = torch.topk(captured_logits, 5)
    print("\ntop-5 logits, captured from the REAL incremental forward pass:")
    for val, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {idx} ({tokenizer.decode([idx])!r}): {val:.4f}")

    matches = top5.indices[0].item() == actual_next_token
    print(f"\ntop-1 matches actual generated token: {matches}")
    margin = (top5.values[0] - top5.values[1]).item()
    print(f"top-1 vs top-2 margin: {margin:.6f}")
    print("small margin (~0.01-0.03) = fp16 near-tie, already-characterized.")
    print("large margin = something else is still wrong.")


if __name__ == "__main__":
    main()
