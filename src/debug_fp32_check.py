"""
debug_fp32_check.py - reruns the exact divergence case from debug_divergence.py
in fp32 instead of fp16, to confirm precision is the full explanation.
if fp32 shows no divergence, precision was the complete cause.
if fp32 still diverges, something else is also contributing.
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model, TEST_PROMPTS
from adaptive_policy import adaptive_speculative_decode


def run_baseline_tokens(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    past_key_values = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                out = model(input_ids=generated, use_cache=True)
            else:
                out = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return generated[0].tolist()


def main():
    config = load_config()
    device = config["device"]
    max_new_tokens = config["max_new_tokens"]
    prompt = TEST_PROMPTS[2]  # same French Revolution prompt that diverged in fp16

    print("loading models in fp32...")
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], "float32", device)
    draft_model = load_model(config["draft_model"], "float32", device)

    print("running baseline (fp32)...")
    baseline_tokens = run_baseline_tokens(target_model, tokenizer, prompt, max_new_tokens, device)

    print("running adaptive (fp32)...")
    adaptive_result = adaptive_speculative_decode(draft_model, target_model, tokenizer, prompt, max_new_tokens, device)
    adaptive_tokens = adaptive_result["token_ids"]

    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    min_len = min(len(baseline_tokens), len(adaptive_tokens))

    diverge_at = None
    for i in range(prompt_len, min_len):
        if baseline_tokens[i] != adaptive_tokens[i]:
            diverge_at = i
            break

    if diverge_at is None:
        print("\nno divergence in fp32 — outputs identical.")
        print("this confirms fp16 rounding was the full explanation for the earlier mismatch.")
    else:
        print(f"\nfp32 still diverges at position {diverge_at - prompt_len}")
        print(f"baseline token: {baseline_tokens[diverge_at]} -> {tokenizer.decode([baseline_tokens[diverge_at]])!r}")
        print(f"adaptive token: {adaptive_tokens[diverge_at]} -> {tokenizer.decode([adaptive_tokens[diverge_at]])!r}")
        print("this means precision alone does not fully explain the mismatch — something else is contributing.")


if __name__ == "__main__":
    main()
