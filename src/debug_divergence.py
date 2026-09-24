"""
debug_divergence.py - locates the exact token where adaptive-k diverges from
baseline, then checks whether it's a genuine bug or floating-point
non-determinism between single-token and batched (cached-prefix) forward passes.
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
    dtype = config["dtype"]
    max_new_tokens = config["max_new_tokens"]
    prompt = TEST_PROMPTS[2]  # French Revolution prompt - the one that mismatched

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], dtype, device)
    draft_model = load_model(config["draft_model"], dtype, device)

    print("running baseline...")
    baseline_tokens = run_baseline_tokens(target_model, tokenizer, prompt, max_new_tokens, device)

    print("running adaptive...")
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
        print("no divergence found in overlapping range")
        return

    print(f"\nfirst divergence at generated position {diverge_at - prompt_len}")
    print(f"baseline token: {baseline_tokens[diverge_at]} -> {tokenizer.decode([baseline_tokens[diverge_at]])!r}")
    print(f"adaptive token: {adaptive_tokens[diverge_at]} -> {tokenizer.decode([adaptive_tokens[diverge_at]])!r}")

    context = torch.tensor([baseline_tokens[:diverge_at]]).to(device)

    with torch.no_grad():
        # path A: pure single-context forward, same as baseline's ground-truth computation
        single_out = target_model(input_ids=context, use_cache=False)
        single_logits = single_out.logits[0, -1, :]

        # path B: cached-prefix + batched-last-2-tokens forward, mirrors how verify_input
        # actually computes logits during speculative verification
        if context.shape[1] > 2:
            prime_out = target_model(input_ids=context[:, :-2], use_cache=True)
            multi_out = target_model(input_ids=context[:, -2:], past_key_values=prime_out.past_key_values, use_cache=False)
            multi_logits = multi_out.logits[0, -1, :]
        else:
            multi_logits = single_logits

    top5_single = torch.topk(single_logits, 5)
    top5_multi = torch.topk(multi_logits, 5)

    print("\nsingle-context top-5 logits (baseline's actual computation path):")
    for val, idx in zip(top5_single.values.tolist(), top5_single.indices.tolist()):
        print(f"  {idx} ({tokenizer.decode([idx])!r}): {val:.4f}")

    print("\ncached-prefix + batched-2 top-5 logits (verify's actual computation path):")
    for val, idx in zip(top5_multi.values.tolist(), top5_multi.indices.tolist()):
        print(f"  {idx} ({tokenizer.decode([idx])!r}): {val:.4f}")

    margin = (top5_single.values[0] - top5_single.values[1]).item()
    print(f"\ntop-1 vs top-2 margin, single-context path: {margin:.6f}")
    print("a small margin here (well under 1.0) supports fp16 rounding as the cause.")
    print("a large margin would mean this is a real logic bug, not floating-point noise.")


if __name__ == "__main__":
    main()
