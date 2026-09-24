"""
benchmark.py - compares baseline vs fixed-k speculative vs adaptive-k speculative.
verifies correctness (outputs must match exactly) before trusting any speedup number.
"""

import json
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import load_config, load_model, run_baseline, TEST_PROMPTS
from speculative_decode import speculative_decode
from adaptive_policy import adaptive_speculative_decode

RESULTS_PATH = Path(__file__).parent.parent / "results" / "runs.jsonl"
FIXED_K = 2  # near-optimal fixed k, established from the k=2/4/8 sweep


def main():
    config = load_config()
    device = config["device"]
    dtype = config["dtype"]
    max_new_tokens = config["max_new_tokens"]

    print(f"loading target model: {config['target_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], dtype, device)

    print(f"loading draft model: {config['draft_model']}")
    draft_model = load_model(config["draft_model"], dtype, device)

    print("warming up mps backend...")
    run_baseline(target_model, tokenizer, "Warmup.", max_new_tokens=10, device=device)
    speculative_decode(draft_model, target_model, tokenizer, "Warmup.", max_new_tokens=10, k=FIXED_K, device=device)
    adaptive_speculative_decode(draft_model, target_model, tokenizer, "Warmup.", max_new_tokens=10, device=device)

    mismatches = 0

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] prompt: {prompt[:50]}...")

        baseline_result = run_baseline(target_model, tokenizer, prompt, max_new_tokens, device)
        fixed_result = speculative_decode(draft_model, target_model, tokenizer, prompt, max_new_tokens, FIXED_K, device)
        adaptive_result = adaptive_speculative_decode(draft_model, target_model, tokenizer, prompt, max_new_tokens, device)

        fixed_match = baseline_result["generated_text"] == fixed_result["generated_text"]
        adaptive_match = baseline_result["generated_text"] == adaptive_result["generated_text"]
        if not fixed_match or not adaptive_match:
            mismatches += 1
            print(f"  ⚠ MISMATCH — fixed_match={fixed_match}, adaptive_match={adaptive_match}")

        fixed_speedup = fixed_result["tokens_per_sec"] / baseline_result["tokens_per_sec"]
        adaptive_speedup = adaptive_result["tokens_per_sec"] / baseline_result["tokens_per_sec"]

        print(f"  baseline:    {baseline_result['tokens_per_sec']:.2f} tok/s")
        print(f"  fixed k={FIXED_K}:   {fixed_result['tokens_per_sec']:.2f} tok/s ({fixed_speedup:.2f}x)")
        print(f"  adaptive:    {adaptive_result['tokens_per_sec']:.2f} tok/s ({adaptive_speedup:.2f}x), avg_k={adaptive_result['avg_k']:.2f}")
        print(f"  outputs match — fixed: {fixed_match}, adaptive: {adaptive_match}")

        log_entry = {
            "prompt": prompt,
            "baseline_tok_s": baseline_result["tokens_per_sec"],
            "fixed_k": FIXED_K,
            "fixed_tok_s": fixed_result["tokens_per_sec"],
            "fixed_speedup": fixed_speedup,
            "fixed_match": fixed_match,
            "adaptive_tok_s": adaptive_result["tokens_per_sec"],
            "adaptive_speedup": adaptive_speedup,
            "adaptive_avg_k": adaptive_result["avg_k"],
            "adaptive_match": adaptive_match,
        }
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\n{'='*50}")
    print(f"total mismatches: {mismatches}/{len(TEST_PROMPTS)}")
    if mismatches == 0:
        print("correctness confirmed: both fixed-k and adaptive-k identical to baseline")
    else:
        print("correctness FAILED — do not trust speedup numbers until this is fixed")


if __name__ == "__main__":
    main()
