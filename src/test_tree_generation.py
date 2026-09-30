"""
test_tree_generation.py - end-to-end correctness: multi-round tree-based
speculative decoding output must be BYTE-IDENTICAL to plain greedy
baseline, since every accept/correct/bonus decision defers to the
target's own argmax. No KV-cache reuse in this version, so this will be
slower than linear speculative decoding - correctness first.
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model, run_baseline
from tree_speculative import run_tree_speculative_decode

BRANCH_FACTOR = 2
MAX_NEW_TOKENS = 40
PROMPT = "The best way to learn a new skill is"


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], config["dtype"], device)
    draft_model = load_model(config["draft_model"], config["dtype"], device)

    print(f"running plain greedy baseline ({MAX_NEW_TOKENS} tokens)...")
    baseline_result = run_baseline(target_model, tokenizer, PROMPT, MAX_NEW_TOKENS, device)
    print(f"  baseline text: {baseline_result['generated_text']!r}")

    print(f"\nrunning tree speculative decode (branch_factor={BRANCH_FACTOR})...")
    tree_token_ids = run_tree_speculative_decode(draft_model, target_model, tokenizer, PROMPT, MAX_NEW_TOKENS, BRANCH_FACTOR, device)
    tree_text = tokenizer.decode(tree_token_ids, skip_special_tokens=True)
    print(f"  tree text:     {tree_text!r}")

    match = baseline_result["generated_text"] == tree_text
    print(f"\noutputs match: {match}")
    if not match:
        print("MISMATCH - real bug, needs root-causing before trusting this further")


if __name__ == "__main__":
    main()
