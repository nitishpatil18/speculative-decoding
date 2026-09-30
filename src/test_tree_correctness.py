"""
test_tree_correctness.py - proves tree-batched verification produces
IDENTICAL logits to running each path linearly and alone.

Fixed the same conceptual bug in the LINEAR comparison side: the logit
that verifies token at depth_pos in a path is read from the position of
ITS PARENT (prompt_len - 1 + depth_pos for a genuinely linear chain,
where parent always happens to be physically adjacent) - made explicit
here rather than implicitly assumed.
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model
from tree_speculative import build_draft_tree, verify_tree

BRANCH_FACTOR = 2
PROMPT = "The best way to learn a new skill is"


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    target_model = load_model(config["target_model"], config["dtype"], device)
    draft_model = load_model(config["draft_model"], config["dtype"], device)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    nodes, paths = build_draft_tree(draft_model, input_ids, BRANCH_FACTOR, device)
    print(f"tree has {len(nodes)} nodes, {len(paths)} root-to-leaf paths")
    for i, n in enumerate(nodes):
        print(f"  node {i}: token={tokenizer.decode([n['token_id']])!r}, parent={n['parent']}, depth={n['depth']}")

    tree_verify_logits = verify_tree(target_model, input_ids, nodes, device)

    print(f"\nchecking each node's VERIFICATION logit against a plain linear run of its path...")
    max_diff_overall = 0.0

    for path in paths:
        path_token_ids = [nodes[idx]["token_id"] for idx in path]
        linear_input = torch.cat([input_ids, torch.tensor([path_token_ids], device=device)], dim=1)

        with torch.no_grad():
            out_linear = target_model(input_ids=linear_input, use_cache=False)

        for depth_pos, node_idx in enumerate(path):
            # verifying logit for the token at depth_pos is read from its
            # PARENT's position: prompt_len - 1 + depth_pos, for a linear chain
            linear_verify_pos = prompt_len - 1 + depth_pos
            linear_logits = out_linear.logits[0, linear_verify_pos, :]
            tree_logits = tree_verify_logits[node_idx, :]

            diff = (linear_logits - tree_logits).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            print(f"  path {path}, node {node_idx} (depth {depth_pos+1}): max logit diff = {diff:.6f}")

    print(f"\noverall max difference across all nodes/paths: {max_diff_overall:.6f}")
    if max_diff_overall < 0.05:
        print("PASS: tree-batched verification matches linear per-path verification (within fp16 noise)")
    else:
        print("FAIL: still a real bug")


if __name__ == "__main__":
    main()
