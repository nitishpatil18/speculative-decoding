"""
tree_speculative.py - tree-based speculative decoding.

Branch factor 2, depth 2: draft model proposes top-2 tokens at depth 1,
then top-2 continuations for each, forming 6 tree nodes and 4 root-to-leaf
candidate paths, verified by the target model in ONE batched forward pass
using a tree-structured attention mask.

No KV-cache persistence across rounds in this version - each round
recomputes draft and target forward passes from the full current sequence.
Correctness first, matching this project's established priority; caching
tree-based verification is real additional complexity deferred as future
work, same as how linear speculative decoding was built correctness-first
before optimizing.

BUG HISTORY (see full details in commit history / test_tree_correctness.py):
1. verify_tree originally read logits by flat array position instead of
   actual tree-parent position.
2. build_tree_attention's mask was originally constructed inverted,
   breaking causality WITHIN the prompt itself.
Both fixed and verified: node logits match linear per-path runs within
fp16 noise (max diff 0.023).
"""

import torch


def build_draft_tree(draft_model, input_ids, branch_factor, device):
    nodes = []

    with torch.no_grad():
        out = draft_model(input_ids=input_ids, use_cache=False)
        depth1_logits = out.logits[0, -1, :]
        depth1_top = torch.topk(depth1_logits, branch_factor).indices.tolist()

        for token_id in depth1_top:
            nodes.append({"token_id": token_id, "parent": None, "depth": 1})

        paths = []
        for parent_idx, parent_token in enumerate(depth1_top):
            extended = torch.cat([input_ids, torch.tensor([[parent_token]], device=device)], dim=1)
            out2 = draft_model(input_ids=extended, use_cache=False)
            depth2_logits = out2.logits[0, -1, :]
            depth2_top = torch.topk(depth2_logits, branch_factor).indices.tolist()

            for token_id in depth2_top:
                child_idx = len(nodes)
                nodes.append({"token_id": token_id, "parent": parent_idx, "depth": 2})
                paths.append([parent_idx, child_idx])

    return nodes, paths


def build_tree_attention(input_ids, nodes, device):
    prompt_len = input_ids.shape[1]
    num_nodes = len(nodes)
    seq_len = prompt_len + num_nodes

    tree_token_ids = torch.tensor([[n["token_id"] for n in nodes]], device=device)
    extended_input_ids = torch.cat([input_ids, tree_token_ids], dim=1)

    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    for i, node in enumerate(nodes):
        abs_i = prompt_len + i
        ancestors = set()
        walk = node["parent"]
        while walk is not None:
            ancestors.add(prompt_len + walk)
            walk = nodes[walk]["parent"]

        for j in range(num_nodes):
            abs_j = prompt_len + j
            if abs_j == abs_i or abs_j >= abs_i:
                continue
            if abs_j not in ancestors:
                mask[0, 0, abs_i, abs_j] = float("-inf")

    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    tree_positions = torch.tensor([[prompt_len + n["depth"] - 1 for n in nodes]], device=device)
    position_ids = torch.cat([position_ids, tree_positions], dim=1)

    return extended_input_ids, mask, position_ids


def node_verify_position(node, prompt_len, nodes):
    if node["parent"] is None:
        return prompt_len - 1
    return prompt_len + node["parent"]


def verify_tree(target_model, input_ids, nodes, device):
    """
    Returns (verify_logits, self_logits), each shape (num_nodes, vocab):
      verify_logits[i] - the distribution that VERIFIES node i's own token
                          (read from node i's parent's position).
      self_logits[i]   - node i's OWN output position's logits, i.e. the
                          prediction for whatever comes AFTER node i. Used
                          for the bonus-token case when a full path is
                          accepted - free from this same forward pass, no
                          extra computation needed.
    """
    extended_input_ids, mask, position_ids = build_tree_attention(input_ids, nodes, device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = target_model(
            input_ids=extended_input_ids,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            use_cache=False,
        )

    verify_logits = []
    self_logits = []
    for i, node in enumerate(nodes):
        pos = node_verify_position(node, prompt_len, nodes)
        verify_logits.append(out.logits[0, pos, :])
        self_logits.append(out.logits[0, prompt_len + i, :])

    return torch.stack(verify_logits, dim=0), torch.stack(self_logits, dim=0)


def select_best_path(nodes, paths, verify_logits):
    """
    For each root-to-leaf path, walks it and counts how many leading nodes
    match the target's argmax at that node's verify position (i.e. how
    many draft tokens the target would have actually chosen too). Returns
    the path with the longest accepted prefix, and that prefix length.
    Ties broken by first path found - arbitrary but deterministic.
    """
    best_path = None
    best_len = -1

    for path in paths:
        length = 0
        for node_idx in path:
            target_choice = verify_logits[node_idx].argmax().item()
            if target_choice == nodes[node_idx]["token_id"]:
                length += 1
            else:
                break
        if length > best_len:
            best_len = length
            best_path = path

    return best_path, best_len


def run_tree_speculative_decode(draft_model, target_model, tokenizer, prompt, max_new_tokens, branch_factor, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        nodes, paths = build_draft_tree(draft_model, input_ids, branch_factor, device)
        verify_logits, self_logits = verify_tree(target_model, input_ids, nodes, device)

        best_path, best_len = select_best_path(nodes, paths, verify_logits)

        for node_idx in best_path[:best_len]:
            token_id = nodes[node_idx]["token_id"]
            input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=device)], dim=1)
            tokens_generated += 1
            if token_id == tokenizer.eos_token_id:
                return input_ids[0, :prompt_len + tokens_generated].tolist()

        if best_len < len(best_path):
            mismatch_node = best_path[best_len]
            correction_token = verify_logits[mismatch_node].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[correction_token]], device=device)], dim=1)
            tokens_generated += 1
            if correction_token == tokenizer.eos_token_id:
                break
        else:
            leaf_node = best_path[-1]
            bonus_token = self_logits[leaf_node].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[bonus_token]], device=device)], dim=1)
            tokens_generated += 1
            if bonus_token == tokenizer.eos_token_id:
                break

        if tokens_generated >= max_new_tokens:
            overshoot = tokens_generated - max_new_tokens
            if overshoot > 0:
                input_ids = input_ids[:, :-overshoot]
            break

    return input_ids[0].tolist()
