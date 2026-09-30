"""
tree_speculative.py - tree-based speculative decoding, one round.

BUG HISTORY - two real bugs found here, both worth recording:

1. verify_tree originally read "verification logits" via flat physical
   array adjacency, not actual tree-parent position. Fixed by reading each
   node's logits from node_verify_position (its real parent's absolute
   position).

2. build_tree_attention originally built the mask as "-inf everywhere,
   then explicitly allow prompt visibility" via
   mask[0,0,:,:prompt_len] = 0.0 for ALL rows. This unconditionally let
   even EARLY prompt rows see LATER prompt columns, breaking ordinary
   causality inside the prompt itself. Since transformer layers stack,
   that corruption cascaded through every later layer/position - which is
   why even depth-1 node logits (derived from the last prompt position)
   came out wrong by ~13, not just tree-to-tree relationships.

   FIXED by building a completely standard causal mask for the full
   extended sequence FIRST (prompt-internal causality and prompt-to-tree
   visibility both fall out of this for free, since the prompt genuinely
   precedes all tree nodes), then explicitly BLOCKING only the
   non-ancestor tree-to-tree edges on top of that - the reverse of the
   original approach.
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
    """
    Standard causal mask for the WHOLE extended sequence first (correct
    prompt-internal causality, correct prompt-to-tree visibility, for
    free). Then explicitly block non-ancestor tree-to-tree edges on top -
    NOT the reverse.
    """
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
    for node in nodes:
        pos = node_verify_position(node, prompt_len, nodes)
        verify_logits.append(out.logits[0, pos, :])

    return torch.stack(verify_logits, dim=0)
