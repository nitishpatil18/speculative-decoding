"""
test_tree_mask.py - verifies a genuinely NON-LINEAR attention mask actually
changes model behavior in the expected way, before building tree-based
speculative decoding on top of it.

Setup: a 2-token prompt, then two SIBLING branch tokens at the same depth
(positions 2 and 3), which should NOT be able to attend to each other -
only to the shared prompt before them. This is the one property a flat
causal sequence structurally cannot express, so it's the one property
worth directly testing.

Test: token at position 3 (branch B) should produce IDENTICAL logits
whether or not there's a real, different token sitting at position 2
(branch A) - because a correct tree mask blocks position 3 from seeing
position 2 entirely. If changing branch A's token content changes branch
B's logits, the mask leaked and siblings are seeing each other - a bug
that would silently corrupt tree-based verification.
"""

import torch
from transformers import AutoTokenizer
from baseline import load_config, load_model


def build_tree_mask(seq_len, sibling_pairs, device):
    """
    Standard causal mask, plus explicit blocking between given sibling
    position pairs (blocked both directions) that would otherwise be
    visible to each other under plain causal masking.
    """
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    for i, j in sibling_pairs:
        mask[0, 0, i, j] = float("-inf")
        mask[0, 0, j, i] = float("-inf")
    return mask


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    prompt_ids = tokenizer("The weather today is", return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    branch_a_token = tokenizer(" sunny", add_special_tokens=False).input_ids[0]
    branch_a_alt_token = tokenizer(" freezing", add_special_tokens=False).input_ids[0]
    branch_b_token = tokenizer(" and", add_special_tokens=False).input_ids[0]

    seq_len = prompt_len + 2
    sibling_pairs = [(prompt_len, prompt_len + 1)]

    def run(branch_a_val):
        input_ids = torch.cat([
            prompt_ids,
            torch.tensor([[branch_a_val, branch_b_token]], device=device),
        ], dim=1)
        mask = build_tree_mask(seq_len, sibling_pairs, device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask={"full_attention": mask}, use_cache=False)
        return out.logits[0, -1, :]

    logits_a = run(branch_a_token)
    logits_a_alt = run(branch_a_alt_token)

    diff = (logits_a - logits_a_alt).abs().max().item()
    print(f"branch B's logits, changing branch A's token content (should be ~0 if tree mask is correct):")
    print(f"  max difference: {diff:.6f}")

    if diff < 0.05:
        print("  PASS: branch B is correctly blind to branch A's content")
    else:
        print("  FAIL: branch B's output changed based on sibling content - mask is leaking")

    plain_causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    input_ids_a = torch.cat([prompt_ids, torch.tensor([[branch_a_token, branch_b_token]], device=device)], dim=1)
    input_ids_a_alt = torch.cat([prompt_ids, torch.tensor([[branch_a_alt_token, branch_b_token]], device=device)], dim=1)
    with torch.no_grad():
        out_plain = model(input_ids=input_ids_a, attention_mask={"full_attention": plain_causal_mask}, use_cache=False)
        out_plain_alt = model(input_ids=input_ids_a_alt, attention_mask={"full_attention": plain_causal_mask}, use_cache=False)
    plain_diff = (out_plain.logits[0, -1, :] - out_plain_alt.logits[0, -1, :]).abs().max().item()
    print(f"\ncontrol: same test under PLAIN causal mask (branch B SHOULD see branch A here):")
    print(f"  max difference: {plain_diff:.6f}")
    if plain_diff > 0.05:
        print("  PASS: plain causal mask correctly lets branch B see branch A (confirms the diff metric itself is sensitive)")
    else:
        print("  INCONCLUSIVE: even plain causal shows no difference - something else is wrong with this test setup")


if __name__ == "__main__":
    main()
