"""
evaluate_quality.py - quantifies the coherence cost of streaming eviction with
teacher-forced perplexity, rather than eyeballing generated text.

Method: generate a reference continuation with an unbounded cache (greedy,
self-consistent by construction - this is the "ground truth" sequence).
Then replay that EXACT same token sequence through both an unbounded cache
and a streaming cache under teacher forcing, recording the model's loss at
predicting each actual next token. Comparing losses before vs after the
eviction window fills isolates exactly how much information loss eviction
costs, in bits, rather than relying on spotting an anecdote like a pronoun
drift.
"""

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from baseline import load_config, load_model
from kv_cache import evict_streaming, StreamingPositionTracker

N_SINK = 4
N_WINDOW = 256
REFERENCE_LENGTH = 1200


def generate_reference(model, tokenizer, prompt, num_new_tokens, device):
    cache = DynamicCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    with torch.no_grad():
        for step in range(num_new_tokens):
            if step == 0:
                out = model(input_ids=generated, past_key_values=cache, use_cache=True)
            else:
                out = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return generated


def teacher_forced_losses(model, token_ids, device, streaming, n_sink=N_SINK, n_window=N_WINDOW):
    cache = DynamicCache()
    tracker = StreamingPositionTracker(start=0) if streaming else None
    seq_len = token_ids.shape[1]
    losses = []

    with torch.no_grad():
        for step in range(seq_len - 1):
            cur_input = token_ids[:, step : step + 1]
            target_id = token_ids[0, step + 1].item()

            if streaming:
                position_ids = tracker.position_ids(1, device)
                out = model(input_ids=cur_input, position_ids=position_ids, past_key_values=cache, use_cache=True)
                tracker.advance(1)
            else:
                out = model(input_ids=cur_input, past_key_values=cache, use_cache=True)

            cache = out.past_key_values
            if streaming:
                evict_streaming(cache, n_sink=n_sink, n_window=n_window)

            logits = out.logits[:, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -log_probs[0, target_id].item()
            losses.append(loss)

    return losses


def summarize(losses, cutoff, label):
    pre = losses[:cutoff]
    post = losses[cutoff:]
    pre_ppl = torch.exp(torch.tensor(sum(pre) / len(pre))).item() if pre else float("nan")
    post_ppl = torch.exp(torch.tensor(sum(post) / len(post))).item() if post else float("nan")
    print(f"  {label}:")
    print(f"    pre-eviction  (positions 0-{cutoff}):   avg loss={sum(pre)/len(pre):.4f}, ppl={pre_ppl:.2f}")
    print(f"    post-eviction (positions {cutoff}-{len(losses)}): avg loss={sum(post)/len(post):.4f}, ppl={post_ppl:.2f}")
    return pre_ppl, post_ppl


def main():
    config = load_config()
    device = config["device"]
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], config["dtype"], device)

    prompt = "Write a long, detailed story about a journey across a vast continent, describing many places, characters, and events along the way."

    print(f"generating {REFERENCE_LENGTH}-token reference sequence (unbounded, greedy)...")
    reference = generate_reference(model, tokenizer, prompt, REFERENCE_LENGTH, device)
    print(f"  reference length: {reference.shape[1]} tokens")

    cutoff = N_SINK + N_WINDOW

    print("\nreplaying reference under unbounded cache (control - should be self-consistent)...")
    unbounded_losses = teacher_forced_losses(model, reference, device, streaming=False)
    unbounded_pre, unbounded_post = summarize(unbounded_losses, cutoff, "unbounded")

    print("\nreplaying reference under streaming cache (measuring eviction cost)...")
    streaming_losses = teacher_forced_losses(model, reference, device, streaming=True)
    streaming_pre, streaming_post = summarize(streaming_losses, cutoff, "streaming")

    print(f"\n{'='*50}")
    ppl_increase = ((streaming_post / unbounded_post) - 1) * 100
    print(f"post-eviction perplexity increase from streaming cache: {ppl_increase:+.1f}%")
    print(f"(pre-eviction match check: unbounded={unbounded_pre:.2f} vs streaming={streaming_pre:.2f} - should be near-identical, confirms no bug before window fills)")


if __name__ == "__main__":
    main()
