"""
speculative_streaming.py - speculative decoding with a bounded (StreamingLLM)
kv-cache on the target model.

SCOPE NOTE: "correctness" means identical output to run_streaming_baseline()
- the target model alone under the same eviction policy - not the unbounded
baseline. Eviction itself already changes context regardless of speculative
decoding (see evaluate_quality.py, +15.2% perplexity post-eviction); this
file only has to prove speculative decoding adds no FURTHER divergence.

BUG HISTORY (both wrong theories kept here since they ruled out real
possibilities and the next person debugging this should not re-waste time
on them):
  1. WRONG: "position tracker desync" - checking the actual trace showed
     tracker positions and cache lengths were both correct and consistent.
  2. PARTIALLY WRONG: "transient overshoot during batched verification,
     fixable by shrinking the eviction cap by k" - shrinking the cap did
     NOT fix the divergence (it just moved earlier, position 54 -> 40),
     proving capacity was never the issue.

ACTUAL CAUSE: eviction GRANULARITY, not capacity. The streaming baseline
evicts after EVERY SINGLE TOKEN (one eviction op per token). A speculative
round adds 1-3 tokens in one batch, then evicts ONCE for the whole batch.
Evicting once after 3 tokens does not equal evicting 3 times, once after
each token, whenever the window boundary falls inside that span - "the
last n_window entries as of now" differs depending on whether you check
that after every token or only after a multi-token batch.

FIX: loop the eviction call once per token actually added this round,
replicating the baseline's exact per-token cadence, instead of one
eviction call per round regardless of how many tokens landed.
"""

import time
import torch
from transformers.cache_utils import DynamicCache
from kv_cache import evict_streaming, StreamingPositionTracker


def run_streaming_baseline(model, tokenizer, prompt, max_new_tokens, device, n_sink, n_window):
    cache = DynamicCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    tracker = StreamingPositionTracker(start=0)
    step_times = []

    with torch.no_grad():
        torch.mps.synchronize()
        start = time.perf_counter()
        position_ids = tracker.position_ids(generated.shape[1], device)
        out = model(input_ids=generated, position_ids=position_ids, past_key_values=cache, use_cache=True)
        tracker.advance(generated.shape[1])
        cache = out.past_key_values
        evict_streaming(cache, n_sink=n_sink, n_window=n_window)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        torch.mps.synchronize()
        step_times.append(time.perf_counter() - start)

        for _ in range(max_new_tokens - 1):
            torch.mps.synchronize()
            start = time.perf_counter()
            position_ids = tracker.position_ids(1, device)
            out = model(input_ids=next_token, position_ids=position_ids, past_key_values=cache, use_cache=True)
            tracker.advance(1)
            cache = out.past_key_values
            evict_streaming(cache, n_sink=n_sink, n_window=n_window)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            torch.mps.synchronize()
            step_times.append(time.perf_counter() - start)

            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = sum(step_times)
    tokens_generated = len(step_times)
    return {
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "token_ids": generated[0].tolist(),
        "tokens_generated": tokens_generated,
        "total_time_s": total_time,
        "tokens_per_sec": tokens_generated / total_time if total_time > 0 else 0.0,
    }


def evict_streaming_per_token(cache, n_sink, n_window, num_tokens_added):
    """
    Calls evict_streaming once per token in num_tokens_added, matching the
    baseline's exact per-token cadence - NOT one call covering all tokens
    at once, which is mathematically different once more than 1 token
    lands in a single round. Each call only meaningfully changes anything
    once total length exceeds n_sink+n_window; calling it "too early" is a
    harmless no-op (see evict_streaming's own seq_len <= max_len check).
    """
    for _ in range(num_tokens_added):
        evict_streaming(cache, n_sink=n_sink, n_window=n_window)


def speculative_decode_streaming(draft_model, target_model, tokenizer, prompt, max_new_tokens, k, device, n_sink, n_window):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    draft_past = None
    prompt_len = input_ids.shape[1]
    tokens_generated = 0
    accepted_lengths = []
    step_times = []
    tracker = StreamingPositionTracker(start=0)

    with torch.no_grad():
        if input_ids.shape[1] > 1:
            prime_position_ids = tracker.position_ids(input_ids.shape[1] - 1, device)
            prime_out = target_model(input_ids=input_ids[:, :-1], position_ids=prime_position_ids, use_cache=True)
            target_past = prime_out.past_key_values
            tracker.advance(input_ids.shape[1] - 1)
            evict_streaming_per_token(target_past, n_sink, n_window, input_ids.shape[1] - 1)
        else:
            target_past = None

        while tokens_generated < max_new_tokens:
            round_start = time.perf_counter()

            draft_tokens = []
            draft_input = input_ids if draft_past is None else input_ids[:, -1:]

            for _ in range(k):
                out = draft_model(input_ids=draft_input, past_key_values=draft_past, use_cache=True)
                draft_past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                draft_input = next_token
                input_ids = torch.cat([input_ids, next_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)

            verify_input = input_ids[:, -(k + 1):]
            verify_position_ids = tracker.position_ids(k + 1, device)

            out = target_model(input_ids=verify_input, position_ids=verify_position_ids, past_key_values=target_past, use_cache=True)
            target_logits = out.logits
            target_tokens = target_logits.argmax(dim=-1)

            num_accepted = 0
            for i in range(k):
                if target_tokens[0, i].item() == draft_tokens_tensor[0, i].item():
                    num_accepted += 1
                else:
                    break

            accepted_lengths.append(num_accepted)

            if num_accepted < k:
                input_ids = input_ids[:, : input_ids.shape[1] - k + num_accepted]
                correction_token = target_tokens[0, num_accepted].unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                tokens_generated += num_accepted + 1
                draft_past = None
            else:
                bonus_token = target_tokens[0, k].unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, bonus_token], dim=1)
                tokens_generated += k + 1

            new_target_past = out.past_key_values
            valid_length = new_target_past.get_seq_length() - (k - num_accepted)
            new_target_past.crop(valid_length)
            real_tokens_this_round = num_accepted + 1
            tracker.advance(real_tokens_this_round)
            evict_streaming_per_token(new_target_past, n_sink, n_window, real_tokens_this_round)
            target_past = new_target_past

            step_times.append(time.perf_counter() - round_start)

            eos_positions = (input_ids[0, prompt_len:] == tokenizer.eos_token_id).nonzero()
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                input_ids = input_ids[:, : prompt_len + first_eos + 1]
                tokens_generated = first_eos + 1
                break

            if tokens_generated >= max_new_tokens:
                overshoot = tokens_generated - max_new_tokens
                if overshoot > 0:
                    input_ids = input_ids[:, :-overshoot]
                tokens_generated = max_new_tokens
                break

    total_time = sum(step_times)
    avg_acceptance = sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0

    return {
        "generated_text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "token_ids": input_ids[0].tolist(),
        "tokens_generated": tokens_generated,
        "total_time_s": total_time,
        "tokens_per_sec": tokens_generated / total_time if total_time > 0 else 0.0,
        "avg_acceptance_length": avg_acceptance,
        "acceptance_rate": avg_acceptance / k,
        "num_rounds": len(accepted_lengths),
    }
