"""
speculative_streaming.py - speculative decoding with a bounded (StreamingLLM)
kv-cache on the target model.

SCOPE NOTE: "correctness" here means identical output to run_streaming_baseline()
- the target model running standalone under the SAME eviction policy - not
identical to the unbounded baseline used elsewhere in this project. Eviction
changes the model's effective context regardless of speculative decoding;
that's a separate, already-quantified cost (see evaluate_quality.py, +15.2%
perplexity post-eviction). This file only has to prove speculative decoding
doesn't add further divergence ON TOP of that.

Draft model's cache is NOT evicted - it resets to None on every rejection
anyway (see speculative_decode.py), so its cache never grows large enough
for eviction to matter, and its own position_ids can keep using the default
cache-length-based computation safely.
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
            evict_streaming(target_past, n_sink=n_sink, n_window=n_window)
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
            new_target_past.crop(valid_length)  # correctness: drop rejected-token cache entries
            tracker.advance(num_accepted + 1)
            evict_streaming(new_target_past, n_sink=n_sink, n_window=n_window)  # THEN bound memory
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


def run_streaming_baseline_with_logit_capture(model, tokenizer, prompt, max_new_tokens, device, n_sink, n_window, capture_at_step):
    """
    Identical to run_streaming_baseline, but captures the full logit vector
    at one specific step, for debugging a specific divergence point using
    the ACTUAL incremental state at that point - not a reconstruction.
    """
    cache = DynamicCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    tracker = StreamingPositionTracker(start=0)
    captured_logits = None

    with torch.no_grad():
        position_ids = tracker.position_ids(generated.shape[1], device)
        out = model(input_ids=generated, position_ids=position_ids, past_key_values=cache, use_cache=True)
        tracker.advance(generated.shape[1])
        cache = out.past_key_values
        evict_streaming(cache, n_sink=n_sink, n_window=n_window)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        for step in range(max_new_tokens - 1):
            position_ids = tracker.position_ids(1, device)
            out = model(input_ids=next_token, position_ids=position_ids, past_key_values=cache, use_cache=True)
            tracker.advance(1)
            cache = out.past_key_values
            evict_streaming(cache, n_sink=n_sink, n_window=n_window)

            if step == capture_at_step:
                captured_logits = out.logits[0, -1, :].clone()

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated[0].tolist(), captured_logits


def speculative_decode_streaming_debug(draft_model, target_model, tokenizer, prompt, max_new_tokens, k, device, n_sink, n_window, stop_at_tokens_generated):
    """
    Identical to speculative_decode_streaming, but prints cache occupancy and
    tracker position every round, and stops early once tokens_generated
    passes stop_at_tokens_generated - for inspecting exactly what state the
    system is in right as a known divergence point is approached.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    draft_past = None
    prompt_len = input_ids.shape[1]
    tokens_generated = 0
    tracker = StreamingPositionTracker(start=0)

    with torch.no_grad():
        if input_ids.shape[1] > 1:
            prime_position_ids = tracker.position_ids(input_ids.shape[1] - 1, device)
            prime_out = target_model(input_ids=input_ids[:, :-1], position_ids=prime_position_ids, use_cache=True)
            target_past = prime_out.past_key_values
            tracker.advance(input_ids.shape[1] - 1)
            evict_streaming(target_past, n_sink=n_sink, n_window=n_window)
        else:
            target_past = None

        round_num = 0
        while tokens_generated < max_new_tokens:
            round_num += 1
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

            print(f"round {round_num}: tokens_generated={tokens_generated}, cache_len_before={target_past.get_seq_length() if target_past else 0}, tracker_pos_before={tracker.true_position}, verify_position_ids={verify_position_ids.tolist()}")

            out = target_model(input_ids=verify_input, position_ids=verify_position_ids, past_key_values=target_past, use_cache=True)
            target_tokens = out.logits.argmax(dim=-1)

            num_accepted = 0
            for i in range(k):
                if target_tokens[0, i].item() == draft_tokens_tensor[0, i].item():
                    num_accepted += 1
                else:
                    break

            if num_accepted < k:
                input_ids = input_ids[:, : input_ids.shape[1] - k + num_accepted]
                correction_token = target_tokens[0, num_accepted].unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                tokens_generated += num_accepted + 1
                draft_past = None
                print(f"  REJECTION at round {round_num}: num_accepted={num_accepted}/{k}")
            else:
                bonus_token = target_tokens[0, k].unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, bonus_token], dim=1)
                tokens_generated += k + 1

            new_target_past = out.past_key_values
            cache_len_before_crop = new_target_past.get_seq_length()
            valid_length = new_target_past.get_seq_length() - (k - num_accepted)
            new_target_past.crop(valid_length)
            cache_len_after_crop = new_target_past.get_seq_length()
            tracker.advance(num_accepted + 1)
            evict_streaming(new_target_past, n_sink=n_sink, n_window=n_window)
            cache_len_after_evict = new_target_past.get_seq_length()
            target_past = new_target_past

            print(f"  cache: before_crop={cache_len_before_crop}, after_crop={cache_len_after_crop}, after_evict={cache_len_after_evict}, tracker_pos_after={tracker.true_position}")

            if tokens_generated >= stop_at_tokens_generated:
                print(f"\nstopping at tokens_generated={tokens_generated} (requested stop point)")
                break

    return input_ids[0].tolist()
