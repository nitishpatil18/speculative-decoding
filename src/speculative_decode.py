"""
speculative_decode.py - core speculative decoding algorithm.
draft model proposes k tokens, target model verifies all k in one forward pass.
lossless: output is provably identical to running target alone (greedy).
"""

import time
import torch


def speculative_decode(draft_model, target_model, tokenizer, prompt, max_new_tokens, k, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    draft_past = None
    prompt_len = input_ids.shape[1]
    tokens_generated = 0
    accepted_lengths = []
    step_times = []
    draft_time_total = 0.0
    verify_time_total = 0.0

    with torch.no_grad():
        if input_ids.shape[1] > 1:
            prime_out = target_model(input_ids=input_ids[:, :-1], use_cache=True)
            target_past = prime_out.past_key_values
        else:
            target_past = None

        while tokens_generated < max_new_tokens:
            round_start = time.perf_counter()

            torch.mps.synchronize()
            draft_phase_start = time.perf_counter()
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
            torch.mps.synchronize()
            draft_time_total += time.perf_counter() - draft_phase_start

            verify_phase_start = time.perf_counter()
            verify_input = input_ids[:, -(k + 1):]

            out = target_model(input_ids=verify_input, past_key_values=target_past, use_cache=True)
            torch.mps.synchronize()
            verify_time_total += time.perf_counter() - verify_phase_start
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
            target_past = new_target_past

            torch.mps.synchronize()
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
        "tokens_generated": tokens_generated,
        "total_time_s": total_time,
        "tokens_per_sec": tokens_generated / total_time if total_time > 0 else 0.0,
        "avg_acceptance_length": avg_acceptance,
        "acceptance_rate": avg_acceptance / k,
        "num_rounds": len(accepted_lengths),
        "draft_time_total_s": draft_time_total,
        "verify_time_total_s": verify_time_total,
        "draft_time_pct": (draft_time_total / total_time * 100) if total_time > 0 else 0.0,
        "verify_time_pct": (verify_time_total / total_time * 100) if total_time > 0 else 0.0,
    }
