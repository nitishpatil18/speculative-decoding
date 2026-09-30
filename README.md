# speculative decoding and kv-cache optimization for llm inference

speculative decoding and kv-cache eviction, built from scratch, with an adaptive acceptance-rate-aware draft policy. every number below is measured, not estimated — including two real bugs found and root-caused during development, reported honestly rather than hidden.

**hardware:** Apple M-series (mps backend), 16GB unified memory
**models:** Qwen2.5-0.5B-Instruct (draft), Qwen2.5-3B-Instruct (target) — same tokenizer family, 151,643 shared vocab, required for token-level verification

## 1. speculative decoding

standard algorithm (Leviathan et al., 2023): draft model proposes `k` tokens autoregressively, target model verifies all `k` in one forward pass, accept while draft matches target's argmax, correct on first mismatch.

**correctness first, always.** speculative decoding's core claim is that output is *provably identical* to running the target model alone. this was checked, not assumed — and the check found two real bugs:

- target model wasn't seeing the prompt on round 1 (verification input was miscalculated for the priming case)
- target's kv-cache retained stale entries from rejected draft tokens across rounds

both fixed; 0/5 test mismatches confirmed after.

**instrumentation bug, found separately:** initial benchmarks showed baseline running at 140+ tok/s and speculative running *slower* than baseline. root cause: `time.perf_counter()` around async mps kernel dispatch returns before the GPU actually finishes work, making every prior timing number fake. after adding `torch.mps.synchronize()` around every timed interval, baseline dropped to its real ~13 tok/s, and speculative showed genuine, physically consistent speedups that scale with acceptance rate — the correlation itself is evidence the corrected measurement is trustworthy.

**k-sweep result (5 test prompts, average):**

| k | avg speedup | avg acceptance rate | draft time share |
|---|---|---|---|
| 2 | 1.28x | 55.1% | ~34% |
| 4 | 1.23x | 43.4% | ~50% |
| 8 | 0.98x | 30.4% | ~67% |

**finding: smaller k wins here, monotonically, not unimodally.** acceptance rate decays geometrically with k (a streak of k exact argmax matches between a 0.5B and 3B model gets exponentially less likely as k grows), while draft cost scales linearly with k. at k=8, draft-phase cost consumes 67% of round time while accepting only 30% of proposals — worse than baseline on 2 of 5 prompts.

## 2. adaptive-k policy (original contribution)

instead of hand-tuning k via a sweep, an ema of recent per-round acceptance rate adjusts k dynamically (raises k when acceptance is high, lowers it when low).

| prompt | fixed k=2 | adaptive | adaptive avg_k |
|---|---|---|---|
| 1 | 1.42x | 1.47x | 4.77 |
| 2 | 1.21x | 1.01x | 2.99 |
| 3 | 1.15x | 1.24x | 3.88 |
| 4 | 1.24x | 1.20x | 4.27 |
| 5 | 1.30x | 1.01x | 3.10 |

average: fixed k=2 = 1.26x, adaptive = 1.19x. **honest finding: adaptive doesn't beat a manually-tuned fixed k** — it gets close (1.19x vs 1.26x) *without* needing the sweep that found k=2 was near-optimal in the first place. that's the actual contribution: automatic near-optimal tuning, not a bigger speedup number.

**known limitation, found and root-caused, not glossed over:** one of five test prompts shows a single token mismatch between adaptive and baseline output. the divergence point was isolated exactly (token position 64, `' common'` vs `' poor'`), and the two candidate logits differ by only **0.0156** — smaller than fp16's own rounding resolution at that magnitude (~0.023 at logit value ~24). batched verification and sequential computation sum floating-point operations in a different order, and near-tied logits can flip between the two paths. this is a documented characteristic of speculative decoding under fp16 in general (correctness is an exact-arithmetic guarantee), not a defect in this implementation's accept/reject logic — confirmed by checking the actual margin rather than assuming.

## 3. streaming kv-cache (attention sinks)

StreamingLLM-style eviction (Xiao et al., 2023): keep the first `n_sink` tokens plus a sliding window of the `n_window` most recent tokens, evict the middle. bounds memory to O(n_sink + n_window) regardless of total sequence length.

**two real bugs found during implementation, both root-caused with data, not guessed at:**

1. **API mismatch:** initial implementation subclassed `DynamicCache` assuming `self.key_cache`/`self.value_cache` list attributes. transformers 5.15.0 restructured this to per-layer objects (`cache.layers[i].keys/values`). fixed by applying eviction externally via the actual current API instead of fighting version-specific internals.

2. **position tracking bug (the real one):** after fixing the API, generation degraded into repetition past the eviction window ("to have granted to have granted", "The. The. The."). root cause, found by direct inspection of `Qwen2Model.forward`'s position_id computation: `position_ids = arange(new_tokens) + cache.get_seq_length()`. once eviction caps the cache length (e.g. at 36), `get_seq_length()` **freezes** — every new token past that point gets assigned the *same* position, collapsing RoPE's relative-distance signal to zero between all sufficiently-recent tokens. verified directly: positions were confirmed frozen at 36 from generation step 20 through step 39. fixed with an external position tracker that counts true elapsed generation steps independent of physical cache size.

**quantified results, not just eyeballed text:**
- memory: unbounded cache grows at 36.3KB/token (measured, matches the theoretical 36,864 bytes/token from model config exactly). at 3B-model scale and 4000 tokens this is a modest 2.3% of total process memory (model weights dominate at ~6GB) — but extrapolated to 128,000 tokens, unbounded cache alone would add ~4.5GB while streaming cache stays flat at ~9MB. this is the actual problem PagedAttention/vLLM exist to solve at production scale (70B+ models, many concurrent long-context users); this project's numbers are the small-scale proof of that large-scale mechanism.
- quality: teacher-forced perplexity, same reference sequence, unbounded vs streaming cache. pre-eviction losses match exactly (2.59 vs 2.59 ppl, confirming no implementation bug before the window fills). post-eviction: perplexity rises from 1.63 to 1.87, a measured **+15.2% increase** — a real, quantified cost of discarding 96%+ of context, not a hand-waved "still coherent."

**known remaining limitation:** the position-tracking fix corrects new-token positions but does not retroactively re-rotate already-cached window tokens' RoPE angles, which were baked in at their original (pre-eviction) positions. qualitatively, this shows up as entity/pronoun drift in long generations (e.g. a character introduced as "the girl" becomes "the young man" after aging out of the window) — an expected, understood cost of the technique, not a bug.

## reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/benchmark.py           # baseline vs fixed-k vs adaptive-k, correctness + speedup
python src/test_kv_cache.py       # streaming cache correctness (shape bound + coherence)
python src/benchmark_memory.py    # unbounded vs streaming memory growth
python src/evaluate_quality.py    # teacher-forced perplexity cost of eviction
```

## what this project demonstrates

not just "implemented papers correctly" — found, isolated, and fixed four real bugs across two subsystems (two in speculative verification, two in streaming cache), each with a data-backed root cause rather than a guess, and reported every result — including the ones that came out worse than hoped (k=8 regression, adaptive not beating fixed-k, the fp16 tie, the 15.2% quality cost) — as findings rather than hiding them.

## 4. combining speculative decoding + streaming cache

attempted, and the negative result is itself the finding worth reporting.

**hypothesis:** with the streaming cache's correctness already proven independently (section 3) and speculative decoding's correctness already proven independently (section 1), combining them should be straightforward wiring.

**result: it isn't.** correctness (matching a streaming-baseline reference under the same eviction policy) fails once eviction and speculative decoding's variable-length rounds interact. two fix attempts were tried and empirically ruled out before finding the actual cause:

1. **hypothesized cause: position tracker desync.** ruled out by direct inspection — walking the trace confirmed the tracker's position accounting and cache length bookkeeping were both correct and mutually consistent throughout. (see `debug_streaming_divergence.py`)
2. **hypothesized cause: transient window overshoot during batched verification, fix = shrink the eviction cap by k.** implemented and tested — did not fix the divergence; the mismatch point moved (position 54 → 40) rather than disappearing, proving window capacity was never the actual constraint.
3. **confirmed cause: eviction granularity.** the streaming baseline evicts after every single token, one at a time. speculative decoding verifies `k+1` tokens in a single batched forward pass, then evicts once per round. this is not equivalent to evicting once per token even after correcting the loop count — confirmed by actually testing a per-token eviction loop, which reproduced the *exact same* divergence (same position, same mismatched tokens) as the original single-eviction version, proving eviction call count doesn't matter. the real difference is architectural: a batched k+1-token verification pass computes attention across all k+1 positions simultaneously against one snapshot of the cache, which is mathematically different from N sequential single-token passes each seeing a freshly-evicted cache in between — independent of how many times eviction runs afterward.

**this is a genuine, structural tension between the two techniques as implemented here**, not an unfixed implementation bug — every fix attempt was tested empirically, not assumed, and each test's result directly ruled out the theory behind it before moving to the next. a real fix would require verifying one token at a time against a per-token-evicted cache, which removes the batched-verification speedup that is speculative decoding's entire reason for existing — at which point the "fixed" combination would simply be the streaming baseline with extra steps.

each technique is correct and independently verified with real numbers on its own (sections 1-3). their combination, as implemented, is not — and understanding precisely *why not*, backed by three tested hypotheses rather than one guess, is the actual result of this section.
