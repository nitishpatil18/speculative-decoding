"""
kv_cache.py - StreamingLLM-style eviction (Xiao et al. 2023, "attention sinks"),
applied externally to a standard DynamicCache after each forward pass.

Keeps the first n_sink tokens plus a sliding window of the n_window most
recent tokens. Evicts the middle. Bounds memory to O(n_sink + n_window)
regardless of total sequence length.

POSITION TRACKING - the actual bug history here matters, recorded for future
reference: position_ids must NOT be derived from cache.get_seq_length() once
eviction is active. That length is CAPPED (e.g. 36) and stops changing once
the cache is full, so every token generated after that point gets assigned
the same frozen position (confirmed empirically: position stuck at 36 from
step 20 through step 39, i.e. real distinct tokens becoming indistinguishable
to RoPE, producing zero relative distance and driving the repetition collapse
seen in earlier tests). The correct position for a NEW query token is the
TRUE elapsed step count, tracked externally, independent of physical cache
size - not read from the cache at all.

This still does NOT fix rotation angles already baked into surviving window
tokens at their original (pre-renumbering) positions - a real inconsistency
between old and new tokens' relative distances remains. Untested whether
that residual error alone is enough to break coherence; the frozen-position
bug above was severe enough on its own to fully explain the collapse seen so
far, so it must be fixed and retested before that separate question is worth
investigating.
"""

import torch


def evict_streaming(cache, n_sink=4, n_window=256):
    """
    Mutates `cache` in place: for each layer, keeps the first n_sink tokens
    and the last n_window tokens, drops everything in between.
    Safe to call every step - no-ops if the layer is still under the cap.
    """
    max_len = n_sink + n_window

    for layer in cache.layers:
        seq_len = layer.get_seq_length()
        if seq_len <= max_len:
            continue

        sink_k = layer.keys[..., :n_sink, :]
        sink_v = layer.values[..., :n_sink, :]
        window_k = layer.keys[..., -n_window:, :]
        window_v = layer.values[..., -n_window:, :]

        layer.keys = torch.cat([sink_k, window_k], dim=-2)
        layer.values = torch.cat([sink_v, window_v], dim=-2)


class StreamingPositionTracker:
    """
    Tracks the TRUE elapsed generation position, independent of the cache's
    physical (capped) size. Call .advance(n) after each forward pass with
    the number of new tokens just processed, and .position_ids(n, device)
    before the NEXT forward pass to get correctly-numbered positions for the
    next n tokens.
    """

    def __init__(self, start=0):
        self.true_position = start

    def position_ids(self, num_new_tokens, device):
        positions = torch.arange(
            self.true_position, self.true_position + num_new_tokens, device=device
        )
        return positions.unsqueeze(0)

    def advance(self, num_new_tokens):
        self.true_position += num_new_tokens
