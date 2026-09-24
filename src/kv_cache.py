"""
kv_cache.py - StreamingLLM-style eviction (Xiao et al. 2023, "attention sinks"),
applied externally to a standard DynamicCache after each forward pass.

Keeps the first n_sink tokens (models allocate disproportionate attention to
early tokens regardless of content) plus a sliding window of the n_window
most recent tokens. Evicts the middle. Bounds memory to O(n_sink + n_window)
regardless of total sequence length, instead of growing without bound.

Applied post-hoc via each layer's .keys/.values tensors, matching the actual
DynamicLayer interface in this transformers version - not via subclassing,
since DynamicCache's internals (layer storage structure, attribute names)
are not a stable API across versions and already changed once during this
project (self.key_cache/value_cache no longer exist as of transformers
5.15.0, replaced by self.layers[i].keys/values).
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
