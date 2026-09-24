"""
baseline.py - vanilla autoregressive decoding loop.
control run: every speculative/adaptive result is measured against this.
"""

import json
import time
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
RESULTS_PATH = Path(__file__).parent.parent / "results" / "runs.jsonl"

TEST_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of the French Revolution?",
    "Describe how a transformer's attention mechanism works.",
    "Give three tips for writing clean Python code.",
]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_model(model_name, dtype, device):
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype_map[dtype])
    model.to(device)
    model.eval()
    return model


def run_baseline(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    step_times = []

    with torch.no_grad():
        torch.mps.synchronize()
        start = time.perf_counter()
        out = model(input_ids=generated, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        torch.mps.synchronize()
        step_times.append(time.perf_counter() - start)

        for _ in range(max_new_tokens - 1):
            torch.mps.synchronize()
            start = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            torch.mps.synchronize()
            step_times.append(time.perf_counter() - start)

            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = sum(step_times)
    tokens_generated = len(step_times)
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0.0

    return {
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "tokens_generated": tokens_generated,
        "total_time_s": total_time,
        "tokens_per_sec": tokens_per_sec,
        "avg_latency_per_token_ms": (total_time / tokens_generated) * 1000 if tokens_generated else 0.0,
    }


def main():
    config = load_config()
    device = config["device"]
    dtype = config["dtype"]
    max_new_tokens = config["max_new_tokens"]

    print(f"loading target model: {config['target_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    model = load_model(config["target_model"], dtype, device)

    RESULTS_PATH.parent.mkdir(exist_ok=True)

    print("warming up mps backend...")
    run_baseline(model, tokenizer, "Warmup.", max_new_tokens=10, device=device)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] prompt: {prompt[:50]}...")
        result = run_baseline(model, tokenizer, prompt, max_new_tokens, device)
        print(f"  tokens/sec: {result['tokens_per_sec']:.2f}, avg latency: {result['avg_latency_per_token_ms']:.2f}ms/token")

        log_entry = {
            "run_type": "baseline",
            "model": config["target_model"],
            "prompt": prompt,
            **result,
        }
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nresults appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
