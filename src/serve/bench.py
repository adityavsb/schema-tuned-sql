"""T4 serving benchmark (vLLM FP16 / 4-bit quantized) and API break-even cost model."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.pricing import PRICE_TABLE_DATE, estimate_cost_usd
from src.utils.config import ServeConfig


def compute_serving_economics(
    api_model: str,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
    t4_hourly_usd: float,
    throughput_qps: float,
) -> dict[str, Any]:
    """Calculates USD per 1,000 queries for OpenRouter API vs dedicated T4 and break-even query volume."""
    if throughput_qps <= 0.0:
        raise ValueError("throughput_qps must be positive")

    single_api_usd = estimate_cost_usd(api_model, avg_prompt_tokens, avg_completion_tokens)
    api_usd_per_1k = round(single_api_usd * 1000.0, 4)

    queries_per_hour = throughput_qps * 3600.0
    t4_usd_per_1k = round((t4_hourly_usd / queries_per_hour) * 1000.0, 4)

    cost_ratio = round(api_usd_per_1k / max(1e-6, t4_usd_per_1k), 2)
    daily_t4_cost = round(t4_hourly_usd * 24.0, 2)
    monthly_t4_cost = round(daily_t4_cost * 30.0, 2)

    # Break-even volume assuming a dedicated 24/7 single T4 instance vs pay-per-token API
    break_even_per_day = int(math.ceil(daily_t4_cost / max(1e-9, single_api_usd)))
    break_even_per_month = break_even_per_day * 30

    return {
        "price_table_date": PRICE_TABLE_DATE,
        "comparison_api_model": api_model,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "api_usd_per_1k_queries": api_usd_per_1k,
        "t4_hourly_usd_assumption": t4_hourly_usd,
        "t4_throughput_qps": round(throughput_qps, 3),
        "t4_saturated_usd_per_1k_queries": t4_usd_per_1k,
        "saturated_cost_advantage_x": cost_ratio,
        "dedicated_t4_daily_usd": daily_t4_cost,
        "dedicated_t4_monthly_usd": monthly_t4_cost,
        "break_even_queries_per_day": break_even_per_day,
        "break_even_queries_per_month": break_even_per_month,
    }


def run_serving_benchmark(
    config: ServeConfig,
    mode: str = "dry_run",
) -> dict[str, Any]:
    """Runs or models T4 serving latency, batch throughput, VRAM, and cost break-even."""
    if mode == "dry_run":
        # Dry-run without physical CUDA GPU: GPU throughput and break-even fields remain None (N/A)
        return {
            "run_name": config.run_name,
            "mode": "dry_run",
            "status": "pending_t4_gpu_run",
            "is_hardware_telemetry": False,
            "base_model": config.base_model,
            "backend": config.backend,
            "batch_sizes": config.batch_sizes,
            "batches": None,
            "economics": {
                "price_table_date": PRICE_TABLE_DATE,
                "comparison_api_model": config.comparison_api_model,
                "avg_prompt_tokens": None,
                "avg_completion_tokens": None,
                "api_usd_per_1k_queries": None,
                "t4_hourly_usd_assumption": None,
                "t4_throughput_qps": None,
                "t4_saturated_usd_per_1k_queries": None,
                "saturated_cost_advantage_x": None,
                "dedicated_t4_daily_usd": None,
                "dedicated_t4_monthly_usd": None,
                "break_even_queries_per_day": None,
                "break_even_queries_per_month": None,
            },
        }

    # Live CUDA T4 execution path
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for --mode cuda_live serving benchmark.")

    device_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if config.backend == "hf_bnb_4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    sample_prompt = "SELECT account_id, SUM(mrr_amount) FROM subscriptions WHERE status = 'active' GROUP BY account_id"
    batch_results: dict[str, Any] = {}

    for b_size in config.batch_sizes:
        prompts = [sample_prompt] * b_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        latencies_ms: list[float] = []
        total_gen_tokens = 0

        n_iters = max(2, config.num_queries // b_size)
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            new_toks = int(out.shape[1] - inputs["input_ids"].shape[1]) * b_size
            total_gen_tokens += max(1, new_toks)

        total_sec = sum(latencies_ms) / 1000.0
        qps = (n_iters * b_size) / max(1e-6, total_sec)
        tps = total_gen_tokens / max(1e-6, total_sec)
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

        batch_results[f"batch_{b_size}"] = {
            "batch_size": b_size,
            "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "throughput_qps": round(qps, 3),
            "throughput_tok_per_s": round(tps, 1),
            "peak_vram_gb": peak_vram_gb,
        }

    ref_qps = batch_results.get("batch_8", next(iter(batch_results.values())))["throughput_qps"]
    econ = compute_serving_economics(
        api_model=config.comparison_api_model,
        avg_prompt_tokens=config.avg_prompt_tokens,
        avg_completion_tokens=config.avg_completion_tokens,
        t4_hourly_usd=config.t4_hourly_usd,
        throughput_qps=ref_qps,
    )

    return {
        "run_name": config.run_name,
        "mode": "cuda_live",
        "is_hardware_telemetry": True,
        "gpu_name": device_name,
        "compute_capability": f"{cc_major}.{cc_minor}",
        "total_vram_gb": total_vram_gb,
        "base_model": config.base_model,
        "backend": config.backend,
        "batches": batch_results,
        "economics": econ,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T4 serving & cost break-even benchmark")
    parser.add_argument("--config", default="configs/serve.yaml", help="Path to serve.yaml")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "cuda_live"],
        default="dry_run",
        help="Execution mode ('dry_run' on CPU or 'cuda_live' on NVIDIA T4)",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    cfg = ServeConfig.from_yaml(args.config)
    report = run_serving_benchmark(cfg, mode=args.mode)
    out_path = Path(args.out or cfg.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
