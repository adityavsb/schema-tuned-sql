"""Unit tests for T4 hardware constraint checks, completion-only loss masking, Schema-RAG, and serving economics."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.rag import SchemaDefinitionRetriever
from src.serve.bench import compute_serving_economics
from src.train.collator import format_sft_example, mask_completion_only_labels
from src.train.prefs import build_execution_preference_pairs
from src.train.sft import (
    T4HardwareConstraintError,
    run_sft,
    t4_trainer_precision_flags,
    validate_t4_constraints,
)
from src.utils.config import SFTConfig
from src.utils.tracking import _inside_git_work_tree

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_t4_hardware_constraints_reject_bf16_and_flash_attn():
    cfg = SFTConfig.from_yaml(REPO_ROOT / "configs" / "sft_qwen15b_lora.yaml")
    validate_t4_constraints(cfg)

    # Corrupt precision to bf16 -> must fail on T4
    cfg_bad_prec = cfg.model_copy(deep=True)
    cfg_bad_prec.precision = "bf16"  # type: ignore[assignment]
    with pytest.raises(T4HardwareConstraintError, match="does not support native bf16"):
        validate_t4_constraints(cfg_bad_prec)

    # Corrupt attention to flash_attention_2 -> must fail on T4
    cfg_bad_attn = cfg.model_copy(deep=True)
    cfg_bad_attn.attn_implementation = "flash_attention_2"  # type: ignore[assignment]
    with pytest.raises(T4HardwareConstraintError, match="FlashAttention-2 requires Ampere"):
        validate_t4_constraints(cfg_bad_attn)

    # Excessive sequence length beyond safe T4 VRAM budget -> must fail on T4
    cfg_bad_len = cfg.model_copy(deep=True)
    cfg_bad_len.max_seq_len = 8192
    with pytest.raises(T4HardwareConstraintError, match="exceeds safe 16 GB T4 VRAM budget"):
        validate_t4_constraints(cfg_bad_len)


def test_completion_only_label_masking():
    retriever = SchemaDefinitionRetriever()
    sample = format_sft_example(
        {
            "question": "Count active accounts",
            "sql": "SELECT COUNT(*) FROM accounts WHERE status = 'active'",
        },
        retriever=retriever,
        include_schema=True,
    )
    assert "<|im_start|>assistant\n" in sample["prompt"]
    assert "SELECT COUNT(*)" in sample["completion"]

    input_ids = [1, 2, 3, 42, 43, 101, 102, 103]
    masked = mask_completion_only_labels(input_ids, response_template_ids=[42, 43])
    assert masked == [-100, -100, -100, -100, -100, 101, 102, 103]

    try:
        import torch
        from src.train.collator import CompletionOnlyDataCollator

        collator = CompletionOnlyDataCollator()
        batch = collator(
            [
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 42]},
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 99]},
            ]
        )
        assert batch["labels"][0].tolist() == [-100, -100, 42]
        assert batch["labels"][1].tolist() == [-100, 99, -100]
    except ImportError:
        pass


def test_sft_dry_run_and_dpo_preference_builder():
    cfg = SFTConfig.from_yaml(REPO_ROOT / "configs" / "sft_qwen3b_qlora.yaml")
    cfg.train_data_path = str(REPO_ROOT / "data" / "dev.jsonl")
    cfg.train_subset_size = 50
    res = run_sft(cfg, dry_run=True)
    assert res["status"] == "pending_t4_gpu_run"
    assert res["qlora_enabled"] is True
    assert res["num_examples"] == 50

    prefs = build_execution_preference_pairs(
        verified_items=[
            {
                "question": "Count active accounts",
                "sql": "SELECT COUNT(*) FROM accounts WHERE status = 'active'",
                "is_negative_query": False,
            }
        ],
        rejected_candidates=[],
    )
    assert len(prefs) == 1
    assert "WHERE status = 'active'" in prefs[0]["chosen"]
    assert prefs[0]["chosen"] != prefs[0]["rejected"]


def test_schema_definition_retriever_and_serving_economics():
    retriever = SchemaDefinitionRetriever()
    bundle = retriever.retrieve(
        "What is the Weekly Active User (WAU) count?", top_tables=4, top_defs=2, top_examples=3
    )
    table_names = {t["name"] for t in bundle["tables"]}
    assert "events" in table_names
    assert any("Weekly Active User" in d["title"] for d in bundle["definitions"])
    assert len(bundle["examples"]) == 3

    econ = compute_serving_economics(
        api_model="deepseek/deepseek-chat",
        avg_prompt_tokens=950,
        avg_completion_tokens=110,
        t4_hourly_usd=0.35,
        throughput_qps=2.45,
    )
    assert econ["api_usd_per_1k_queries"] == 0.164
    assert econ["t4_saturated_usd_per_1k_queries"] == 0.0397
    assert econ["saturated_cost_advantage_x"] == 4.13
    assert econ["break_even_queries_per_day"] == 51220


def test_qlora_disables_fp16_grad_scaler_on_t4():
    assert t4_trainer_precision_flags(False) == {"fp16": True, "bf16": False}
    assert t4_trainer_precision_flags(True) == {"fp16": False, "bf16": False}


def test_git_metadata_probe_stops_at_kaggle_mount():
    assert _inside_git_work_tree(Path("/kaggle/working")) is False
    assert _inside_git_work_tree(Path("/kaggle/working/schema-tuned-sql/src")) is False
    assert _inside_git_work_tree(REPO_ROOT) is True
    assert _inside_git_work_tree(REPO_ROOT / "notebooks") is True
