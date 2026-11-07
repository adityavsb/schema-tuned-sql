"""Provenance test verifying README.md matches results/*.json and never reports dry-run/offline numbers as model or GPU results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_numbers_and_anti_slop_provenance():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    distill = json.loads(
        (REPO_ROOT / "results" / "distillation_manifest.json").read_text(encoding="utf-8")
    )
    serving = json.loads(
        (REPO_ROOT / "results" / "serving_cost_analysis.json").read_text(encoding="utf-8")
    )
    sft_dry = json.loads(
        (REPO_ROOT / "results" / "t4_sft_dryrun_manifest.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (REPO_ROOT / "results" / "offline_smoke_summary.json").read_text(encoding="utf-8")
    )

    # 1. Verify frozen evaluation split SHA-256 hashes in README match actual files on disk
    test_sha = hashlib.sha256((REPO_ROOT / "data" / "test_gold.jsonl").read_bytes()).hexdigest()
    dev_sha = hashlib.sha256((REPO_ROOT / "data" / "dev.jsonl").read_bytes()).hexdigest()
    holdout_sha = hashlib.sha256(
        (REPO_ROOT / "data" / "general_sql_holdout.jsonl").read_bytes()
    ).hexdigest()

    assert test_sha[:12] in readme
    assert dev_sha[:12] in readme
    assert holdout_sha[:12] in readme
    assert smoke["general_holdout_sha256"] == holdout_sha

    # 2. Distillation manifest provenance
    if not distill.get("is_live_teacher_distillation"):
        assert distill["status"] == "pending_openrouter_run"
        assert distill["total_verified_pairs"] is None
        for bucket in ("easy", "medium", "hard"):
            assert distill["acceptance_by_difficulty"][bucket]["acceptance_rate"] is None
        assert "status: pending_openrouter_run" in readme
    else:
        assert str(distill["total_verified_pairs"]) in readme
        assert f"{distill['acceptance_by_difficulty']['easy']['acceptance_rate'] * 100:.2f}%" in readme
        assert f"{distill['acceptance_by_difficulty']['medium']['acceptance_rate'] * 100:.2f}%" in readme
        assert f"{distill['acceptance_by_difficulty']['hard']['acceptance_rate'] * 100:.2f}%" in readme
        for split_meta in distill["scaling_splits"].values():
            assert split_meta["sha256"] in readme

    # 3. If Tesla T4 GPU serving benchmark has not run yet, GPU telemetry must be None and README must show N/A
    if not serving.get("is_hardware_telemetry"):
        assert serving["status"] == "pending_t4_gpu_run"
        econ = serving["economics"]
        assert econ["t4_throughput_qps"] is None
        assert econ["t4_saturated_usd_per_1k_queries"] is None
        assert econ["break_even_queries_per_day"] is None
        assert "status: pending_t4_gpu_run" in readme

    # 4. SFT config matrix in README matches configs
    for cfg_key, meta in sft_dry.items():
        assert cfg_key in readme
        assert meta["base_model"] in readme
        if not meta.get("is_hardware_telemetry"):
            assert meta["status"] == "pending_t4_gpu_run"
        else:
            assert meta["status"] == "completed_t4_gpu_run"
            if meta.get("final_train_loss") is not None:
                assert f'{meta["final_train_loss"]:.4f}' in readme

    t4_eval_file = REPO_ROOT / "results" / "t4_cuda_eval_summary.json"
    if t4_eval_file.is_file():
        t4_data = json.loads(t4_eval_file.read_text(encoding="utf-8"))
        if t4_data.get("is_model_evidence"):
            for sid in ("S1", "S2"):
                acc = t4_data["systems"][sid]["execution_accuracy"]
                assert f"{acc * 100:.2f}%" in readme
            assert "t4_cuda_eval_summary.json" in readme

    # 5. Honest mode disclosure and live API evaluation numbers
    or_eval_file = REPO_ROOT / "results" / "openrouter_eval_summary.json"
    if or_eval_file.is_file():
        or_data = json.loads(or_eval_file.read_text(encoding="utf-8"))
        if or_data.get("is_model_evidence"):
            t1 = or_data["systems"]["T1"]
            t2 = or_data["systems"]["T2"]
            assert f"{t1['execution_accuracy'] * 100:.2f}%" in readme
            assert f"{t2['execution_accuracy'] * 100:.2f}%" in readme
            assert f"{t1['valid_sql_rate'] * 100:.2f}%" in readme
            assert f"{t2['valid_sql_rate'] * 100:.2f}%" in readme
            assert "openrouter_eval_summary.json" in readme

    assert "is_hardware_telemetry: false" in readme
