"""Provenance test verifying every metric and checksum cited in README.md matches results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_numbers_trace_to_results_json():
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

    # 1. Distillation acceptance rates and rejection counts in README must match distillation_manifest.json
    easy_rate = f"{distill['acceptance_by_difficulty']['easy']['acceptance_rate'] * 100:.2f}%"
    med_rate = f"{distill['acceptance_by_difficulty']['medium']['acceptance_rate'] * 100:.2f}%"
    hard_rate = f"{distill['acceptance_by_difficulty']['hard']['acceptance_rate'] * 100:.2f}%"
    assert easy_rate in readme, f"Easy acceptance rate {easy_rate} missing from README"
    assert med_rate in readme, f"Medium acceptance rate {med_rate} missing from README"
    assert hard_rate in readme, f"Hard acceptance rate {hard_rate} missing from README"

    binder_rej = str(distill["rejection_breakdown"]["execution_error: BinderException"])
    guard_rej = str(distill["rejection_breakdown"]["guardrail_rejected"])
    neg_rej = str(distill["rejection_breakdown"]["negative_query_not_refused"])
    assert binder_rej in readme
    assert guard_rej in readme
    assert neg_rej in readme

    # 2. All 4 scaling split SHA-256 hashes must match
    for split_key, split_meta in distill["scaling_splits"].items():
        assert split_meta["sha256"] in readme, f"Missing SHA-256 for {split_key} in README"

    # 3. Serving economics numbers in README must match serving_cost_analysis.json
    econ = serving["economics"]
    assert f"{econ['t4_saturated_usd_per_1k_queries']:.4f}" in readme
    assert f"{econ['saturated_cost_advantage_x']}×" in readme
    assert f"{econ['break_even_queries_per_day']:,}" in readme
    assert f"{econ['break_even_queries_per_month']:,}" in readme

    # 4. SFT dry-run prompt/completion character lengths must match
    s1 = sft_dry["sft-qwen15b-lora"]
    assert f"{s1['sample_prompt_chars']:,}" in readme
    assert str(s1["sample_completion_chars"]) in readme

    # 5. General holdout SHA prefix & honest mode disclosure must be present
    assert smoke["general_holdout_sha256"][:12] in readme
    assert "is_model_evidence: false" in readme
