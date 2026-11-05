"""Pydantic-validated configuration schemas and environment loader for schema-tuned-sql."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Fallback YAML loader for standard flat/nested config files when PyYAML is absent."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    def _parse_scalar(val_str: str) -> Any:
        val_str = val_str.strip()
        if not val_str:
            return None
        if val_str.lower() == "true":
            return True
        if val_str.lower() == "false":
            return False
        if val_str.lower() in ("null", "none", "~"):
            return None
        if (val_str.startswith("[") and val_str.endswith("]")) or (
            val_str.startswith("{") and val_str.endswith("}")
        ):
            try:
                return ast.literal_eval(val_str)
            except Exception:
                return val_str
        if (val_str.startswith('"') and val_str.endswith('"')) or (
            val_str.startswith("'") and val_str.endswith("'")
        ):
            return val_str[1:-1]
        try:
            if "." in val_str or "e" in val_str.lower():
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str

    root: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            i += 1
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if indent == 0 and ":" in stripped:
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v:
                root[k] = _parse_scalar(v)
                i += 1
            else:
                # Inspect indented block
                i += 1
                sub_lines: list[str] = []
                while i < len(lines):
                    nxt = lines[i].split("#", 1)[0].rstrip()
                    if not nxt.strip():
                        i += 1
                        continue
                    nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                    if nxt_indent <= indent:
                        break
                    sub_lines.append(nxt.strip())
                    i += 1
                if sub_lines and sub_lines[0].startswith("- "):
                    items_list: list[Any] = []
                    cur_dict: dict[str, Any] | None = None
                    for sl in sub_lines:
                        if sl.startswith("- "):
                            after = sl[2:].strip()
                            if after.startswith("["):
                                items_list.append(_parse_scalar(after))
                                cur_dict = None
                            elif ":" in after:
                                if cur_dict is not None:
                                    items_list.append(cur_dict)
                                ck, cv = after.split(":", 1)
                                cur_dict = {ck.strip(): _parse_scalar(cv)}
                            else:
                                items_list.append(_parse_scalar(after))
                        elif ":" in sl and cur_dict is not None:
                            ck, cv = sl.split(":", 1)
                            cur_dict[ck.strip()] = _parse_scalar(cv)
                    if cur_dict is not None:
                        items_list.append(cur_dict)
                    root[k] = items_list
                else:
                    sub_dict: dict[str, Any] = {}
                    for sl in sub_lines:
                        if ":" in sl:
                            sk, sv = sl.split(":", 1)
                            sub_dict[sk.strip()] = _parse_scalar(sv)
                    root[k] = sub_dict
        else:
            i += 1
    return root


def load_dotenv(env_path: str | Path = ".env") -> dict[str, str]:
    """Loads environment variables from a local .env file without external dependencies."""
    candidates = [
        Path(env_path),
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    target: Path | None = None
    for cand in candidates:
        if cand.is_file():
            target = cand
            break
    if target is None:
        return {}

    loaded: dict[str, str] = {}
    try:
        for raw_line in target.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and val and key not in os.environ:
                os.environ[key] = val
                loaded[key] = val
    except OSError:
        pass
    return loaded


load_dotenv()


class DataGenConfig(BaseModel):
    """Configuration for execution-verified teacher distillation."""

    run_name: str = "metricpulse-distill-v1"
    primary_teacher: str = "deepseek/deepseek-chat"
    secondary_teacher: str = "qwen/qwen-2.5-coder-32b-instruct"
    require_dual_agreement: bool = False
    target_verified_pairs: int = Field(default=2000, gt=0)
    scaling_splits: list[int] = Field(default_factory=lambda: [250, 500, 1000, 2000])
    questions_per_call: int = Field(default=8, ge=1, le=50)
    difficulty_weights: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.30, "medium": 0.40, "hard": 0.30}
    )
    negative_ratio: float = Field(default=0.08, ge=0.0, le=0.5)
    max_usd: float = Field(default=15.0, gt=0.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1800, gt=0)
    execution_timeout_sec: float = Field(default=5.0, gt=0.0)
    contamination_sim_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    seed: int = 42
    cache_dir: str = ".cache/llm_responses"
    budget_ledger_path: str = ".cache/budget_ledger.json"
    output_dir: str = "data/train"

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataGenConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = _parse_simple_yaml(p.read_text(encoding="utf-8"))
        return cls(**data)


class LoRAConfig(BaseModel):
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class QuantizationConfig(BaseModel):
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["float16"] = "float16"


class SFTConfig(BaseModel):
    """T4-safe LoRA / QLoRA SFT configuration."""

    run_name: str = "sft-qwen15b-lora"
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    precision: Literal["fp16"] = "fp16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    max_seq_len: int = Field(default=1536, gt=0, le=4096)
    per_device_train_batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = True
    optimizer: Literal["paged_adamw_8bit", "adamw_torch"] = "paged_adamw_8bit"
    learning_rate: float = Field(default=2.0e-4, gt=0.0)
    lr_scheduler: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    epochs: int = Field(default=2, gt=0)
    loss_on: Literal["completion_only", "full_sequence"] = "completion_only"
    include_schema_in_prompt: bool = True
    use_execution_filtered_data: bool = True
    train_subset_size: int = Field(default=2000, gt=0)
    eval_every_steps: int = Field(default=100, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [13, 42, 7])
    train_data_path: str = "data/train/train_2000.jsonl"
    dev_data_path: str = "data/dev.jsonl"
    output_dir: str = ".artifacts/checkpoints"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "schema-tuned-sql-sft"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = _parse_simple_yaml(p.read_text(encoding="utf-8"))
        return cls(**data)


class SystemSpec(BaseModel):
    id: str
    name: str
    kind: Literal["zero_shot", "few_shot", "schema_rag", "api_teacher", "sft_student", "sft_rag"]
    model: str
    adapter_path: str | None = None
    few_shot_k: int = 0
    rag_top_tables: int = 4
    rag_top_defs: int = 3
    rag_top_examples: int = 3


class EvalConfig(BaseModel):
    """Evaluation suite configuration."""

    run_name: str = "metricpulse-benchmark-v1"
    test_path: str = "data/test_gold.jsonl"
    dev_path: str = "data/dev.jsonl"
    general_holdout_path: str = "data/general_sql_holdout.jsonl"
    warehouse_seed: int = 42
    bootstrap_resamples: int = Field(default=2000, gt=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    float_tolerance: float = Field(default=1e-4, gt=0.0)
    max_usd: float = Field(default=15.0, gt=0.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    cache_dir: str = ".cache/llm_responses"
    budget_ledger_path: str = ".cache/budget_ledger.json"
    results_dir: str = "results"
    systems: list[SystemSpec] = Field(default_factory=list)
    mcnemar_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [("S2", "T1"), ("S2", "B2"), ("S1", "S2")]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = _parse_simple_yaml(p.read_text(encoding="utf-8"))
        return cls(**data)


class ServeConfig(BaseModel):
    """T4 serving and economic break-even benchmark configuration."""

    run_name: str = "t4-serving-bench"
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    adapter_path: str | None = ".artifacts/checkpoints/sft-qwen3b-qlora"
    backend: Literal["vllm_fp16", "hf_fp16", "hf_bnb_4bit", "llamacpp_q4_k_m"] = "vllm_fp16"
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32])
    num_queries: int = Field(default=64, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)
    t4_hourly_usd: float = Field(default=0.35, gt=0.0)
    comparison_api_model: str = "deepseek/deepseek-chat"
    avg_prompt_tokens: int = 950
    avg_completion_tokens: int = 110
    results_path: str = "results/serving_cost_analysis.json"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServeConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = _parse_simple_yaml(p.read_text(encoding="utf-8"))
        return cls(**data)
