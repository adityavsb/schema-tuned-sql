"""T4-safe LoRA / QLoRA Supervised Fine-Tuning (SFT) runner for Qwen2.5-Coder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.rag import SchemaDefinitionRetriever
from src.train.collator import format_sft_example, mask_completion_only_labels
from src.utils.config import SFTConfig
from src.utils.tracking import MLflowTracker


class T4HardwareConstraintError(ValueError):
    """Raised when a training config violates NVIDIA Tesla T4 (sm_75, 16GB) hardware limits."""

    pass


def validate_t4_constraints(config: SFTConfig) -> None:
    """Enforces NVIDIA Tesla T4 (Turing compute capability 7.5) hardware constraints."""
    if config.precision != "fp16":
        raise T4HardwareConstraintError(
            f"Tesla T4 (sm_75) does not support native bf16; got precision='{config.precision}'. Use 'fp16'."
        )
    if config.attn_implementation not in ("sdpa", "eager"):
        raise T4HardwareConstraintError(
            f"FlashAttention-2 requires Ampere (sm_80+); Tesla T4 must use 'sdpa' or 'eager', got '{config.attn_implementation}'."
        )
    if config.max_seq_len > 3584:
        raise T4HardwareConstraintError(
            f"Sequence length {config.max_seq_len} exceeds safe 16 GB T4 VRAM budget (max 3584)."
        )
    if config.quantization.enabled and config.quantization.bnb_4bit_compute_dtype != "float16":
        raise T4HardwareConstraintError(
            "QLoRA on Tesla T4 requires bnb_4bit_compute_dtype='float16'."
        )


def t4_trainer_precision_flags(quantized: bool) -> dict[str, bool]:
    """Mixed-precision flags for Tesla T4.

    Full-precision LoRA keeps the fp16 GradScaler. QLoRA must not: Qwen2.5
    checkpoints are bfloat16, and the fp16 scaler raises
    ``_amp_foreach_non_finite_check_and_unscale_cuda`` for that dtype.
    QLoRA keeps fp32 adapters and float16 compute inside bitsandbytes.
    """
    if quantized:
        return {"fp16": False, "bf16": False}
    return {"fp16": True, "bf16": False}


def apply_t4_trainer_precision(trl_args: Any, quantized: bool) -> None:
    """Write T4 precision flags onto a Trainer config after it is constructed."""
    for key, value in t4_trainer_precision_flags(quantized).items():
        if hasattr(trl_args, key):
            setattr(trl_args, key, value)
    if quantized and isinstance(getattr(trl_args, "mixed_precision", None), str):
        trl_args.mixed_precision = "no"


def prepare_formatted_dataset(
    data_path: str | Path,
    subset_size: int = 5000,
    include_schema: bool = True,
) -> list[dict[str, str]]:
    """Loads JSONL training pairs and applies ChatML formatting."""
    p = Path(data_path)
    if not p.is_file():
        raise FileNotFoundError(f"Training dataset not found: {p}")

    raw_items = [
        json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
    ][:subset_size]

    retriever = SchemaDefinitionRetriever()
    return [
        format_sft_example(item, retriever=retriever, include_schema=include_schema)
        for item in raw_items
    ]


def run_sft(
    config: SFTConfig,
    dry_run: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Validates T4 safety, formats data, and executes (or dry-runs) LoRA/QLoRA SFT."""
    validate_t4_constraints(config)

    if dry_run:
        # Verify completion-only token masking on synthetic token IDs
        dummy_prompt_ids = [10, 11, 12, 99, 100]
        dummy_resp_ids = [201, 202, 203]
        masked = mask_completion_only_labels(
            dummy_prompt_ids + dummy_resp_ids,
            response_template_ids=[99, 100],
        )
        assert masked[:5] == [-100, -100, -100, -100, -100]
        assert masked[5:] == [201, 202, 203]

        data_file = Path(config.train_data_path)
        if data_file.is_file():
            examples = prepare_formatted_dataset(
                data_path=config.train_data_path,
                subset_size=config.train_subset_size,
                include_schema=config.include_schema_in_prompt,
            )
            num_examples: int | None = len(examples)
            sample_prompt_chars: int | None = len(examples[0]["prompt"])
            sample_completion_chars: int | None = len(examples[0]["completion"])
        else:
            num_examples = None
            sample_prompt_chars = None
            sample_completion_chars = None

        return {
            "status": "pending_t4_gpu_run",
            "is_hardware_telemetry": False,
            "run_name": config.run_name,
            "base_model": config.base_model,
            "precision": config.precision,
            "attn_implementation": config.attn_implementation,
            "qlora_enabled": config.quantization.enabled,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.alpha,
            "num_examples": num_examples,
            "sample_prompt_chars": sample_prompt_chars,
            "sample_completion_chars": sample_completion_chars,
        }

    examples = prepare_formatted_dataset(
        data_path=config.train_data_path,
        subset_size=config.train_subset_size,
        include_schema=config.include_schema_in_prompt,
    )

    # Live GPU execution path (Tesla T4)
    import torch
    import inspect
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer
    from trl import SFTConfig as TRLSFTConfig

    # Neutralize incompatible pre-installed torchao in remote cloud GPU environments
    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao

        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    # Robust import for completion-only data collator across TRL versions (0.12+ and 0.20+)
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        try:
            from trl.trainer import DataCollatorForCompletionOnlyLM  # type: ignore[no-redef]
        except ImportError:
            try:
                from trl.trainer.utils import DataCollatorForCompletionOnlyLM  # type: ignore[no-redef]
            except ImportError:
                from src.train.collator import (  # type: ignore[no-redef]
                    CompletionOnlyDataCollator as DataCollatorForCompletionOnlyLM,
                )

    active_seeds = [seed_override] if seed_override is not None else config.seeds
    tracker = MLflowTracker(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_tracking_uri,
    )
    seed_reports: list[dict[str, Any]] = []

    for seed in active_seeds:
        run_label = f"{config.run_name}-seed{seed}"
        with tracker.start_run(run_name=run_label, params=config.model_dump()):
            tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bnb_config = None
            if config.quantization.enabled:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=config.quantization.load_in_4bit,
                    bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                attn_implementation=config.attn_implementation,
                device_map="auto",
                trust_remote_code=True,
            )
            # Qwen2.5 config.json stores bfloat16. PEFT copies that onto LoRA
            # weights unless the loaded config is pinned to float16 first.
            if getattr(model, "config", None) is not None and hasattr(model.config, "torch_dtype"):
                model.config.torch_dtype = torch.float16

            if config.quantization.enabled:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=config.gradient_checkpointing
                )

            peft_cfg = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)

            # Upcast trainable LoRA adapter weights to fp32 to prevent Qwen2.5 fp16 NaN overflow on T4
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)

            hf_ds = Dataset.from_list(examples)
            collator = None
            if config.loss_on == "completion_only":
                collator = DataCollatorForCompletionOnlyLM(
                    response_template="<|im_start|>assistant\n",
                    tokenizer=tokenizer,
                )

            out_dir = Path(config.output_dir) / f"seed_{seed}"

            # Dynamically match parameters supported by TRLSFTConfig.__init__ across versions
            candidate_kwargs: dict[str, Any] = {
                "output_dir": str(out_dir),
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "gradient_checkpointing": config.gradient_checkpointing,
                "optim": config.optimizer,
                "learning_rate": config.learning_rate,
                "lr_scheduler_type": config.lr_scheduler,
                "warmup_ratio": config.warmup_ratio,
                "max_grad_norm": config.max_grad_norm,
                "num_train_epochs": config.epochs,
                **t4_trainer_precision_flags(config.quantization.enabled),
                "logging_steps": 20,
                "save_strategy": "epoch",
                "seed": seed,
                "dataset_text_field": "text",
                "report_to": "none",
                "max_seq_length": config.max_seq_len,
                "max_length": config.max_seq_len,
            }

            sig_sft = inspect.signature(TRLSFTConfig.__init__).parameters
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_sft.values())

            if has_var_kwargs:
                filtered_kwargs = dict(candidate_kwargs)
                if "max_length" in sig_sft and "max_seq_length" in filtered_kwargs:
                    del filtered_kwargs["max_seq_length"]
            else:
                filtered_kwargs = {
                    k: v for k, v in candidate_kwargs.items() if k in sig_sft
                }

            trl_args = TRLSFTConfig(**filtered_kwargs)
            apply_t4_trainer_precision(trl_args, config.quantization.enabled)
            print(
                "T4 precision: "
                f"quantized={config.quantization.enabled} "
                f"fp16={getattr(trl_args, 'fp16', None)} "
                f"bf16={getattr(trl_args, 'bf16', None)}"
            )

            # Apply any remaining training attributes directly onto the config object
            for k, v in candidate_kwargs.items():
                if k not in filtered_kwargs and hasattr(trl_args, k):
                    try:
                        setattr(trl_args, k, v)
                    except Exception:
                        pass

            # Support both tokenizer and processing_class across transformers/trl releases
            trainer_candidate_kwargs: dict[str, Any] = {
                "model": model,
                "train_dataset": hf_ds,
                "args": trl_args,
                "data_collator": collator,
                "tokenizer": tokenizer,
                "processing_class": tokenizer,
            }
            sig_trainer = inspect.signature(SFTTrainer.__init__).parameters
            has_trainer_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_trainer.values()
            )

            if has_trainer_var_kwargs:
                filtered_trainer_kwargs = {
                    "model": model,
                    "train_dataset": hf_ds,
                    "args": trl_args,
                    "data_collator": collator,
                }
                if "processing_class" in sig_trainer:
                    filtered_trainer_kwargs["processing_class"] = tokenizer
                elif "tokenizer" in sig_trainer:
                    filtered_trainer_kwargs["tokenizer"] = tokenizer
            else:
                filtered_trainer_kwargs = {
                    k: v for k, v in trainer_candidate_kwargs.items() if k in sig_trainer
                }
                if "processing_class" in filtered_trainer_kwargs and "tokenizer" in filtered_trainer_kwargs:
                    del filtered_trainer_kwargs["tokenizer"]

            trainer = SFTTrainer(**filtered_trainer_kwargs)
            # Trainer/PEFT can cast adapters back to the checkpoint dtype (bf16).
            for param in trainer.model.parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)

            def _bf16_grad_to_fp32(grad: torch.Tensor | None) -> torch.Tensor | None:
                if grad is not None and grad.dtype == torch.bfloat16:
                    return grad.float()
                return grad

            if config.quantization.enabled:
                for param in trainer.model.parameters():
                    if param.requires_grad:
                        param.register_hook(_bf16_grad_to_fp32)

            train_res = trainer.train()
            trainer.save_model(str(out_dir))
            # Also save to base output_dir for direct PEFT checkpoint resolution
            trainer.save_model(str(config.output_dir))
            metrics = {
                "train_loss": float(train_res.training_loss),
                "seed": seed,
            }
            tracker.log_metrics(metrics)
            seed_reports.append(metrics)

    return {"run_name": config.run_name, "seeds": seed_reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="T4-safe LoRA/QLoRA SFT trainer")
    parser.add_argument(
        "--config", default="configs/sft_qwen15b_lora.yaml", help="Path to SFT YAML"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate T4 config and dataset formatting on CPU"
    )
    parser.add_argument(
        "--lora-r", type=int, default=None, help="Optional LoRA rank override for ablations"
    )
    parser.add_argument(
        "--subset-size", type=int, default=None, help="Optional training set size override"
    )
    parser.add_argument(
        "--schema-free", action="store_true", help="Ablation: omit schema from prompt"
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed override")
    args = parser.parse_args()

    cfg = SFTConfig.from_yaml(args.config)
    if args.lora_r is not None:
        cfg.lora.r = args.lora_r
        cfg.lora.alpha = args.lora_r * 2
    if args.subset_size is not None:
        cfg.train_subset_size = args.subset_size
    if args.schema_free:
        cfg.include_schema_in_prompt = False

    res = run_sft(cfg, dry_run=args.dry_run, seed_override=args.seed)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
