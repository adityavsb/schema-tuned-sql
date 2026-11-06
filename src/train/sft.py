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
    if config.max_seq_len > 2048:
        raise T4HardwareConstraintError(
            f"Sequence length {config.max_seq_len} exceeds safe 16 GB T4 VRAM budget (max 2048)."
        )
    if config.quantization.enabled and config.quantization.bnb_4bit_compute_dtype != "float16":
        raise T4HardwareConstraintError(
            "QLoRA on Tesla T4 requires bnb_4bit_compute_dtype='float16'."
        )


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
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
    from trl import SFTConfig as TRLSFTConfig

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
            trl_args = TRLSFTConfig(
                output_dir=str(out_dir),
                max_seq_length=config.max_seq_len,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=config.gradient_checkpointing,
                optim=config.optimizer,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler,
                warmup_ratio=config.warmup_ratio,
                max_grad_norm=config.max_grad_norm,
                num_train_epochs=config.epochs,
                fp16=True,
                bf16=False,
                logging_steps=20,
                save_strategy="epoch",
                seed=seed,
                dataset_text_field="text",
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=hf_ds,
                args=trl_args,
                data_collator=collator,
            )
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
