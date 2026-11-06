"""Chat-template formatting and completion-only token masking for SQL SFT."""

from __future__ import annotations

from typing import Any

from src.eval.rag import SYSTEM_INSTRUCTION, SchemaDefinitionRetriever

ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def format_sft_example(
    item: dict[str, Any],
    retriever: SchemaDefinitionRetriever,
    include_schema: bool = True,
) -> dict[str, str]:
    """Formats a training item using the Qwen ChatML template with SQL-only completion."""
    user_content = retriever.format_prompt(
        question=item["question"],
        mode="zero_shot",
        include_schema=include_schema,
    )
    sql_or_refusal = item.get("sql") or item.get("gold_sql") or ""
    if item.get("is_negative_query") or sql_or_refusal.startswith("REFUSAL"):
        assistant_content = "REFUSAL: UNANSWERABLE_FROM_SCHEMA"
    else:
        assistant_content = f"```sql\n{sql_or_refusal.strip()}\n```"

    prompt_prefix = (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTION}{IM_END}\n"
        f"<|im_start|>user\n{user_content}{IM_END}\n"
        f"{ASSISTANT_HEADER}"
    )
    full_text = f"{prompt_prefix}{assistant_content}{IM_END}\n"
    return {
        "prompt": prompt_prefix,
        "completion": f"{assistant_content}{IM_END}\n",
        "text": full_text,
    }


def mask_completion_only_labels(
    input_ids: list[int],
    response_template_ids: list[int],
    ignore_index: int = -100,
) -> list[int]:
    """Masks all prompt/schema tokens with -100 up to the end of response_template_ids."""
    labels = list(input_ids)
    tpl_len = len(response_template_ids)
    if tpl_len == 0 or tpl_len > len(input_ids):
        return [ignore_index] * len(input_ids)

    match_end = -1
    for i in range(len(input_ids) - tpl_len + 1):
        if input_ids[i : i + tpl_len] == response_template_ids:
            match_end = i + tpl_len

    if match_end == -1:
        return [ignore_index] * len(input_ids)

    for j in range(match_end):
        labels[j] = ignore_index
    return labels


class CompletionOnlyDataCollator:
    """Fallback data collator for completion-only loss masking compatible across TRL/Transformers versions."""

    def __init__(
        self,
        response_template: str = "<|im_start|>assistant\n",
        tokenizer: Any = None,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.ignore_index = ignore_index
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        else:
            self.response_template_ids = []

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        # If examples already have pre-tokenized 'labels' (e.g., built by modern TRL prompt/completion handling)
        if "labels" in examples[0]:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
            max_len = max(len(ex["input_ids"]) for ex in examples)
            batch_input_ids = []
            batch_attention = []
            batch_labels = []
            for ex in examples:
                diff = max_len - len(ex["input_ids"])
                batch_input_ids.append(list(ex["input_ids"]) + [pad_id] * diff)
                att = list(ex.get("attention_mask", [1] * len(ex["input_ids"]))) + [0] * diff
                batch_attention.append(att)
                batch_labels.append(list(ex["labels"]) + [self.ignore_index] * diff)
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

        if "input_ids" in examples[0]:
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            texts = [ex["text"] for ex in examples]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=getattr(self.tokenizer, "model_max_length", 3072),
                return_tensors="pt",
            )

        labels = batch["input_ids"].clone()
        for i in range(len(labels)):
            seq = labels[i].tolist()
            masked = mask_completion_only_labels(
                seq,
                response_template_ids=self.response_template_ids,
                ignore_index=self.ignore_index,
            )
            labels[i] = torch.tensor(masked, dtype=labels.dtype)

        # Mask padding tokens in labels with ignore_index
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = self.ignore_index

        batch["labels"] = labels
        return batch
