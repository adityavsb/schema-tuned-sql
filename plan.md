# Engineering Plan: `schema-tuned-sql`

Fine-tuning small open LLMs (`Qwen2.5-Coder-1.5B-Instruct` and `3B-Instruct`) on NVIDIA Tesla T4 GPUs for customer-specific analytics text-to-SQL in DuckDB via execution-verified distillation.

---

## 1. Headline Question

> *Can a 1.5B–3B model, fine-tuned on a few thousand execution-verified examples for one customer schema, match a much larger API model on that schema at a fraction of the cost per query, and what does it lose on SQL in general?*

---

## 2. Quantitative Targets (Pre-Registered)

| Target ID | Metric | Target Threshold |
| :--- | :--- | :--- |
| **T-1 (Schema Adaptation Gain)** | 3B QLoRA student (`S2`) vs best prompting baseline (`B2` Schema-RAG) on frozen `test_gold.jsonl` ($n=200$ answerable) | **$\ge +10.0\text{ pp}$ EX** |
| **T-2 (Teacher Parity)** | 3B QLoRA student (`S2`) vs DeepSeek Teacher API (`T1`) on frozen `test_gold.jsonl` | **Within $5.0\text{ pp}$ EX** |
| **T-3 (Unit Economics)** | Serving cost per 1,000 queries on a single Tesla T4 (batch 8) vs OpenRouter Teacher API | **$\ge 5\times$ cheaper** |
| **T-4 (General SQL Retention)** | Catastrophic forgetting on 200-item cross-domain holdout (`data/general_sql_holdout.jsonl`) | **$\le 5.0\text{ pp}$ EX drop** |

---

## 3. Systems Compared

| ID | System | Model / Configuration |
| :--- | :--- | :--- |
| **B0** | Base Zero-Shot | `Qwen/Qwen2.5-Coder-3B-Instruct` (full schema + business definitions in prompt) |
| **B1** | Base Few-Shot ($k=5$) | `Qwen/Qwen2.5-Coder-3B-Instruct` + 5 dev demonstrations |
| **B2** | Base + Schema/Definition RAG | `Qwen/Qwen2.5-Coder-3B-Instruct` + BM25 top-5 tables, top-2 definitions, top-3 dev examples |
| **T1** | Teacher API (DeepSeek) | `deepseek/deepseek-chat` via OpenRouter |
| **T2** | Peer API (Qwen-32B) | `qwen/qwen-2.5-coder-32b-instruct` via OpenRouter |
| **S1** | LoRA SFT 1.5B | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (FP16 LoRA, $r=16$, $\alpha=32$, completion-only loss) |
| **S2** | QLoRA SFT 3B | `Qwen/Qwen2.5-Coder-3B-Instruct` (4-bit NF4 QLoRA, double quant, FP16 compute) |
| **S3** | QLoRA SFT 7B (Stretch) | `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit NF4 QLoRA, max seq 1024) |
| **S4** | QLoRA SFT 3B + RAG | `S2` checkpoint + BM25 schema/definition retrieval at inference |
| **S5** | Execution DPO/ORPO (Stretch) | Preference pairs `(verified_sql, execution_failed_sql)` on top of `S1` |

---

## 4. Milestones

- **M1:** Warehouse (`warehouse/schema.sql`, `generate_data.py`), business definitions (`business_definitions.md`), AST guardrail (`src/utils/db.py`), and frozen evaluation splits (`data/test_gold.jsonl`, `data/dev.jsonl`).
- **M2:** OpenRouter provider layer (`src/llm/`), SHA-256 disk cache, hard dollar budget cap (`max_usd: 15.0`), and RAG baseline (`src/eval/rag.py`).
- **M3:** Execution-verified distillation pipeline (`src/datagen/`), contamination filtering, scaling splits (`500`, `1k`, `2k`, `5k`), and 200-item general SQL holdout.
- **M4:** T4-safe SFT trainer (`src/train/sft.py`), completion-only loss masking (`src/train/collator.py`), preference builder (`src/train/prefs.py`), and MLflow tracking (`src/utils/tracking.py`).
- **M5:** T4 serving benchmark and break-even cost model (`src/serve/bench.py`).
- **M6:** Technical README report, ADRs (`docs/decisions.md`), remote T4 runner notebook (`notebooks/t4_runner.ipynb`), and automated README-to-JSON provenance tests.
