# Dataset Card: MetricPulse Analytics Text-to-SQL Benchmark

**Repository:** `schema-tuned-sql`  
**License:** MIT / Synthetic Data  
**Primary Dialect:** DuckDB  
**Author:** Aditya Bhadoria (github.com/adityavsb)

---

## 1. Overview & Dataset Lineage

This dataset provides an authoritative, execution-verified evaluation and distillation corpus modeling a fictional B2B SaaS analytics platform ("MetricPulse"). The warehouse contains 11 normalized relational tables covering organization hierarchy, product telemetry events, subscription billing, A/B experiments, and customer support tickets.

Every ground-truth SQL query in this dataset has been:
1. Validated through AST parse inspection using `sqlglot`.
2. Verified against an AST read-only security guardrail blocking mutations and unauthorized functions.
3. Executed against a populated DuckDB analytical instance (`warehouse/generate_data.py`, seed=42) to confirm query termination and non-empty result sets.

---

## 2. Evaluation Splits & Cryptographic Hashes

To ensure evaluation integrity and prevent data contamination during distillation or hyperparameter selection, the test suite is strictly frozen:

| Split | File | Item Count | Difficulty Mix | Primary Purpose | SHA-256 Checksum |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Test (Frozen)** | `data/test_gold.jsonl` | **220** | 60 Easy (30%) · 80 Med (40%) · 60 Hard (30%) · 20 Refusal | Final unbiased system ranking | `84a938a29aa438c1c901f4dfadbfe13af01451ef2c55177010493133f90b6b5c` |
| **Dev** | `data/dev.jsonl` | **100** | 30 Easy (30%) · 40 Med (40%) · 30 Hard (30%) | Prompt tuning, few-shot demos, hyperparameter selection | `2e5064b46fe8001332aff4b21398fa86d134fa15d26c1113606603c57e44925f` |
| **Train (Distilled)** | `data/train/` | **2,000** | 591 Easy (30%) · 765 Med (38%) · 644 Hard (32%) | Scaled across 250 / 500 / 1k / 2k subsets | `529c352e0907...` (`data/train/manifest.json`) |

---

## 3. Difficulty Stratification

- **Easy (30%):** Single-table aggregations, counts, sums, basic filters, and status group-bys. Tests model syntax fluency and table mapping.
- **Medium (40%):** Multi-table joins (2–3 tables: accounts + subscriptions, users + events, invoices + subscriptions), date truncations (`DATE_TRUNC`), `HAVING` filters, and subqueries.
- **Hard (30%):** Complex multi-step CTEs, window functions (`ROW_NUMBER()`, `DENSE_RANK()`, `LAG()`), and customer business definition metrics:
  - **Weekly Active Users (WAU):** 7-day rolling window excluding passive `page_view` events.
  - **Account Activation:** $\ge 3$ users and $\ge 1$ `query_run` within 14 days of account creation.
  - **MRR & ARR:** Active recurring revenue filtering.
  - **Urgent SLA Breaches:** Urgent support tickets resolved $> 4\text{ hours}$ after creation.
  - **Experiment Lift:** Cross-variant event frequency comparison (`exp_001`).
- **Refusal / Adversarial Negatives (20 items):** Unanswerable queries requesting non-existent attributes (employee salaries, credit scores, plaintext passwords, external stock forecasts) requiring the model to emit an explicit refusal or clarification token rather than hallucinating SQL.

---

## 4. Contamination Prevention Protocol

1. **Strict Partitioning:** Test queries are permanently quarantined. `data/test_gold.jsonl` is never used for prompt demonstrations, few-shot in-context examples, or data generation seeds.
2. **Deduplication Filter:** The distillation pipeline (`src/datagen/dedupe.py`) compares all teacher-generated candidate queries against `test_gold.jsonl` and `dev.jsonl` using both normalized AST equivalence and token edit distance. Any synthetic item with similarity $\ge 0.85$ or identical AST is rejected.
3. **Reproducibility:** Evaluation splits can be re-verified at any time via:
   ```bash
   python scripts/build_gold_eval_suites.py
   ```
