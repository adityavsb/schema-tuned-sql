"""Schema, business-definition, and few-shot example retriever for B1, B2, and S4 systems."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "warehouse" / "schema.sql"
DEFAULT_DEFS_PATH = REPO_ROOT / "warehouse" / "business_definitions.md"
DEFAULT_DEV_PATH = REPO_ROOT / "data" / "dev.jsonl"

SYSTEM_INSTRUCTION = (
    "You are a DuckDB SQL analytics assistant for MetricPulse. "
    "Given the database schema, business metric definitions, and user question, "
    "return ONLY a valid read-only DuckDB SQL query inside a ```sql ... ``` block. "
    "If the question cannot be answered from the schema or requests private/unavailable data, "
    "respond with exactly: REFUSAL: UNANSWERABLE_FROM_SCHEMA"
)


def tokenize(text: str) -> list[str]:
    """Lowercases and extracts alphanumeric/underscore tokens."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Scorer:
    """Lightweight, zero-dependency BM25 Okapi ranker for schema and example retrieval."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.n_docs = len(documents)
        self.doc_freqs: list[Counter[str]] = [Counter(t) for t in self.doc_tokens]
        self.idf: dict[str, float] = {}
        self._build_idf()

    def _build_idf(self) -> None:
        df: Counter[str] = Counter()
        for tf in self.doc_freqs:
            for term in tf:
                df[term] += 1
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score_all(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        scores = [0.0] * self.n_docs
        for idx in range(self.n_docs):
            tf = self.doc_freqs[idx]
            dl = self.doc_lens[idx]
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                f = tf[term]
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self.avgdl)))
                s += idf * (f * (self.k1 + 1.0)) / denom
            scores[idx] = s
        return scores

    def top_k_indices(self, query: str, k: int) -> list[int]:
        scores = self.score_all(query)
        ranked = sorted(range(self.n_docs), key=lambda i: (scores[i], -i), reverse=True)
        return ranked[: min(k, self.n_docs)]


class SchemaDefinitionRetriever:
    """Retrieves relevant table DDLs, business definitions, and dev examples for a query."""

    def __init__(
        self,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        defs_path: Path = DEFAULT_DEFS_PATH,
        dev_path: Path = DEFAULT_DEV_PATH,
    ):
        self.schema_text = schema_path.read_text(encoding="utf-8")
        self.defs_text = defs_path.read_text(encoding="utf-8")
        self.tables = self._parse_tables(self.schema_text)
        self.definitions = self._parse_definitions(self.defs_text)
        self.dev_examples = self._load_dev_examples(dev_path)

        self._table_scorer = BM25Scorer([f"{t['name']} {t['ddl']}" for t in self.tables])
        self._def_scorer = BM25Scorer([f"{d['title']} {d['body']}" for d in self.definitions])
        self._ex_scorer = BM25Scorer(
            [
                f"{ex['question']} {ex.get('category', '')} {ex.get('difficulty', '')}"
                for ex in self.dev_examples
            ]
        )

    @staticmethod
    def _parse_tables(schema_sql: str) -> list[dict[str, str]]:
        blocks = re.findall(
            r"(CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([a-zA-Z0-9_]+)\s*\(.*?\);)",
            schema_sql,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return [{"name": name.lower(), "ddl": ddl.strip()} for ddl, name in blocks]

    @staticmethod
    def _parse_definitions(md_text: str) -> list[dict[str, str]]:
        sections = re.split(r"\n###\s+", md_text)
        parsed: list[dict[str, str]] = []
        for sec in sections[1:]:
            lines = sec.strip().splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            parsed.append({"title": title, "body": f"### {title}\n{body}"})
        return parsed

    @staticmethod
    def _load_dev_examples(dev_path: Path) -> list[dict[str, Any]]:
        if not dev_path.is_file():
            return []
        return [
            json.loads(line)
            for line in dev_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def retrieve(
        self,
        question: str,
        top_tables: int = 5,
        top_defs: int = 2,
        top_examples: int = 3,
    ) -> dict[str, Any]:
        """Retrieves top-scoring tables (with FK expansion), definitions, and dev examples."""
        expanded_q = self._expand_domain_terms(question)
        t_indices = self._table_scorer.top_k_indices(expanded_q, top_tables)
        selected_tables = [self.tables[i] for i in t_indices]

        # Ensure foreign-key parent tables (e.g. accounts, users) are included if referenced
        selected_names = {t["name"] for t in selected_tables}
        combined_ddl = "\n".join(t["ddl"] for t in selected_tables)
        for tbl in self.tables:
            if f"REFERENCES {tbl['name']}(" in combined_ddl and tbl["name"] not in selected_names:
                selected_tables.append(tbl)
                selected_names.add(tbl["name"])

        d_indices = self._def_scorer.top_k_indices(expanded_q, top_defs)
        selected_defs = [self.definitions[i] for i in d_indices]

        e_indices = self._ex_scorer.top_k_indices(question, top_examples)
        selected_examples = [self.dev_examples[i] for i in e_indices] if self.dev_examples else []

        return {
            "tables": selected_tables,
            "definitions": selected_defs,
            "examples": selected_examples,
        }

    @staticmethod
    def _expand_domain_terms(question: str) -> str:
        q_low = question.lower()
        extras: list[str] = []
        if "wau" in q_low or "weekly active" in q_low or "active user" in q_low:
            extras.append(
                "Weekly Active User WAU events dashboard_create query_run export_csv invite_sent"
            )
        if "activat" in q_low:
            extras.append("Activated Account users events query_run 14 days")
        if "mrr" in q_low or "arr" in q_low or "recurring" in q_low or "revenue" in q_low:
            extras.append(
                "Monthly Annual Recurring Revenue subscriptions status active mrr_amount invoices"
            )
        if "sla" in q_low or "urgent" in q_low or "ticket" in q_low:
            extras.append(
                "Urgent Ticket SLA Compliance support_tickets priority resolved_at created_at"
            )
        if (
            "experiment" in q_low
            or "variant" in q_low
            or "treatment" in q_low
            or "control" in q_low
        ):
            extras.append("Variant Conversion experiments experiment_assignments accounts events")
        return f"{question} {' '.join(extras)}".strip()

    def format_prompt(
        self,
        question: str,
        mode: str = "zero_shot",
        few_shot_k: int = 5,
        top_tables: int = 5,
        top_defs: int = 2,
        top_examples: int = 3,
        include_schema: bool = True,
    ) -> str:
        """Formats the user prompt for B0 (zero_shot), B1 (few_shot), B2/S4 (schema_rag), or SFT."""
        parts: list[str] = []

        if mode == "schema_rag":
            bundle = self.retrieve(
                question,
                top_tables=top_tables,
                top_defs=top_defs,
                top_examples=top_examples,
            )
            if include_schema:
                ddl_block = "\n\n".join(t["ddl"] for t in bundle["tables"])
                parts.append(f"## Relevant Database Tables (DuckDB)\n```sql\n{ddl_block}\n```")
            if bundle["definitions"]:
                def_block = "\n\n".join(d["body"] for d in bundle["definitions"])
                parts.append(f"## Relevant Business Definitions\n{def_block}")
            if bundle["examples"]:
                ex_lines = []
                for ex in bundle["examples"]:
                    ex_lines.append(f"Q: {ex['question']}\n```sql\n{ex['gold_sql']}\n```")
                parts.append("## Similar Verified Examples\n" + "\n\n".join(ex_lines))
        else:
            if include_schema:
                ddl_block = "\n\n".join(t["ddl"] for t in self.tables)
                parts.append(f"## Database Schema (DuckDB)\n```sql\n{ddl_block}\n```")
                def_block = "\n\n".join(d["body"] for d in self.definitions)
                parts.append(f"## Customer Business Definitions\n{def_block}")

            if mode == "few_shot" and self.dev_examples and few_shot_k > 0:
                demos = self.dev_examples[:few_shot_k]
                ex_lines = [f"Q: {ex['question']}\n```sql\n{ex['gold_sql']}\n```" for ex in demos]
                parts.append("## Few-Shot Examples\n" + "\n\n".join(ex_lines))

        parts.append(
            f"## Question\n{question}\n\nReturn only the DuckDB SQL query or REFUSAL: UNANSWERABLE_FROM_SCHEMA."
        )
        return "\n\n".join(parts)
