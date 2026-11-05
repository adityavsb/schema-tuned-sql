"""Analytical question templates and persona styles for teacher distillation."""

from __future__ import annotations

import random
from typing import Any

from warehouse.generate_data import (
    COUNTRIES,
    EVENT_TYPES,
    INDUSTRIES,
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
)

PERSONAS = [
    ("terse_analyst", "Analyst query"),
    ("executive", "Executive summary request"),
    ("ops_lead", "Operations review"),
    ("finance_controller", "Finance audit check"),
    ("product_pm", "Product telemetry review"),
    ("cs_manager", "Customer health check"),
]


def generate_candidate_candidates(seed: int = 42, count: int = 5500) -> list[dict[str, Any]]:
    """Generates a diverse pool of analytical question prompts across difficulty buckets and personas."""
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []

    tiers = ["starter", "growth", "enterprise"]
    roles = ["admin", "member", "viewer"]
    devices = ["desktop", "mobile", "tablet"]
    cycles = ["monthly", "annual"]
    inv_statuses = ["paid", "open", "void"]

    idx = 1
    while len(candidates) < count:
        persona_id, persona_label = rng.choice(PERSONAS)
        roll = rng.random()

        if roll < 0.08:
            unanswerable_topics = [
                f"[{persona_label} #{idx}] What is the personal credit score of billing contact #{idx} in {rng.choice(COUNTRIES)}?",
                f"[{persona_label} #{idx}] List the employee base salaries for support agents handling {rng.choice(TICKET_CATEGORIES)} tickets.",
                f"[{persona_label} #{idx}] Show the home street address and private cell number for users in {rng.choice(INDUSTRIES)} accounts.",
                f"[{persona_label} #{idx}] Predict the exact stock price of our {rng.choice(tiers)} tier customers {rng.randint(10, 90)} days from now.",
                f"[{persona_label} #{idx}] Provide the plaintext password and session cookie for user_{idx:04d}.",
                f"[{persona_label} #{idx}] What is the AWS EC2 instance hardware temperature for workspace_{idx:03d}?",
            ]
            q = rng.choice(unanswerable_topics)
            sql = "REFUSAL: UNANSWERABLE_FROM_SCHEMA"
            candidates.append(
                {
                    "candidate_id": f"cand_{idx:05d}",
                    "difficulty": "hard",
                    "category": "refusal",
                    "persona": persona_id,
                    "question": q,
                    "candidate_sql": sql,
                    "is_negative_query": True,
                }
            )
            idx += 1
            continue

        diff = rng.choices(["easy", "medium", "hard"], weights=[0.30, 0.40, 0.30], k=1)[0]
        day_cutoff = f"2023-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}"

        if diff == "easy":
            variant = idx % 6
            if variant == 0:
                ind = rng.choice(INDUSTRIES)
                tier = rng.choice(tiers)
                q = f"[{persona_label} #{idx}] For '{ind}' organizations on '{tier}' or any tier created on or after '2023-01-01', count accounts matching industry '{ind}'."
                sql = f"SELECT COUNT(*) AS cnt_{idx} FROM accounts WHERE industry = '{ind}' AND created_at >= '2023-01-01'::TIMESTAMP"
            elif variant == 1:
                role = rng.choice(roles)
                q = f"[{persona_label} #{idx}] How many users have role '{role}' and were created on or after '2023-01-01'?"
                sql = f"SELECT COUNT(*) AS cnt_{idx} FROM users WHERE role = '{role}' AND created_at >= '2023-01-01'::TIMESTAMP"
            elif variant == 2:
                et = rng.choice(EVENT_TYPES)
                dev = rng.choice(devices)
                q = f"[{persona_label} #{idx}] Count all '{et}' telemetry events logged from '{dev}' clients since {day_cutoff}."
                sql = f"SELECT COUNT(*) AS cnt_{idx} FROM events WHERE event_type = '{et}' AND device = '{dev}' AND timestamp >= '2023-01-01'::TIMESTAMP"
            elif variant == 3:
                prio = rng.choice(TICKET_PRIORITIES)
                q = f"[{persona_label} #{idx}] How many support tickets have priority '{prio}' created since {day_cutoff}?"
                sql = f"SELECT COUNT(*) AS cnt_{idx} FROM support_tickets WHERE priority = '{prio}' AND created_at >= '2023-01-01'::TIMESTAMP"
            elif variant == 4:
                cyc = rng.choice(cycles)
                q = f"[{persona_label} #{idx}] What is the total MRR across active '{cyc}' billing subscriptions (audit ref {idx})?"
                sql = f"SELECT SUM(mrr_amount) AS mrr_{idx} FROM subscriptions WHERE status = 'active' AND billing_cycle = '{cyc}'"
            else:
                ist = rng.choice(inv_statuses)
                q = f"[{persona_label} #{idx}] What is the average invoice amount for '{ist}' invoices (batch {idx})?"
                sql = f"SELECT AVG(amount) AS avg_{idx} FROM invoices WHERE status = '{ist}'"

            candidates.append(
                {
                    "candidate_id": f"cand_{idx:05d}",
                    "difficulty": "easy",
                    "category": "aggregation",
                    "persona": persona_id,
                    "question": q,
                    "candidate_sql": sql,
                    "is_negative_query": False,
                }
            )

        elif diff == "medium":
            variant = idx % 6
            if variant == 0:
                ind = rng.choice(INDUSTRIES)
                q = f"[{persona_label} #{idx}] What is the total active MRR for '{ind}' accounts (slice {idx})?"
                sql = (
                    f"SELECT SUM(s.mrr_amount) AS mrr_{idx} FROM subscriptions s "
                    f"JOIN accounts a ON s.account_id = a.account_id "
                    f"WHERE s.status = 'active' AND a.industry = '{ind}'"
                )
            elif variant == 1:
                tier = rng.choice(tiers)
                et = rng.choice(EVENT_TYPES)
                lim = (idx % 8) + 3
                q = f"[{persona_label} #{idx}] Show the top {lim} '{tier}' accounts by volume of '{et}' events."
                sql = (
                    f"SELECT a.name, COUNT(e.event_id) AS evt_cnt_{idx} "
                    f"FROM accounts a JOIN users u ON a.account_id = u.account_id "
                    f"JOIN events e ON u.user_id = e.user_id "
                    f"WHERE a.plan_tier = '{tier}' AND e.event_type = '{et}' "
                    f"GROUP BY a.name ORDER BY evt_cnt_{idx} DESC LIMIT {lim}"
                )
            elif variant == 2:
                ctry = rng.choice(COUNTRIES)
                q = f"[{persona_label} #{idx}] Sum paid invoice amounts by account for '{ctry}' organizations (ref {idx})."
                sql = (
                    f"SELECT a.name, SUM(i.amount) AS paid_{idx} "
                    f"FROM accounts a JOIN invoices i ON a.account_id = i.account_id "
                    f"WHERE a.country = '{ctry}' AND i.status = 'paid' "
                    f"GROUP BY a.name ORDER BY paid_{idx} DESC"
                )
            elif variant == 3:
                prio = rng.choice(TICKET_PRIORITIES)
                q = f"[{persona_label} #{idx}] Breakdown the count of '{prio}' priority tickets by account plan tier (ref {idx})."
                sql = (
                    f"SELECT a.plan_tier, COUNT(t.ticket_id) AS cnt_{idx} "
                    f"FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id "
                    f"WHERE t.priority = '{prio}' "
                    f"GROUP BY a.plan_tier ORDER BY cnt_{idx} DESC"
                )
            elif variant == 4:
                role = rng.choice(roles)
                tier = rng.choice(tiers)
                q = f"[{persona_label} #{idx}] Count distinct users with role '{role}' who triggered events in '{tier}' tier accounts."
                sql = (
                    f"SELECT COUNT(DISTINCT u.user_id) AS u_{idx} "
                    f"FROM users u JOIN accounts a ON u.account_id = a.account_id "
                    f"JOIN events e ON u.user_id = e.user_id "
                    f"WHERE u.role = '{role}' AND a.plan_tier = '{tier}'"
                )
            else:
                exp_id = rng.choice(["exp_001", "exp_002"])
                var = rng.choice(["control", "treatment"])
                q = f"[{persona_label} #{idx}] What is the total active MRR of accounts assigned to '{var}' in experiment '{exp_id}'?"
                sql = (
                    f"SELECT SUM(s.mrr_amount) AS exp_mrr_{idx} "
                    f"FROM experiment_assignments ea JOIN subscriptions s ON ea.account_id = s.account_id "
                    f"WHERE ea.experiment_id = '{exp_id}' AND ea.variant = '{var}' AND s.status = 'active'"
                )

            candidates.append(
                {
                    "candidate_id": f"cand_{idx:05d}",
                    "difficulty": "medium",
                    "category": "join",
                    "persona": persona_id,
                    "question": q,
                    "candidate_sql": sql,
                    "is_negative_query": False,
                }
            )

        else:
            variant = idx % 5
            if variant == 0:
                ctry = rng.choice(COUNTRIES)
                q = f"[{persona_label} #{idx}] Calculate the Weekly Active Users (WAU) definition count (excluding page_view) for '{ctry}' accounts."
                sql = (
                    f"SELECT COUNT(DISTINCT e.user_id) AS wau_{idx} "
                    f"FROM events e JOIN users u ON e.user_id = u.user_id "
                    f"JOIN accounts a ON u.account_id = a.account_id "
                    f"WHERE e.event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent') "
                    f"AND a.country = '{ctry}'"
                )
            elif variant == 1:
                tier = rng.choice(tiers)
                q = f"[{persona_label} #{idx}] For '{tier}' tier accounts, evaluate Activated Account status (>=3 users and >=1 query_run within 14 days of creation)."
                sql = (
                    f"WITH u_cnt AS ("
                    f"SELECT a.account_id, COUNT(DISTINCT u.user_id) AS n_users "
                    f"FROM accounts a JOIN users u ON a.account_id = u.account_id "
                    f"WHERE u.created_at <= a.created_at + INTERVAL '14 days' GROUP BY a.account_id"
                    f"), q_cnt AS ("
                    f"SELECT a.account_id, COUNT(e.event_id) AS n_queries "
                    f"FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id "
                    f"WHERE e.event_type = 'query_run' AND e.timestamp <= a.created_at + INTERVAL '14 days' GROUP BY a.account_id"
                    f") SELECT a.account_id, CASE WHEN COALESCE(u.n_users, 0) >= 3 AND COALESCE(q.n_queries, 0) >= 1 THEN TRUE ELSE FALSE END AS is_act_{idx} "
                    f"FROM accounts a LEFT JOIN u_cnt u ON a.account_id = u.account_id LEFT JOIN q_cnt q ON a.account_id = q.account_id "
                    f"WHERE a.plan_tier = '{tier}' ORDER BY a.account_id"
                )
            elif variant == 2:
                ind = rng.choice(INDUSTRIES)
                lim = (idx % 4) + 2
                q = f"[{persona_label} #{idx}] Rank the top {lim} accounts by active MRR within industry '{ind}' using DENSE_RANK."
                sql = (
                    f"WITH rnk_cte AS ("
                    f"SELECT a.name, s.mrr_amount, DENSE_RANK() OVER (ORDER BY s.mrr_amount DESC) AS rnk_{idx} "
                    f"FROM accounts a JOIN subscriptions s ON a.account_id = s.account_id "
                    f"WHERE s.status = 'active' AND a.industry = '{ind}'"
                    f") SELECT name, mrr_amount, rnk_{idx} FROM rnk_cte WHERE rnk_{idx} <= {lim}"
                )
            elif variant == 3:
                cat = rng.choice(TICKET_CATEGORIES)
                hrs = rng.choice([8, 12, 24, 36])
                q = f"[{persona_label} #{idx}] Compute the percentage of resolved support tickets in category '{cat}' resolved within {hrs} hours."
                sql = (
                    f"SELECT (CAST(COUNT(CASE WHEN resolved_at <= created_at + INTERVAL '{hrs} hours' THEN 1 END) AS DOUBLE) / COUNT(*)) * 100.0 AS pct_{idx} "
                    f"FROM support_tickets WHERE category = '{cat}' AND status = 'resolved'"
                )
            else:
                exp_id = rng.choice(["exp_001", "exp_002"])
                et = rng.choice(["query_run", "dashboard_create", "export_csv"])
                q = f"[{persona_label} #{idx}] Compare per-account '{et}' event rates between control and treatment cohorts in experiment '{exp_id}'."
                sql = (
                    f"SELECT ea.variant, CAST(COUNT(e.event_id) AS DOUBLE) / COUNT(DISTINCT a.account_id) AS rate_{idx} "
                    f"FROM experiment_assignments ea JOIN accounts a ON ea.account_id = a.account_id "
                    f"JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id "
                    f"WHERE ea.experiment_id = '{exp_id}' AND e.event_type = '{et}' "
                    f"GROUP BY ea.variant ORDER BY ea.variant"
                )

            candidates.append(
                {
                    "candidate_id": f"cand_{idx:05d}",
                    "difficulty": "hard",
                    "category": "business_definition",
                    "persona": persona_id,
                    "question": q,
                    "candidate_sql": sql,
                    "is_negative_query": False,
                }
            )

        idx += 1

    return candidates
