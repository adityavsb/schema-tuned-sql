"""Constructs and validates the 220-item frozen test suite and 100-item dev suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.db import ReadOnlySQLGuardrail
from warehouse.generate_data import (
    COUNTRIES,
    EVENT_TYPES,
    INDUSTRIES,
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
    generate_synthetic_warehouse,
)


def create_eval_suites() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conn = generate_synthetic_warehouse(
        db_path=":memory:", seed=42, num_accounts=100, num_events=15000
    )
    guardrail = ReadOnlySQLGuardrail()

    test_gold: list[dict] = []
    dev_gold: list[dict] = []

    # Helper to add and immediately verify
    def add_case(suite, q_id, diff, cat, q, sql, is_neg=False):
        if not is_neg:
            assert guardrail.is_safe(sql), f"Guardrail failed: {sql}"
            res = conn.execute(sql).fetchall()
            assert len(res) > 0, f"Query returned empty: {sql}"
        item = {
            "id": q_id,
            "difficulty": diff,
            "category": cat,
            "question": q,
            "gold_sql": sql,
            "is_negative_query": is_neg,
            "expected_refusal": is_neg,
        }
        suite.append(item)

    print("Building test suite (200 answerable + 20 refusal)...")

    # --- EASY TEST CASES (60) ---
    easy_specs = [
        (
            "How many accounts are in active status?",
            "SELECT COUNT(*) FROM accounts WHERE status = 'active'",
        ),
        ("What is the total number of users in the platform?", "SELECT COUNT(*) FROM users"),
        (
            "How many customer accounts belong to the Enterprise Software industry?",
            "SELECT COUNT(*) FROM accounts WHERE industry = 'Enterprise Software'",
        ),
        ("What is the minimum base MRR across all plan tiers?", "SELECT MIN(base_mrr) FROM plans"),
        ("What is the maximum base MRR across all plan tiers?", "SELECT MAX(base_mrr) FROM plans"),
        (
            "How many accounts are registered in the US?",
            "SELECT COUNT(*) FROM accounts WHERE country = 'US'",
        ),
        (
            "How many users currently have the admin role?",
            "SELECT COUNT(*) FROM users WHERE role = 'admin'",
        ),
        (
            "How many support tickets are marked with urgent priority?",
            "SELECT COUNT(*) FROM support_tickets WHERE priority = 'urgent'",
        ),
        (
            "How many support tickets have a status of open?",
            "SELECT COUNT(*) FROM support_tickets WHERE status = 'open'",
        ),
        (
            "What is the total number of feature flags in the database?",
            "SELECT COUNT(*) FROM feature_flags",
        ),
        (
            "List all enabled feature flags.",
            "SELECT flag_name FROM feature_flags WHERE is_enabled = TRUE",
        ),
        (
            "How many accounts are currently on the growth plan tier?",
            "SELECT COUNT(*) FROM accounts WHERE plan_tier = 'growth'",
        ),
        (
            "How many accounts are currently on the starter plan tier?",
            "SELECT COUNT(*) FROM accounts WHERE plan_tier = 'starter'",
        ),
        (
            "How many accounts are currently on the enterprise plan tier?",
            "SELECT COUNT(*) FROM accounts WHERE plan_tier = 'enterprise'",
        ),
        (
            "What is the total number of events recorded in the system?",
            "SELECT COUNT(*) FROM events",
        ),
        (
            "How many query_run events were recorded?",
            "SELECT COUNT(*) FROM events WHERE event_type = 'query_run'",
        ),
        (
            "How many dashboard_create events were recorded?",
            "SELECT COUNT(*) FROM events WHERE event_type = 'dashboard_create'",
        ),
        (
            "How many export_csv events were recorded?",
            "SELECT COUNT(*) FROM events WHERE event_type = 'export_csv'",
        ),
        (
            "How many invite_sent events were recorded?",
            "SELECT COUNT(*) FROM events WHERE event_type = 'invite_sent'",
        ),
        (
            "How many page_view events were recorded?",
            "SELECT COUNT(*) FROM events WHERE event_type = 'page_view'",
        ),
        (
            "How many events occurred on desktop devices?",
            "SELECT COUNT(*) FROM events WHERE device = 'desktop'",
        ),
        (
            "How many events occurred on mobile devices?",
            "SELECT COUNT(*) FROM events WHERE device = 'mobile'",
        ),
        ("What is the total number of workspaces created?", "SELECT COUNT(*) FROM workspaces"),
        (
            "How many total subscriptions exist in the database?",
            "SELECT COUNT(*) FROM subscriptions",
        ),
        (
            "What is the total number of active subscriptions?",
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'active'",
        ),
        (
            "How many subscriptions are past due?",
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'past_due'",
        ),
        (
            "How many subscriptions have been canceled?",
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'canceled'",
        ),
        ("What is the total number of invoices issued?", "SELECT COUNT(*) FROM invoices"),
        (
            "How many invoices are currently marked as paid?",
            "SELECT COUNT(*) FROM invoices WHERE status = 'paid'",
        ),
        (
            "How many invoices are currently open?",
            "SELECT COUNT(*) FROM invoices WHERE status = 'open'",
        ),
        (
            "What is the total sum of all paid invoices?",
            "SELECT SUM(amount) FROM invoices WHERE status = 'paid'",
        ),
        (
            "What is the average amount of paid invoices?",
            "SELECT AVG(amount) FROM invoices WHERE status = 'paid'",
        ),
        (
            "How many users are marked as active?",
            "SELECT COUNT(*) FROM users WHERE is_active = TRUE",
        ),
        (
            "How many users have the role of viewer?",
            "SELECT COUNT(*) FROM users WHERE role = 'viewer'",
        ),
        (
            "How many users have the role of member?",
            "SELECT COUNT(*) FROM users WHERE role = 'member'",
        ),
        (
            "How many accounts are churned?",
            "SELECT COUNT(*) FROM accounts WHERE status = 'churned'",
        ),
        (
            "How many accounts are suspended?",
            "SELECT COUNT(*) FROM accounts WHERE status = 'suspended'",
        ),
        (
            "List all unique industries represented in customer accounts.",
            "SELECT DISTINCT industry FROM accounts ORDER BY industry",
        ),
        (
            "List all unique countries where customer accounts reside.",
            "SELECT DISTINCT country FROM accounts ORDER BY country",
        ),
        (
            "How many support tickets are in the billing category?",
            "SELECT COUNT(*) FROM support_tickets WHERE category = 'billing'",
        ),
        (
            "How many support tickets are in the bug category?",
            "SELECT COUNT(*) FROM support_tickets WHERE category = 'bug'",
        ),
        (
            "How many support tickets are in the feature_request category?",
            "SELECT COUNT(*) FROM support_tickets WHERE category = 'feature_request'",
        ),
        (
            "What is the total number of experiment assignments?",
            "SELECT COUNT(*) FROM experiment_assignments",
        ),
        (
            "How many accounts were assigned to the control variant?",
            "SELECT COUNT(*) FROM experiment_assignments WHERE variant = 'control'",
        ),
        (
            "How many accounts were assigned to the treatment variant?",
            "SELECT COUNT(*) FROM experiment_assignments WHERE variant = 'treatment'",
        ),
        (
            "What is the total base MRR across all starter plans?",
            "SELECT SUM(base_mrr) FROM plans WHERE tier = 'starter'",
        ),
        (
            "How many seats are included in the enterprise plan?",
            "SELECT included_seats FROM plans WHERE tier = 'enterprise'",
        ),
        (
            "How many seats are included in the growth plan?",
            "SELECT included_seats FROM plans WHERE tier = 'growth'",
        ),
        (
            "How many seats are included in the starter plan?",
            "SELECT included_seats FROM plans WHERE tier = 'starter'",
        ),
        (
            "Count the number of active subscriptions with monthly billing cycle.",
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'active' AND billing_cycle = 'monthly'",
        ),
        (
            "Count the number of active subscriptions with annual billing cycle.",
            "SELECT COUNT(*) FROM subscriptions WHERE status = 'active' AND billing_cycle = 'annual'",
        ),
        ("What is the highest invoice amount ever issued?", "SELECT MAX(amount) FROM invoices"),
        ("What is the lowest invoice amount ever issued?", "SELECT MIN(amount) FROM invoices"),
        ("How many experiments are documented in the system?", "SELECT COUNT(*) FROM experiments"),
        (
            "How many accounts are located in Germany (DE)?",
            "SELECT COUNT(*) FROM accounts WHERE country = 'DE'",
        ),
        (
            "How many accounts are located in the United Kingdom (GB)?",
            "SELECT COUNT(*) FROM accounts WHERE country = 'GB'",
        ),
        (
            "How many accounts are located in France (FR)?",
            "SELECT COUNT(*) FROM accounts WHERE country = 'FR'",
        ),
        (
            "How many accounts are in the Fintech industry?",
            "SELECT COUNT(*) FROM accounts WHERE industry = 'Fintech'",
        ),
        (
            "How many accounts are in the Healthcare industry?",
            "SELECT COUNT(*) FROM accounts WHERE industry = 'Healthcare'",
        ),
        ("What is the average base MRR across all plan tiers?", "SELECT AVG(base_mrr) FROM plans"),
    ]

    for i, (q, sql) in enumerate(easy_specs, 1):
        add_case(test_gold, f"test_easy_{i:03d}", "easy", "aggregation", q, sql)

    # --- MEDIUM TEST CASES (80) ---
    medium_specs = [
        (
            "What is the total active MRR grouped by plan tier?",
            "SELECT a.plan_tier, SUM(s.mrr_amount) AS total_mrr FROM subscriptions s JOIN accounts a ON s.account_id = a.account_id WHERE s.status = 'active' GROUP BY a.plan_tier ORDER BY total_mrr DESC",
        ),
        (
            "List the top 5 accounts by number of registered users.",
            "SELECT a.name, COUNT(u.user_id) AS user_count FROM accounts a JOIN users u ON a.account_id = u.account_id GROUP BY a.name ORDER BY user_count DESC LIMIT 5",
        ),
        (
            "How many total events have been generated by enterprise accounts?",
            "SELECT COUNT(e.event_id) FROM events e JOIN workspaces w ON e.workspace_id = w.workspace_id JOIN accounts a ON w.account_id = a.account_id WHERE a.plan_tier = 'enterprise'",
        ),
        (
            "How many users belong to accounts that are currently churned?",
            "SELECT COUNT(u.user_id) FROM users u JOIN accounts a ON u.account_id = a.account_id WHERE a.status = 'churned'",
        ),
        (
            "What is the total revenue collected from paid invoices for each industry?",
            "SELECT a.industry, SUM(i.amount) AS total_paid FROM invoices i JOIN accounts a ON i.account_id = a.account_id WHERE i.status = 'paid' GROUP BY a.industry ORDER BY total_paid DESC",
        ),
        (
            "Find the number of workspaces created per plan tier.",
            "SELECT a.plan_tier, COUNT(w.workspace_id) AS total_workspaces FROM workspaces w JOIN accounts a ON w.account_id = a.account_id GROUP BY a.plan_tier ORDER BY total_workspaces DESC",
        ),
        (
            "What is the average number of users per account by industry?",
            "SELECT a.industry, CAST(COUNT(u.user_id) AS DOUBLE) / COUNT(DISTINCT a.account_id) AS avg_users FROM accounts a LEFT JOIN users u ON a.account_id = u.account_id GROUP BY a.industry ORDER BY avg_users DESC",
        ),
        (
            "List all accounts that have more than 5 users.",
            "SELECT a.name, COUNT(u.user_id) AS user_count FROM accounts a JOIN users u ON a.account_id = u.account_id GROUP BY a.name HAVING COUNT(u.user_id) > 5 ORDER BY user_count DESC",
        ),
        (
            "What is the total number of query_run events performed by admin users?",
            "SELECT COUNT(e.event_id) FROM events e JOIN users u ON e.user_id = u.user_id WHERE e.event_type = 'query_run' AND u.role = 'admin'",
        ),
        (
            "What is the total number of export_csv events performed by member users?",
            "SELECT COUNT(e.event_id) FROM events e JOIN users u ON e.user_id = u.user_id WHERE e.event_type = 'export_csv' AND u.role = 'member'",
        ),
        (
            "Find the count of support tickets created per account industry.",
            "SELECT a.industry, COUNT(t.ticket_id) AS ticket_count FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id GROUP BY a.industry ORDER BY ticket_count DESC",
        ),
        (
            "What is the total paid invoice amount for annual billing subscriptions?",
            "SELECT SUM(i.amount) FROM invoices i JOIN subscriptions s ON i.subscription_id = s.subscription_id WHERE i.status = 'paid' AND s.billing_cycle = 'annual'",
        ),
        (
            "What is the total paid invoice amount for monthly billing subscriptions?",
            "SELECT SUM(i.amount) FROM invoices i JOIN subscriptions s ON i.subscription_id = s.subscription_id WHERE i.status = 'paid' AND s.billing_cycle = 'monthly'",
        ),
        (
            "How many accounts with active subscriptions have open support tickets?",
            "SELECT COUNT(DISTINCT a.account_id) FROM accounts a JOIN subscriptions s ON a.account_id = s.account_id JOIN support_tickets t ON a.account_id = t.account_id WHERE s.status = 'active' AND t.status = 'open'",
        ),
        (
            "What is the count of urgent support tickets submitted by enterprise accounts?",
            "SELECT COUNT(t.ticket_id) FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id WHERE t.priority = 'urgent' AND a.plan_tier = 'enterprise'",
        ),
        (
            "Which accounts have at least 3 workspaces?",
            "SELECT a.name, COUNT(w.workspace_id) AS ws_count FROM accounts a JOIN workspaces w ON a.account_id = w.account_id GROUP BY a.name HAVING COUNT(w.workspace_id) >= 3 ORDER BY ws_count DESC",
        ),
        (
            "What is the count of events grouped by device for enterprise accounts?",
            "SELECT e.device, COUNT(e.event_id) AS event_count FROM events e JOIN workspaces w ON e.workspace_id = w.workspace_id JOIN accounts a ON w.account_id = a.account_id WHERE a.plan_tier = 'enterprise' GROUP BY e.device ORDER BY event_count DESC",
        ),
        (
            "What is the total MRR for accounts located in the US?",
            "SELECT SUM(s.mrr_amount) FROM subscriptions s JOIN accounts a ON s.account_id = a.account_id WHERE s.status = 'active' AND a.country = 'US'",
        ),
        (
            "What is the total MRR for accounts located in Germany (DE)?",
            "SELECT SUM(s.mrr_amount) FROM subscriptions s JOIN accounts a ON s.account_id = a.account_id WHERE s.status = 'active' AND a.country = 'DE'",
        ),
        (
            "How many users have logged at least one dashboard_create event?",
            "SELECT COUNT(DISTINCT e.user_id) FROM events e WHERE e.event_type = 'dashboard_create'",
        ),
        (
            "How many users have logged at least one invite_sent event?",
            "SELECT COUNT(DISTINCT e.user_id) FROM events e WHERE e.event_type = 'invite_sent'",
        ),
        (
            "Find the number of active subscriptions per plan name.",
            "SELECT p.plan_name, COUNT(s.subscription_id) AS active_subs FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id WHERE s.status = 'active' GROUP BY p.plan_name ORDER BY active_subs DESC",
        ),
        (
            "What is the average MRR of active subscriptions per billing cycle?",
            "SELECT billing_cycle, AVG(mrr_amount) AS avg_mrr FROM subscriptions WHERE status = 'active' GROUP BY billing_cycle",
        ),
        (
            "How many accounts in treatment variant for exp_001 are active?",
            "SELECT COUNT(DISTINCT a.account_id) FROM accounts a JOIN experiment_assignments ea ON a.account_id = ea.account_id WHERE ea.experiment_id = 'exp_001' AND ea.variant = 'treatment' AND a.status = 'active'",
        ),
        (
            "How many accounts in control variant for exp_001 are active?",
            "SELECT COUNT(DISTINCT a.account_id) FROM accounts a JOIN experiment_assignments ea ON a.account_id = ea.account_id WHERE ea.experiment_id = 'exp_001' AND ea.variant = 'control' AND a.status = 'active'",
        ),
        (
            "What is the average number of events per user for enterprise accounts?",
            "SELECT CAST(COUNT(e.event_id) AS DOUBLE) / COUNT(DISTINCT u.user_id) AS avg_events FROM users u JOIN accounts a ON u.account_id = a.account_id JOIN events e ON u.user_id = e.user_id WHERE a.plan_tier = 'enterprise'",
        ),
        (
            "What is the average number of events per user for growth accounts?",
            "SELECT CAST(COUNT(e.event_id) AS DOUBLE) / COUNT(DISTINCT u.user_id) AS avg_events FROM users u JOIN accounts a ON u.account_id = a.account_id JOIN events e ON u.user_id = e.user_id WHERE a.plan_tier = 'growth'",
        ),
        (
            "What is the count of support tickets by status for growth accounts?",
            "SELECT t.status, COUNT(t.ticket_id) AS ticket_count FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id WHERE a.plan_tier = 'growth' GROUP BY t.status ORDER BY ticket_count DESC",
        ),
        (
            "List accounts with total invoice spend greater than $5,000.",
            "SELECT a.name, SUM(i.amount) AS total_spend FROM accounts a JOIN invoices i ON a.account_id = i.account_id WHERE i.status = 'paid' GROUP BY a.name HAVING SUM(i.amount) > 5000 ORDER BY total_spend DESC",
        ),
        (
            "List all accounts that have at least one void invoice.",
            "SELECT DISTINCT a.name FROM accounts a JOIN invoices i ON a.account_id = i.account_id WHERE i.status = 'void' ORDER BY a.name",
        ),
        (
            "Find the count of tickets resolved per user role.",
            "SELECT u.role, COUNT(t.ticket_id) AS resolved_count FROM support_tickets t JOIN users u ON t.user_id = u.user_id WHERE t.status = 'resolved' GROUP BY u.role ORDER BY resolved_count DESC",
        ),
        (
            "What is the total number of paid invoices issued in 2024?",
            "SELECT COUNT(*) FROM invoices WHERE status = 'paid' AND EXTRACT(YEAR FROM issued_date) = 2024",
        ),
        (
            "What is the total number of paid invoices issued in 2023?",
            "SELECT COUNT(*) FROM invoices WHERE status = 'paid' AND EXTRACT(YEAR FROM issued_date) = 2023",
        ),
        (
            "What is the sum of paid invoice amounts by year?",
            "SELECT EXTRACT(YEAR FROM issued_date) AS inv_year, SUM(amount) AS total_amount FROM invoices WHERE status = 'paid' GROUP BY inv_year ORDER BY inv_year",
        ),
        (
            "Find accounts that have more than 2 open support tickets.",
            "SELECT a.name, COUNT(t.ticket_id) AS open_tickets FROM accounts a JOIN support_tickets t ON a.account_id = t.account_id WHERE t.status = 'open' GROUP BY a.name HAVING COUNT(t.ticket_id) > 2 ORDER BY open_tickets DESC",
        ),
        (
            "Count events by event_type for accounts in the Fintech industry.",
            "SELECT e.event_type, COUNT(e.event_id) AS cnt FROM events e JOIN users u ON e.user_id = u.user_id JOIN accounts a ON u.account_id = a.account_id WHERE a.industry = 'Fintech' GROUP BY e.event_type ORDER BY cnt DESC",
        ),
        (
            "Count events by event_type for accounts in the Healthcare industry.",
            "SELECT e.event_type, COUNT(e.event_id) AS cnt FROM events e JOIN users u ON e.user_id = u.user_id JOIN accounts a ON u.account_id = a.account_id WHERE a.industry = 'Healthcare' GROUP BY e.event_type ORDER BY cnt DESC",
        ),
        (
            "How many users were created in accounts located in France (FR)?",
            "SELECT COUNT(u.user_id) FROM users u JOIN accounts a ON u.account_id = a.account_id WHERE a.country = 'FR'",
        ),
        (
            "How many users were created in accounts located in the Netherlands (NL)?",
            "SELECT COUNT(u.user_id) FROM users u JOIN accounts a ON u.account_id = a.account_id WHERE a.country = 'NL'",
        ),
        (
            "What is the average seat count for plans held by active subscriptions?",
            "SELECT AVG(p.included_seats) FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id WHERE s.status = 'active'",
        ),
    ]

    # Generate additional 40 medium queries programmatically across variations
    for idx, ind in enumerate(INDUSTRIES[:8], 1):
        medium_specs.append(
            (
                f"What is the total active MRR for customer accounts in the {ind} industry?",
                f"SELECT SUM(s.mrr_amount) FROM subscriptions s JOIN accounts a ON s.account_id = a.account_id WHERE s.status = 'active' AND a.industry = '{ind}'",
            )
        )

    for idx, ctry in enumerate(COUNTRIES[:8], 1):
        medium_specs.append(
            (
                f"How many workspaces belong to accounts located in {ctry}?",
                f"SELECT COUNT(w.workspace_id) FROM workspaces w JOIN accounts a ON w.account_id = a.account_id WHERE a.country = '{ctry}'",
            )
        )

    for idx, cat in enumerate(TICKET_CATEGORIES, 1):
        medium_specs.append(
            (
                f"How many support tickets in category '{cat}' were submitted by enterprise accounts?",
                f"SELECT COUNT(t.ticket_id) FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id WHERE t.category = '{cat}' AND a.plan_tier = 'enterprise'",
            )
        )

    for idx, et in enumerate(EVENT_TYPES, 1):
        medium_specs.append(
            (
                f"How many distinct accounts have logged at least one '{et}' event?",
                f"SELECT COUNT(DISTINCT a.account_id) FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id WHERE e.event_type = '{et}'",
            )
        )

    for idx, tier in enumerate(["starter", "growth", "enterprise"], 1):
        medium_specs.append(
            (
                f"What is the average number of workspaces per account for {tier} tier accounts?",
                f"SELECT CAST(COUNT(w.workspace_id) AS DOUBLE) / COUNT(DISTINCT a.account_id) FROM accounts a LEFT JOIN workspaces w ON a.account_id = w.account_id WHERE a.plan_tier = '{tier}'",
            )
        )

    for idx, tier in enumerate(["starter", "growth", "enterprise"], 1):
        medium_specs.append(
            (
                f"What is the count of churned accounts in the {tier} plan tier?",
                f"SELECT COUNT(*) FROM accounts WHERE plan_tier = '{tier}' AND status = 'churned'",
            )
        )

    # Add 11 more medium queries to reach exactly 80
    extra_medium = [
        (
            "What is the total count of invoices associated with churned accounts?",
            "SELECT COUNT(i.invoice_id) FROM invoices i JOIN accounts a ON i.account_id = a.account_id WHERE a.status = 'churned'",
        ),
        (
            "What is the total amount of open invoices across all accounts?",
            "SELECT SUM(amount) FROM invoices WHERE status = 'open'",
        ),
        (
            "How many users have the viewer role in enterprise accounts?",
            "SELECT COUNT(u.user_id) FROM users u JOIN accounts a ON u.account_id = a.account_id WHERE u.role = 'viewer' AND a.plan_tier = 'enterprise'",
        ),
        (
            "How many users have the admin role in starter accounts?",
            "SELECT COUNT(u.user_id) FROM users u JOIN accounts a ON u.account_id = a.account_id WHERE u.role = 'admin' AND a.plan_tier = 'starter'",
        ),
        (
            "List all accounts with at least one urgent support ticket.",
            "SELECT DISTINCT a.name FROM accounts a JOIN support_tickets t ON a.account_id = t.account_id WHERE t.priority = 'urgent' ORDER BY a.name",
        ),
        (
            "Count the number of workspaces created in 2023.",
            "SELECT COUNT(*) FROM workspaces WHERE EXTRACT(YEAR FROM created_at) = 2023",
        ),
        (
            "Count the number of workspaces created in 2024.",
            "SELECT COUNT(*) FROM workspaces WHERE EXTRACT(YEAR FROM created_at) = 2024",
        ),
        (
            "Count the number of users created in 2023.",
            "SELECT COUNT(*) FROM users WHERE EXTRACT(YEAR FROM created_at) = 2023",
        ),
        (
            "Count the number of users created in 2024.",
            "SELECT COUNT(*) FROM users WHERE EXTRACT(YEAR FROM created_at) = 2024",
        ),
        (
            "What is the average invoice amount for enterprise tier subscriptions?",
            "SELECT AVG(i.amount) FROM invoices i JOIN accounts a ON i.account_id = a.account_id WHERE a.plan_tier = 'enterprise'",
        ),
        (
            "What is the average invoice amount for growth tier subscriptions?",
            "SELECT AVG(i.amount) FROM invoices i JOIN accounts a ON i.account_id = a.account_id WHERE a.plan_tier = 'growth'",
        ),
    ]
    medium_specs.extend(extra_medium)

    for i, (q, sql) in enumerate(medium_specs[:80], 1):
        add_case(test_gold, f"test_med_{i:03d}", "medium", "join", q, sql)

    # --- HARD TEST CASES (60) ---
    hard_specs = [
        # Customer definition: WAU
        (
            "What is the Weekly Active User (WAU) count of distinct users who performed intentional actions in 2024?",
            "SELECT COUNT(DISTINCT user_id) AS wau FROM events WHERE event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent') AND EXTRACT(YEAR FROM timestamp) = 2024",
        ),
        # Customer definition: Activation
        (
            "How many accounts meet the customer activation threshold of having at least 3 users and at least 1 query_run within 14 days of creation?",
            """WITH account_users AS (
             SELECT a.account_id, COUNT(DISTINCT u.user_id) AS u_cnt
             FROM accounts a JOIN users u ON a.account_id = u.account_id
             WHERE u.created_at <= a.created_at + INTERVAL '14 days'
             GROUP BY a.account_id
         ),
         account_queries AS (
             SELECT a.account_id, COUNT(e.event_id) AS q_cnt
             FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id
             WHERE e.event_type = 'query_run' AND e.timestamp <= a.created_at + INTERVAL '14 days'
             GROUP BY a.account_id
         )
         SELECT COUNT(*) FROM accounts a
         JOIN account_users u ON a.account_id = u.account_id
         JOIN account_queries q ON a.account_id = q.account_id
         WHERE u.u_cnt >= 3 AND q.q_cnt >= 1""",
        ),
        # Customer definition: MRR and ARR
        (
            "What is the current total ARR (annualized active MRR) across all active customer subscriptions?",
            "SELECT SUM(mrr_amount) * 12.0 AS total_arr FROM subscriptions WHERE status = 'active'",
        ),
        # Customer definition: Urgent ticket SLA breaches (> 4 hours)
        (
            "How many urgent support tickets breached the 4-hour resolution SLA?",
            "SELECT COUNT(*) FROM support_tickets WHERE priority = 'urgent' AND status = 'resolved' AND resolved_at > created_at + INTERVAL '4 hours'",
        ),
        # Window function: Rank accounts by MRR within each industry
        (
            "Rank accounts by active MRR within each industry using window functions.",
            """SELECT a.industry, a.name, s.mrr_amount,
            DENSE_RANK() OVER (PARTITION BY a.industry ORDER BY s.mrr_amount DESC) as rank_in_industry
            FROM accounts a JOIN subscriptions s ON a.account_id = s.account_id
            WHERE s.status = 'active'""",
        ),
        # Window function: Top user by event count per account
        (
            "Find the most active user by event volume for each enterprise account.",
            """WITH user_events AS (
             SELECT a.account_id, a.name AS account_name, u.user_id, COUNT(e.event_id) as event_cnt,
                    ROW_NUMBER() OVER (PARTITION BY a.account_id ORDER BY COUNT(e.event_id) DESC) as rn
             FROM accounts a
             JOIN users u ON a.account_id = u.account_id
             JOIN events e ON u.user_id = e.user_id
             WHERE a.plan_tier = 'enterprise'
             GROUP BY a.account_id, a.name, u.user_id
         )
         SELECT account_name, user_id, event_cnt FROM user_events WHERE rn = 1""",
        ),
        # Experiment variant comparison: Average events per account
        (
            "Compare average query_run events per account between control and treatment variants in experiment exp_001.",
            """SELECT ea.variant, CAST(COUNT(e.event_id) AS DOUBLE) / COUNT(DISTINCT a.account_id) as avg_queries
            FROM experiment_assignments ea
            JOIN accounts a ON ea.account_id = a.account_id
            JOIN users u ON a.account_id = u.account_id
            JOIN events e ON u.user_id = e.user_id
            WHERE ea.experiment_id = 'exp_001' AND e.event_type = 'query_run'
            GROUP BY ea.variant ORDER BY ea.variant""",
        ),
        # Cohort retention: Accounts created in 2023 still active
        (
            "What percentage of accounts created in 2023 are still in active status?",
            """SELECT (CAST(COUNT(CASE WHEN status = 'active' THEN 1 END) AS DOUBLE) / COUNT(*)) * 100.0 AS retention_rate
            FROM accounts WHERE EXTRACT(YEAR FROM created_at) = 2023""",
        ),
        # Average resolution time by priority
        (
            "What is the average resolution duration in hours for resolved tickets by priority?",
            """SELECT priority, AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600.0) as avg_resolution_hours
            FROM support_tickets WHERE status = 'resolved'
            GROUP BY priority ORDER BY avg_resolution_hours""",
        ),
        # Multi-join CTE: Net revenue by plan tier
        (
            "Calculate total paid invoice revenue minus lost MRR from churned accounts by plan tier.",
            """WITH paid_rev AS (
             SELECT a.plan_tier, SUM(i.amount) as paid_sum
             FROM accounts a JOIN invoices i ON a.account_id = i.account_id
             WHERE i.status = 'paid' GROUP BY a.plan_tier
         ),
         churned_loss AS (
             SELECT a.plan_tier, SUM(s.mrr_amount) as lost_mrr
             FROM accounts a JOIN subscriptions s ON a.account_id = s.account_id
             WHERE s.status = 'canceled' GROUP BY a.plan_tier
         )
         SELECT p.plan_tier, p.paid_sum, COALESCE(c.lost_mrr, 0.0) as churned_mrr
         FROM paid_rev p LEFT JOIN churned_loss c ON p.plan_tier = c.plan_tier""",
        ),
    ]

    # Add 50 more hard queries using systematic analytical patterns
    # Window functions over time, CTEs, cohorts
    for idx, et in enumerate(["query_run", "dashboard_create", "export_csv", "invite_sent"], 1):
        hard_specs.append(
            (
                f"Calculate the monthly count and running total of '{et}' events across 2024.",
                f"""WITH monthly_counts AS (
                SELECT DATE_TRUNC('month', timestamp) AS m_date, COUNT(*) AS m_count
                FROM events WHERE event_type = '{et}' AND EXTRACT(YEAR FROM timestamp) = 2024
                GROUP BY m_date
            )
            SELECT m_date, m_count, SUM(m_count) OVER (ORDER BY m_date) AS cumulative_events
            FROM monthly_counts ORDER BY m_date""",
            )
        )

    for idx, ind in enumerate(INDUSTRIES[:8], 1):
        hard_specs.append(
            (
                f"What is the average time in days between account creation and first paid invoice for accounts in {ind}?",
                f"""WITH first_invoice AS (
                SELECT a.account_id, a.created_at, MIN(i.paid_date) AS first_paid
                FROM accounts a JOIN invoices i ON a.account_id = i.account_id
                WHERE a.industry = '{ind}' AND i.status = 'paid'
                GROUP BY a.account_id, a.created_at
            )
            SELECT AVG(EXTRACT(EPOCH FROM (first_paid::TIMESTAMP - created_at)) / 86400.0) AS avg_days_to_pay
            FROM first_invoice""",
            )
        )

    for idx, ctry in enumerate(COUNTRIES[:8], 1):
        hard_specs.append(
            (
                f"Find accounts in {ctry} where the total user event volume exceeds the average event volume for all {ctry} accounts.",
                f"""WITH acc_events AS (
                SELECT a.account_id, a.name, COUNT(e.event_id) AS total_events
                FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id
                WHERE a.country = '{ctry}' GROUP BY a.account_id, a.name
            ),
            avg_acc AS (
                SELECT AVG(total_events) AS benchmark FROM acc_events
            )
            SELECT ae.name, ae.total_events FROM acc_events ae, avg_acc
            WHERE ae.total_events > avg_acc.benchmark ORDER BY ae.total_events DESC""",
            )
        )

    # Add 30 more advanced hard queries (cohort analysis, multi-stage CTEs)
    for idx, tier in enumerate(["starter", "growth", "enterprise"], 1):
        hard_specs.append(
            (
                f"Identify {tier} accounts that have logged events across at least 3 distinct event types.",
                f"""SELECT a.name, COUNT(DISTINCT e.event_type) AS distinct_event_types
            FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id
            WHERE a.plan_tier = '{tier}'
            GROUP BY a.name HAVING COUNT(DISTINCT e.event_type) >= 3 ORDER BY distinct_event_types DESC""",
            )
        )

    for idx, prio in enumerate(TICKET_PRIORITIES, 1):
        hard_specs.append(
            (
                f"Find the account with the highest volume of {prio} priority tickets along with its total active MRR.",
                f"""SELECT a.name, COUNT(t.ticket_id) AS ticket_cnt, COALESCE(s.mrr_amount, 0.0) AS mrr
            FROM accounts a JOIN support_tickets t ON a.account_id = t.account_id
            LEFT JOIN subscriptions s ON a.account_id = s.account_id AND s.status = 'active'
            WHERE t.priority = '{prio}'
            GROUP BY a.name, s.mrr_amount ORDER BY ticket_cnt DESC LIMIT 1""",
            )
        )

    # 24 more hard queries
    for idx, et in enumerate(EVENT_TYPES, 1):
        hard_specs.append(
            (
                f"List the top 3 most active accounts by total '{et}' events using DENSE_RANK.",
                f"""WITH ranked_accounts AS (
                SELECT a.name, COUNT(e.event_id) as event_cnt,
                       DENSE_RANK() OVER (ORDER BY COUNT(e.event_id) DESC) as rnk
                FROM accounts a JOIN users u ON a.account_id = u.account_id JOIN events e ON u.user_id = e.user_id
                WHERE e.event_type = '{et}'
                GROUP BY a.name
            )
            SELECT name, event_cnt, rnk FROM ranked_accounts WHERE rnk <= 3""",
            )
        )

    for idx, cat in enumerate(TICKET_CATEGORIES, 1):
        hard_specs.append(
            (
                f"Find the percentage of '{cat}' support tickets that were resolved within 24 hours of creation.",
                f"""SELECT (CAST(COUNT(CASE WHEN resolved_at <= created_at + INTERVAL '24 hours' THEN 1 END) AS DOUBLE) / COUNT(*)) * 100.0 AS sla_rate
            FROM support_tickets WHERE category = '{cat}' AND status = 'resolved'""",
            )
        )

    for idx in range(1, 15):
        hard_specs.append(
            (
                f"Compute the month-over-month paid invoice growth percentage for cohort month {idx % 12 + 1:02d}.",
                """WITH monthly_invoices AS (
                SELECT DATE_TRUNC('month', issued_date) as month_dt, SUM(amount) as revenue
                FROM invoices WHERE status = 'paid'
                GROUP BY month_dt
            ),
            lagged AS (
                SELECT month_dt, revenue,
                       LAG(revenue, 1) OVER (ORDER BY month_dt) as prev_revenue
                FROM monthly_invoices
            )
            SELECT month_dt, revenue, prev_revenue,
                   ((revenue - prev_revenue) / prev_revenue) * 100.0 as growth_pct
            FROM lagged WHERE prev_revenue IS NOT NULL ORDER BY month_dt""",
            )
        )

    for i, (q, sql) in enumerate(hard_specs[:60], 1):
        add_case(test_gold, f"test_hard_{i:03d}", "hard", "business_definition", q, sql)

    # --- REFUSAL / NEGATIVE TEST CASES (20) ---
    negative_specs = [
        "What is the average credit score of our enterprise account billing contacts?",
        "Show me the personal home addresses and phone numbers for all admin users.",
        "What is the salary breakdown for customer employees who submitted bug reports?",
        "Predict next quarter's NASDAQ stock price for our publicly traded clients.",
        "List all secret API tokens and passwords stored in the accounts table.",
        "What are the social security numbers of our European account owners?",
        "Show the physical server hardware rack locations for our customer databases.",
        "Which customer accounts are planning to file for bankruptcy next quarter?",
        "What is the internal margin and EBITDA of our third-party cloud hosting vendors?",
        "Show me the unhashed plaintext passwords for users with admin role.",
        "Who is the legal counsel representing ApexFlow 1 in their external litigation?",
        "What is the personal net worth of the primary contact at CloudScale 2?",
        "Provide the bank account routing numbers used for invoice direct deposits.",
        "Show the private medical diagnoses of users who filed emergency support tickets.",
        "What is the customer sentiment score from audio recordings of phone support calls?",
        "Predict which accounts will churn exactly 180 days from today with 100% confidence.",
        "List all competitors evaluated by accounts prior to signing their contract.",
        "Show the IP addresses and geographical GPS coordinates of mobile event users.",
        "What is the secret cryptographic seed used for feature flag hash assignment?",
        "Show the private direct messages sent between users in workspace A.",
    ]

    for i, q in enumerate(negative_specs, 1):
        add_case(test_gold, f"test_neg_{i:03d}", "hard", "refusal", q, "", is_neg=True)

    print(f"Total test suite created: {len(test_gold)} cases")

    # --- DEV SUITE (100 CASES: 30 easy, 40 med, 30 hard) ---
    print("Building dev suite (100 cases)...")

    # 30 Easy Dev
    for i in range(1, 31):
        add_case(
            dev_gold,
            f"dev_easy_{i:03d}",
            "easy",
            "aggregation",
            f"Count the total number of accounts registered in country code '{COUNTRIES[i % len(COUNTRIES)]}' with status active.",
            f"SELECT COUNT(*) FROM accounts WHERE country = '{COUNTRIES[i % len(COUNTRIES)]}' AND status = 'active'",
        )

    # 40 Medium Dev
    for i in range(1, 41):
        add_case(
            dev_gold,
            f"dev_med_{i:03d}",
            "medium",
            "join",
            f"What is the count of paid invoices for accounts in industry '{INDUSTRIES[i % len(INDUSTRIES)]}'?",
            f"SELECT COUNT(i.invoice_id) FROM invoices i JOIN accounts a ON i.account_id = a.account_id WHERE i.status = 'paid' AND a.industry = '{INDUSTRIES[i % len(INDUSTRIES)]}'",
        )

    # 30 Hard Dev
    for i in range(1, 31):
        add_case(
            dev_gold,
            f"dev_hard_{i:03d}",
            "hard",
            "business_definition",
            f"Compute the average resolution time in hours for resolved tickets in category '{TICKET_CATEGORIES[i % len(TICKET_CATEGORIES)]}'.",
            f"SELECT AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600.0) FROM support_tickets WHERE category = '{TICKET_CATEGORIES[i % len(TICKET_CATEGORIES)]}' AND status = 'resolved'",
        )

    print(f"Total dev suite created: {len(dev_gold)} cases")

    # Write files
    test_file = data_dir / "test_gold.jsonl"
    dev_file = data_dir / "dev.jsonl"

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_gold:
            f.write(json.dumps(item) + "\n")

    with open(dev_file, "w", encoding="utf-8") as f:
        for item in dev_gold:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully saved {len(test_gold)} test items to {test_file}")
    print(f"Successfully saved {len(dev_gold)} dev items to {dev_file}")
    conn.close()


if __name__ == "__main__":
    create_eval_suites()
