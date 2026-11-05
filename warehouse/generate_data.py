"""Deterministic synthetic data generator for the MetricPulse B2B SaaS warehouse.

Generates reproducible rows across 11 tables adhering to warehouse/schema.sql.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import Any

import duckdb

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

INDUSTRIES = [
    "Enterprise Software",
    "Fintech",
    "Healthcare",
    "E-commerce",
    "Logistics",
    "Cybersecurity",
    "Education Technology",
    "Media & Publishing",
]

COUNTRIES = ["US", "GB", "DE", "FR", "CA", "NL", "AU", "JP"]

COMPANIES = [
    "ApexFlow",
    "CloudScale",
    "DataWeave",
    "HyperMetric",
    "Synthetix",
    "PulseLogic",
    "OmniRoute",
    "AeroSync",
    "VectorBase",
    "BrightCore",
    "NexusOps",
    "StratumAI",
    "KiteWave",
    "QuantumReach",
    "Crestline",
    "VortexLab",
    "EchoPoint",
    "PrismGuard",
    "TraceHub",
    "PeakGrid",
]

EVENT_TYPES = ["page_view", "dashboard_create", "query_run", "export_csv", "invite_sent"]
EVENT_WEIGHTS = [0.45, 0.15, 0.25, 0.10, 0.05]

TICKET_CATEGORIES = ["billing", "bug", "feature_request", "integration", "permissions"]
TICKET_PRIORITIES = ["low", "medium", "urgent"]
TICKET_PRIORITY_WEIGHTS = [0.55, 0.35, 0.10]


def init_schema(conn: duckdb.DuckDBPyConnection, schema_path: Path = SCHEMA_FILE) -> None:
    """Executes the DDL schema in the target DuckDB database."""
    ddl = schema_path.read_text(encoding="utf-8")
    conn.execute(ddl)


def generate_synthetic_warehouse(
    db_path: str = ":memory:",
    seed: int = 42,
    num_accounts: int = 100,
    num_events: int = 15000,
) -> duckdb.DuckDBPyConnection:
    """Deterministically populates target DuckDB database with B2B SaaS data."""
    rng = random.Random(seed)
    conn = duckdb.connect(db_path)
    init_schema(conn)

    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    total_days = (end_date - start_date).days

    # 1. Plans
    plans_data = [
        ("plan_starter", "Starter Tier", "starter", 49.0, 5),
        ("plan_growth", "Growth Tier", "growth", 199.0, 20),
        ("plan_enterprise", "Enterprise Tier", "enterprise", 999.0, 100),
    ]
    conn.executemany(
        "INSERT INTO plans (plan_id, plan_name, tier, base_mrr, included_seats) VALUES (?, ?, ?, ?, ?)",
        plans_data,
    )

    # 2. Feature Flags
    flags_data = [
        (
            "flag_ai_assistant",
            "ai_query_assistant",
            True,
            50,
            datetime.datetime(2023, 6, 1, 10, 0, 0),
        ),
        ("flag_csv_export", "csv_bulk_export", True, 100, datetime.datetime(2023, 1, 15, 8, 0, 0)),
        (
            "flag_sso_okta",
            "sso_okta_enforcement",
            True,
            25,
            datetime.datetime(2023, 9, 1, 12, 0, 0),
        ),
        (
            "flag_realtime_dash",
            "realtime_dashboards",
            False,
            0,
            datetime.datetime(2024, 2, 1, 9, 0, 0),
        ),
    ]
    conn.executemany(
        "INSERT INTO feature_flags (flag_id, flag_name, is_enabled, rollout_percentage, created_at) VALUES (?, ?, ?, ?, ?)",
        flags_data,
    )

    # 3. Experiments
    exp_data = [
        (
            "exp_001",
            "Onboarding Checklist V2",
            "ai_query_assistant",
            datetime.date(2024, 1, 15),
            datetime.date(2024, 6, 30),
            "activation_rate",
        ),
        (
            "exp_002",
            "Self-Serve Upgrade Modal",
            "csv_bulk_export",
            datetime.date(2024, 3, 1),
            datetime.date(2024, 9, 30),
            "nrr",
        ),
    ]
    conn.executemany(
        "INSERT INTO experiments (experiment_id, experiment_name, feature_flag, start_date, end_date, target_metric) VALUES (?, ?, ?, ?, ?, ?)",
        exp_data,
    )

    # 4. Accounts
    accounts_rows = []
    account_ids = []
    tiers = ["starter", "growth", "enterprise"]
    tier_weights = [0.55, 0.35, 0.10]

    for i in range(1, num_accounts + 1):
        acc_id = f"acc_{i:04d}"
        account_ids.append(acc_id)
        name_prefix = rng.choice(COMPANIES)
        name = f"{name_prefix} {i}"
        tier = rng.choices(tiers, weights=tier_weights, k=1)[0]
        acc_created_offset = rng.randint(0, int(total_days * 0.75))
        created_at = datetime.datetime.combine(
            start_date + datetime.timedelta(days=acc_created_offset),
            datetime.time(rng.randint(8, 18), rng.randint(0, 59), rng.randint(0, 59)),
        )
        status = rng.choices(["active", "churned", "suspended"], weights=[0.82, 0.14, 0.04], k=1)[0]
        industry = rng.choice(INDUSTRIES)
        country = rng.choice(COUNTRIES)
        accounts_rows.append((acc_id, name, tier, created_at, status, industry, country))

    conn.executemany(
        "INSERT INTO accounts (account_id, name, plan_tier, created_at, status, industry, country) VALUES (?, ?, ?, ?, ?, ?, ?)",
        accounts_rows,
    )

    # 5. Users and Workspaces
    users_rows = []
    user_ids = []
    user_account_map = {}
    workspaces_rows = []
    workspace_ids = []
    account_workspace_map = {}

    user_counter = 1
    ws_counter = 1

    for acc_id, _, tier, acc_created, status, _, _ in accounts_rows:
        num_users = (
            rng.randint(1, 4)
            if tier == "starter"
            else (rng.randint(3, 9) if tier == "growth" else rng.randint(8, 25))
        )
        acc_user_ids = []
        for u_idx in range(num_users):
            u_id = f"usr_{user_counter:05d}"
            user_ids.append(u_id)
            acc_user_ids.append(u_id)
            user_account_map[u_id] = acc_id
            role = (
                "admin"
                if u_idx == 0
                else rng.choices(["member", "viewer"], weights=[0.7, 0.3], k=1)[0]
            )
            email = f"user_{user_counter}@{acc_id}.internal"
            u_offset = rng.randint(0, 14)
            u_created = acc_created + datetime.timedelta(days=u_offset, hours=rng.randint(0, 12))
            is_active = status == "active" and rng.random() > 0.10
            users_rows.append((u_id, acc_id, role, email, u_created, is_active))
            user_counter += 1

        num_ws = 1 if tier == "starter" else rng.randint(1, 3)
        acc_ws_ids = []
        for w_idx in range(num_ws):
            ws_id = f"ws_{ws_counter:04d}"
            workspace_ids.append(ws_id)
            acc_ws_ids.append(ws_id)
            ws_name = f"Workspace {chr(65 + w_idx)}"
            ws_created = acc_created + datetime.timedelta(days=rng.randint(0, 2))
            workspaces_rows.append((ws_id, acc_id, ws_name, ws_created))
            ws_counter += 1
        account_workspace_map[acc_id] = acc_ws_ids

    conn.executemany(
        "INSERT INTO users (user_id, account_id, role, email, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?)",
        users_rows,
    )
    conn.executemany(
        "INSERT INTO workspaces (workspace_id, account_id, name, created_at) VALUES (?, ?, ?, ?)",
        workspaces_rows,
    )

    # 6. Subscriptions and Invoices
    subs_rows = []
    invoices_rows = []
    inv_counter = 1

    for acc_id, _, tier, acc_created, status, _, _ in accounts_rows:
        sub_id = f"sub_{acc_id}"
        plan_id = f"plan_{tier}"
        base_rate = 49.0 if tier == "starter" else (199.0 if tier == "growth" else 999.0)
        # MRR with small expansion variations
        mrr = base_rate * (1.0 + (0.1 * rng.randint(0, 5) if tier != "starter" else 0.0))
        billing_cycle = (
            "annual"
            if tier == "enterprise" or (tier == "growth" and rng.random() > 0.5)
            else "monthly"
        )

        sub_status = (
            "canceled"
            if status == "churned"
            else ("past_due" if status == "suspended" else "active")
        )
        start_d = acc_created.date()
        renewal_d = start_d + datetime.timedelta(days=365 if billing_cycle == "annual" else 30)
        canceled_at = (
            (start_d + datetime.timedelta(days=rng.randint(60, 240)))
            if sub_status == "canceled"
            else None
        )

        subs_rows.append(
            (
                sub_id,
                acc_id,
                plan_id,
                sub_status,
                billing_cycle,
                mrr,
                start_d,
                renewal_d,
                canceled_at,
            )
        )

        # Generate 1-4 invoices per subscription
        num_inv = rng.randint(1, 4)
        for i_idx in range(num_inv):
            inv_id = f"inv_{inv_counter:05d}"
            inv_counter += 1
            inv_date = start_d + datetime.timedelta(
                days=i_idx * (365 if billing_cycle == "annual" else 30)
            )
            if inv_date > end_date:
                break
            inv_amount = mrr * (12.0 if billing_cycle == "annual" else 1.0)
            inv_status = (
                "paid"
                if sub_status == "active" or i_idx < num_inv - 1
                else ("open" if sub_status == "past_due" else "void")
            )
            paid_d = (
                inv_date + datetime.timedelta(days=rng.randint(1, 5))
                if inv_status == "paid"
                else None
            )
            invoices_rows.append((inv_id, sub_id, acc_id, inv_amount, inv_status, inv_date, paid_d))

    conn.executemany(
        "INSERT INTO subscriptions (subscription_id, account_id, plan_id, status, billing_cycle, mrr_amount, start_date, renewal_date, canceled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        subs_rows,
    )
    conn.executemany(
        "INSERT INTO invoices (invoice_id, subscription_id, account_id, amount, status, issued_date, paid_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        invoices_rows,
    )

    # 7. Experiment Assignments
    assign_rows = []
    assign_counter = 1
    for acc_id in account_ids:
        for exp_id, _, _, exp_start, _, _ in exp_data:
            if rng.random() > 0.2:  # 80% inclusion
                var = rng.choice(["control", "treatment"])
                assigned_ts = datetime.datetime.combine(
                    exp_start, datetime.time(9, 0, 0)
                ) + datetime.timedelta(days=rng.randint(0, 10))
                assign_rows.append((f"asg_{assign_counter:05d}", exp_id, acc_id, var, assigned_ts))
                assign_counter += 1

    conn.executemany(
        "INSERT INTO experiment_assignments (assignment_id, experiment_id, account_id, variant, assigned_at) VALUES (?, ?, ?, ?, ?)",
        assign_rows,
    )

    # 8. Events
    event_rows = []
    for e_idx in range(1, num_events + 1):
        u_id = rng.choice(user_ids)
        acc_id = user_account_map[u_id]
        ws_list = account_workspace_map.get(acc_id, workspace_ids)
        ws_id = rng.choice(ws_list)
        etype = rng.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]

        # Diurnal distribution favoring working hours (8am - 6pm UTC)
        day_offset = rng.randint(0, total_days)
        hour = rng.choices(
            list(range(24)),
            weights=[1, 1, 1, 1, 1, 2, 4, 7, 10, 12, 12, 11, 10, 10, 9, 8, 6, 4, 3, 2, 1, 1, 1, 1],
            k=1,
        )[0]
        event_ts = datetime.datetime.combine(
            start_date + datetime.timedelta(days=day_offset),
            datetime.time(hour, rng.randint(0, 59), rng.randint(0, 59)),
        )

        props: dict[str, Any] = {"session_length_sec": rng.randint(15, 900)}
        if etype == "query_run":
            props["execution_ms"] = rng.randint(20, 2400)
            props["rows_scanned"] = rng.randint(100, 500000)
        elif etype == "dashboard_create":
            props["widget_count"] = rng.randint(1, 12)
        elif etype == "export_csv":
            props["row_count"] = rng.randint(50, 5000)

        device = rng.choices(["desktop", "mobile", "tablet"], weights=[0.85, 0.12, 0.03], k=1)[0]
        event_rows.append(
            (f"evt_{e_idx:07d}", u_id, ws_id, etype, event_ts, device, json.dumps(props))
        )

    conn.executemany(
        "INSERT INTO events (event_id, user_id, workspace_id, event_type, timestamp, device, properties) VALUES (?, ?, ?, ?, ?, ?, ?)",
        event_rows,
    )

    # 9. Support Tickets
    ticket_rows = []
    for t_idx in range(1, 301):
        u_id = rng.choice(user_ids)
        acc_id = user_account_map[u_id]
        priority = rng.choices(TICKET_PRIORITIES, weights=TICKET_PRIORITY_WEIGHTS, k=1)[0]
        cat = rng.choice(TICKET_CATEGORIES)
        t_day_offset = rng.randint(30, total_days)
        created_at = datetime.datetime.combine(
            start_date + datetime.timedelta(days=t_day_offset),
            datetime.time(rng.randint(7, 21), rng.randint(0, 59)),
        )
        status = rng.choices(["resolved", "in_progress", "open"], weights=[0.80, 0.12, 0.08], k=1)[
            0
        ]

        resolved_at = None
        if status == "resolved":
            # For urgent, resolve within 1-8 hours; for others, 4-48 hours
            dur_hours = rng.uniform(1.0, 6.0) if priority == "urgent" else rng.uniform(4.0, 48.0)
            resolved_at = created_at + datetime.timedelta(hours=dur_hours)

        ticket_rows.append(
            (f"tkt_{t_idx:04d}", acc_id, u_id, priority, cat, status, created_at, resolved_at)
        )

    conn.executemany(
        "INSERT INTO support_tickets (ticket_id, account_id, user_id, priority, category, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ticket_rows,
    )

    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic MetricPulse DuckDB warehouse")
    parser.add_argument(
        "--db-path", default="warehouse/analytics.duckdb", help="DuckDB database file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--accounts", type=int, default=100, help="Number of customer accounts")
    parser.add_argument("--events", type=int, default=15000, help="Number of product events")

    args = parser.parse_args()
    db_file = Path(args.db_path)
    if db_file.exists() and args.db_path != ":memory:":
        db_file.unlink()

    print(f"Generating synthetic DuckDB warehouse at {args.db_path} (seed={args.seed})...")
    conn = generate_synthetic_warehouse(
        db_path=args.db_path,
        seed=args.seed,
        num_accounts=args.accounts,
        num_events=args.events,
    )

    # Print summary of tables and row counts
    tables = conn.execute("SHOW TABLES").fetchall()
    print("\nWarehouse generation complete. Table row counts:")
    for (tbl,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  - {tbl:25s}: {count:,} rows")

    conn.close()


if __name__ == "__main__":
    main()
