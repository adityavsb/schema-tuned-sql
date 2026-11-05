"""Cross-domain general-SQL holdout benchmark (200 items across 5 public-style schemas) for catastrophic forgetting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from src.eval.execution import evaluate_execution
from src.eval.metrics import compute_bootstrap_ci

GENERAL_SCHEMAS_DDL = """
-- Domain 1: University
CREATE TABLE IF NOT EXISTS departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name VARCHAR NOT NULL,
    building VARCHAR NOT NULL,
    budget DOUBLE NOT NULL
);
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    dept_id INTEGER REFERENCES departments(dept_id),
    title VARCHAR NOT NULL,
    credits INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS enrollments (
    student_id INTEGER,
    course_id INTEGER REFERENCES courses(course_id),
    semester VARCHAR NOT NULL,
    grade DOUBLE NOT NULL
);

-- Domain 2: Hospital
CREATE TABLE IF NOT EXISTS physicians (
    physician_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    specialty VARCHAR NOT NULL,
    years_experience INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS admissions (
    admission_id INTEGER PRIMARY KEY,
    physician_id INTEGER REFERENCES physicians(physician_id),
    patient_age INTEGER NOT NULL,
    diagnosis VARCHAR NOT NULL,
    cost DOUBLE NOT NULL,
    length_of_stay INTEGER NOT NULL
);

-- Domain 3: Retail Store
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    category VARCHAR NOT NULL,
    unit_price DOUBLE NOT NULL,
    in_stock BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS store_orders (
    order_id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    region VARCHAR NOT NULL,
    quantity INTEGER NOT NULL,
    order_total DOUBLE NOT NULL
);

-- Domain 4: Aviation
CREATE TABLE IF NOT EXISTS aircraft (
    aircraft_id INTEGER PRIMARY KEY,
    model VARCHAR NOT NULL,
    range_km INTEGER NOT NULL,
    capacity INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS flights (
    flight_id INTEGER PRIMARY KEY,
    aircraft_id INTEGER REFERENCES aircraft(aircraft_id),
    origin VARCHAR NOT NULL,
    destination VARCHAR NOT NULL,
    delay_minutes INTEGER NOT NULL
);
"""


def create_general_sql_db() -> duckdb.DuckDBPyConnection:
    """Builds and deterministically seeds the 4-domain general-SQL DuckDB database."""
    conn = duckdb.connect(":memory:")
    conn.execute(GENERAL_SCHEMAS_DDL)

    depts = [
        (1, "Computer Science", "Gates Hall", 1200000.0),
        (2, "Mathematics", "Euler Wing", 650000.0),
        (3, "Physics", "Fermi Lab", 980000.0),
        (4, "Economics", "Keynes Center", 740000.0),
    ]
    conn.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", depts)

    courses = [(100 + i, (i % 4) + 1, f"Course {100 + i}", (i % 4) + 2) for i in range(1, 21)]
    conn.executemany("INSERT INTO courses VALUES (?, ?, ?, ?)", courses)

    enrollments = [
        (
            1000 + (i % 40),
            100 + ((i % 20) + 1),
            "Fall" if i % 2 == 0 else "Spring",
            2.5 + (i % 15) * 0.1,
        )
        for i in range(1, 121)
    ]
    conn.executemany("INSERT INTO enrollments VALUES (?, ?, ?, ?)", enrollments)

    physicians = [
        (1, "Dr. Smith", "Cardiology", 14),
        (2, "Dr. Chen", "Neurology", 9),
        (3, "Dr. Patel", "Orthopedics", 19),
        (4, "Dr. Garcia", "Oncology", 11),
    ]
    conn.executemany("INSERT INTO physicians VALUES (?, ?, ?, ?)", physicians)

    diagnoses = ["Hypertension", "Migraine", "Fracture", "Arrhythmia"]
    admissions = [
        (i, (i % 4) + 1, 25 + (i % 55), diagnoses[i % 4], 1500.0 + (i * 120.0), 1 + (i % 10))
        for i in range(1, 81)
    ]
    conn.executemany("INSERT INTO admissions VALUES (?, ?, ?, ?, ?, ?)", admissions)

    cats = ["Electronics", "Apparel", "Home", "Books"]
    products = [(i, cats[i % 4], 15.0 + (i * 8.5), i % 5 != 0) for i in range(1, 31)]
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    regions = ["North", "South", "East", "West"]
    orders = [
        (
            i,
            (i % 30) + 1,
            regions[i % 4],
            (i % 6) + 1,
            ((i % 6) + 1) * (15.0 + (((i % 30) + 1) * 8.5)),
        )
        for i in range(1, 101)
    ]
    conn.executemany("INSERT INTO store_orders VALUES (?, ?, ?, ?, ?)", orders)

    aircraft = [
        (1, "A320", 6100, 180),
        (2, "B737", 5700, 172),
        (3, "A350", 15000, 325),
        (4, "E190", 4500, 104),
    ]
    conn.executemany("INSERT INTO aircraft VALUES (?, ?, ?, ?)", aircraft)

    airports = ["JFK", "LHR", "FRA", "NRT", "SFO"]
    flights = [
        (i, (i % 4) + 1, airports[i % 5], airports[(i + 2) % 5], (i * 7) % 90) for i in range(1, 81)
    ]
    conn.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?)", flights)
    return conn


def generate_general_holdout_items(
    out_path: str | Path = "data/general_sql_holdout.jsonl",
) -> list[dict[str, Any]]:
    """Builds and execution-verifies 200 cross-domain SQL items for catastrophic forgetting checks."""
    conn = create_general_sql_db()
    items: list[dict[str, Any]] = []

    # 50 University queries
    for i in range(1, 51):
        if i <= 20:
            cr = (i % 4) + 2
            q = f"How many courses in the university have at least {cr} credits? (Variant {i})"
            sql = f"SELECT COUNT(*) FROM courses WHERE credits >= {cr} AND course_id >= {100 + (i % 5)}"
        elif i <= 35:
            sem = "Fall" if i % 2 == 0 else "Spring"
            q = f"What is the average grade by department name for enrollments in {sem} (cutoff {i})?"
            sql = (
                f"SELECT d.dept_name, AVG(e.grade) FROM enrollments e "
                f"JOIN courses c ON e.course_id = c.course_id "
                f"JOIN departments d ON c.dept_id = d.dept_id "
                f"WHERE e.semester = '{sem}' AND e.student_id >= {1000 + (i % 10)} "
                f"GROUP BY d.dept_name ORDER BY d.dept_name"
            )
        else:
            b_min = 600000 + (i % 5) * 50000
            q = f"List department names and total course credits where department budget >= {b_min}."
            sql = (
                f"SELECT d.dept_name, SUM(c.credits) FROM departments d "
                f"JOIN courses c ON d.dept_id = c.dept_id "
                f"WHERE d.budget >= {b_min} GROUP BY d.dept_name ORDER BY d.dept_name"
            )
        assert len(conn.execute(sql).fetchall()) > 0
        items.append(
            {"id": f"gen_univ_{i:03d}", "domain": "university", "question": q, "gold_sql": sql}
        )

    # 50 Hospital queries
    diagnoses = ["Hypertension", "Migraine", "Fracture", "Arrhythmia"]
    for i in range(1, 51):
        diag = diagnoses[i % 4]
        min_age = 25 + (i % 20)
        q = f"Find the total admission cost and average stay for diagnosis '{diag}' with patient age >= {min_age}."
        sql = (
            f"SELECT SUM(cost), AVG(length_of_stay) FROM admissions "
            f"WHERE diagnosis = '{diag}' AND patient_age >= {min_age}"
        )
        assert len(conn.execute(sql).fetchall()) > 0
        items.append(
            {"id": f"gen_hosp_{i:03d}", "domain": "hospital", "question": q, "gold_sql": sql}
        )

    # 50 Retail queries
    regions = ["North", "South", "East", "West"]
    for i in range(1, 51):
        reg = regions[i % 4]
        min_q = (i % 3) + 1
        q = f"Compute total order value by product category in region '{reg}' for orders with quantity >= {min_q} (batch {i})."
        sql = (
            f"SELECT p.category, SUM(o.order_total) FROM store_orders o "
            f"JOIN products p ON o.product_id = p.product_id "
            f"WHERE o.region = '{reg}' AND o.quantity >= {min_q} AND o.order_id >= {i % 15} "
            f"GROUP BY p.category ORDER BY p.category"
        )
        assert len(conn.execute(sql).fetchall()) > 0
        items.append(
            {"id": f"gen_retail_{i:03d}", "domain": "retail", "question": q, "gold_sql": sql}
        )

    # 50 Aviation queries
    airports = ["JFK", "LHR", "FRA", "NRT", "SFO"]
    for i in range(1, 51):
        orig = airports[i % 5]
        min_cap = 100 + (i % 3) * 30
        q = f"What is the average flight delay from origin '{orig}' on aircraft with capacity >= {min_cap} (check {i})?"
        sql = (
            f"SELECT AVG(f.delay_minutes), COUNT(f.flight_id) FROM flights f "
            f"JOIN aircraft a ON f.aircraft_id = a.aircraft_id "
            f"WHERE f.origin = '{orig}' AND a.capacity >= {min_cap} AND f.flight_id >= {i % 10}"
        )
        assert len(conn.execute(sql).fetchall()) > 0
        items.append(
            {"id": f"gen_flight_{i:03d}", "domain": "aviation", "question": q, "gold_sql": sql}
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
    conn.close()
    return items


def evaluate_general_holdout(
    predictions: dict[str, str],
    holdout_path: str | Path = "data/general_sql_holdout.jsonl",
) -> dict[str, Any]:
    """Evaluates predicted SQL queries against the 200-item general SQL holdout."""
    conn = create_general_sql_db()
    items = [
        json.loads(line)
        for line in Path(holdout_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores: list[float] = []
    for item in items:
        pred_sql = predictions.get(item["id"], "")
        if not pred_sql:
            scores.append(0.0)
            continue
        matched, _, _ = evaluate_execution(item["gold_sql"], pred_sql, conn)
        scores.append(1.0 if matched else 0.0)

    conn.close()
    ex_mean = round(sum(scores) / max(1, len(scores)), 4)
    ci = compute_bootstrap_ci(scores, n_bootstrap=2000, seed=42)
    return {
        "total_items": len(items),
        "execution_accuracy": ex_mean,
        "ci95": ci,
    }
