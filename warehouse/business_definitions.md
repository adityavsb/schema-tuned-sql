# Customer Business Definitions & Semantic Metric Rules

**Fictional Enterprise:** MetricPulse Systems (B2B SaaS Workspace & Analytics)  
**Dialect:** DuckDB  
**Status:** Authoritative ground-truth definitions for Text-to-SQL generation.

---

## 1. Product Usage & User Activity

### Weekly Active User (WAU)
- **Definition:** A distinct user who has triggered at least one intentional product interaction within a rolling 7-day window.
- **Inclusions:** `event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent')`.
- **Exclusions:** Passive background requests and `page_view` events are explicitly excluded from WAU calculations.
- **Canonical SQL Logic:**
  ```sql
  SELECT COUNT(DISTINCT user_id) AS wau
  FROM events
  WHERE event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent')
    AND timestamp >= (CURRENT_DATE - INTERVAL '7 days');
  ```

### Daily Active User (DAU)
- **Definition:** A distinct user who has logged at least one active product interaction (`event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent')`) on a given calendar day (`timestamp::DATE`).

---

## 2. Customer Activation & Health

### Activated Account
- **Definition:** An account is formally considered "Activated" if and only if both conditions are met within the first 14 days of the account's creation date (`accounts.created_at`):
  1. At least 3 distinct users have registered under the account (`users.account_id`).
  2. At least 1 `query_run` event has been logged by any user in that account.
- **Canonical SQL Logic:**
  ```sql
  WITH account_user_counts AS (
      SELECT a.account_id,
             COUNT(DISTINCT u.user_id) AS total_users
      FROM accounts a
      JOIN users u ON a.account_id = u.account_id
      WHERE u.created_at <= a.created_at + INTERVAL '14 days'
      GROUP BY a.account_id
  ),
  account_query_events AS (
      SELECT a.account_id,
             COUNT(e.event_id) AS total_queries
      FROM accounts a
      JOIN users u ON a.account_id = u.account_id
      JOIN events e ON u.user_id = e.user_id
      WHERE e.event_type = 'query_run'
        AND e.timestamp <= a.created_at + INTERVAL '14 days'
      GROUP BY a.account_id
  )
  SELECT a.account_id,
         CASE WHEN COALESCE(u.total_users, 0) >= 3 AND COALESCE(q.total_queries, 0) >= 1
              THEN TRUE ELSE FALSE END AS is_activated
  FROM accounts a
  LEFT JOIN account_user_counts u ON a.account_id = u.account_id
  LEFT JOIN account_query_events q ON a.account_id = q.account_id;
  ```

---

## 3. Financial & Revenue Metrics

### Monthly Recurring Revenue (MRR)
- **Definition:** The active monthly contracted subscription revenue.
- **Rules:**
  - Must filter on `subscriptions.status = 'active'`.
  - Exclude canceled, trial, or past due subscriptions.
  - Exclude one-off invoices or setup credits.
- **Canonical SQL Logic:**
  ```sql
  SELECT SUM(mrr_amount) AS total_mrr
  FROM subscriptions
  WHERE status = 'active';
  ```

### Annual Recurring Revenue (ARR)
- **Definition:** Annualized recurring run-rate:
  ```sql
  SELECT SUM(mrr_amount) * 12 AS total_arr
  FROM subscriptions
  WHERE status = 'active';
  ```

### Net Revenue Retention (NRR) - Cohort Basis
- **Definition:** For a customer cohort starting at base month $T_0$, the ratio of their active MRR at $T_0 + 12 \text{ months}$ (including upgrades, downgrades, and churn) to their initial starting MRR at $T_0$.
- **Formula:** $\text{NRR} = \frac{\text{Ending Cohort MRR}}{\text{Starting Cohort MRR}} \times 100\%$

### Churned Account
- **Definition:** An account whose status is `'churned'` OR whose subscription status is `'canceled'` and has not renewed.

---

## 4. Customer Support SLAs

### Urgent Ticket SLA Compliance
- **Definition:** Urgent priority support tickets (`support_tickets.priority = 'urgent'`) must be resolved (`status = 'resolved'`) within 4 hours (240 minutes) of creation (`resolved_at - created_at <= INTERVAL '4 hours'`).

---

## 5. Experimentation & A/B Testing

### Variant Conversion / Event Frequency
- **Definition:** Comparing per-account action event frequency between `'control'` and `'treatment'` variants within an experiment's active date range (`assigned_at >= experiments.start_date` and `assigned_at <= experiments.end_date`).
