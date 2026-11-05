-- ==============================================================================
-- Schema: Fictional B2B SaaS Collaboration & Analytics Platform
-- Dialect: DuckDB
-- ==============================================================================

-- 1. Plans catalogue
CREATE TABLE IF NOT EXISTS plans (
    plan_id VARCHAR PRIMARY KEY,
    plan_name VARCHAR NOT NULL,
    tier VARCHAR NOT NULL, -- 'starter', 'growth', 'enterprise'
    base_mrr DOUBLE NOT NULL,
    included_seats INTEGER NOT NULL
);

-- 2. Customer accounts (Organizations)
CREATE TABLE IF NOT EXISTS accounts (
    account_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    plan_tier VARCHAR NOT NULL, -- 'starter', 'growth', 'enterprise'
    created_at TIMESTAMP NOT NULL,
    status VARCHAR NOT NULL, -- 'active', 'churned', 'suspended'
    industry VARCHAR NOT NULL,
    country VARCHAR NOT NULL
);

-- 3. Users belonging to customer accounts
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    role VARCHAR NOT NULL, -- 'admin', 'member', 'viewer'
    email VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- 4. Workspaces created within accounts
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    name VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- 5. Subscriptions (Account billing relationships)
CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    plan_id VARCHAR NOT NULL REFERENCES plans(plan_id),
    status VARCHAR NOT NULL, -- 'active', 'past_due', 'canceled'
    billing_cycle VARCHAR NOT NULL, -- 'monthly', 'annual'
    mrr_amount DOUBLE NOT NULL,
    start_date DATE NOT NULL,
    renewal_date DATE NOT NULL,
    canceled_at DATE
);

-- 6. Invoices generated from subscriptions
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id VARCHAR PRIMARY KEY,
    subscription_id VARCHAR NOT NULL REFERENCES subscriptions(subscription_id),
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    amount DOUBLE NOT NULL,
    status VARCHAR NOT NULL, -- 'paid', 'void', 'open'
    issued_date DATE NOT NULL,
    paid_date DATE
);

-- 7. Feature flags
CREATE TABLE IF NOT EXISTS feature_flags (
    flag_id VARCHAR PRIMARY KEY,
    flag_name VARCHAR NOT NULL UNIQUE,
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_percentage INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL
);

-- 8. Product usage event log
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL REFERENCES users(user_id),
    workspace_id VARCHAR NOT NULL REFERENCES workspaces(workspace_id),
    event_type VARCHAR NOT NULL, -- 'page_view', 'dashboard_create', 'query_run', 'export_csv', 'invite_sent'
    timestamp TIMESTAMP NOT NULL,
    device VARCHAR DEFAULT 'desktop',
    properties JSON
);

-- 9. A/B testing experiments
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR NOT NULL,
    feature_flag VARCHAR NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    target_metric VARCHAR NOT NULL -- 'activation_rate', 'query_frequency', 'nrr'
);

-- 10. Experiment assignments
CREATE TABLE IF NOT EXISTS experiment_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    experiment_id VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    variant VARCHAR NOT NULL, -- 'control', 'treatment'
    assigned_at TIMESTAMP NOT NULL
);

-- 11. Customer support tickets
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    user_id VARCHAR NOT NULL REFERENCES users(user_id),
    priority VARCHAR NOT NULL, -- 'low', 'medium', 'urgent'
    category VARCHAR NOT NULL, -- 'billing', 'bug', 'feature_request'
    status VARCHAR NOT NULL, -- 'open', 'in_progress', 'resolved'
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP
);

-- Performance indexes for analytical queries
CREATE INDEX IF NOT EXISTS idx_users_account ON users(account_id);
CREATE INDEX IF NOT EXISTS idx_workspaces_account ON workspaces(account_id);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_subs_account_status ON subscriptions(account_id, status);
CREATE INDEX IF NOT EXISTS idx_invoices_account ON invoices(account_id, status);
CREATE INDEX IF NOT EXISTS idx_exp_assign_account ON experiment_assignments(account_id, experiment_id);
