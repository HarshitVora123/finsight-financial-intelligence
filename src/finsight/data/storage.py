"""SQLite persistence for reproducible demo analytics."""

from pathlib import Path
import sqlite3
from contextlib import closing
import pandas as pd


def write_bundle_sqlite(bundle, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(destination)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("DROP TABLE IF EXISTS customer_segments")
            for name in ("transactions", "debts", "repayments", "customers"):
                connection.execute(f"DROP TABLE IF EXISTS {name}")
            connection.executescript("""
                CREATE TABLE customers (
                    customer_id TEXT PRIMARY KEY, monthly_income REAL NOT NULL CHECK(monthly_income > 0),
                    employment_months INTEGER NOT NULL, credit_limit REAL NOT NULL CHECK(credit_limit > 0),
                    current_credit_balance REAL NOT NULL CHECK(current_credit_balance >= 0),
                    existing_debt REAL NOT NULL CHECK(existing_debt >= 0),
                    active_credit_accounts INTEGER NOT NULL, credit_history_months INTEGER NOT NULL,
                    savings_balance REAL NOT NULL, minimum_monthly_payment REAL NOT NULL,
                    monthly_budget REAL NOT NULL,
                    simulated_missed_payment_next_month INTEGER NOT NULL CHECK(simulated_missed_payment_next_month IN (0, 1)),
                    data_origin TEXT NOT NULL
                );
                CREATE TABLE transactions (
                    transaction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                    transaction_date TEXT NOT NULL, category TEXT NOT NULL,
                    amount REAL NOT NULL CHECK(amount >= 0), transaction_type TEXT NOT NULL CHECK(transaction_type IN ('credit','debit')),
                    merchant_category TEXT NOT NULL, FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
                CREATE TABLE debts (
                    debt_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                    principal REAL NOT NULL CHECK(principal >= 0), annual_rate REAL NOT NULL CHECK(annual_rate >= 0),
                    minimum_payment REAL NOT NULL CHECK(minimum_payment >= 0), due_day INTEGER NOT NULL CHECK(due_day BETWEEN 1 AND 31),
                    debt_type TEXT NOT NULL, FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
                CREATE TABLE repayments (
                    repayment_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, period TEXT NOT NULL,
                    due_date TEXT NOT NULL, amount_due REAL NOT NULL CHECK(amount_due >= 0),
                    amount_paid REAL NOT NULL CHECK(amount_paid >= 0), on_time INTEGER NOT NULL CHECK(on_time IN (0, 1)),
                    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
            """)
            for name in ("customers", "transactions", "debts", "repayments"):
                getattr(bundle, name).to_sql(name, connection, if_exists="append", index=False)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer_date ON transactions(customer_id, transaction_date)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_repay_customer_period ON repayments(customer_id, period)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_debts_customer ON debts(customer_id)")


def write_segments_sqlite(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    with closing(sqlite3.connect(destination)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("DROP TABLE IF EXISTS customer_segments")
            numeric_columns = [column for column in frame.columns
                               if column not in {"customer_id", "segment"}]
            definitions = []
            for column in numeric_columns:
                kind = "INTEGER" if column == "cluster_id" else "REAL"
                definitions.append(f'"{column}" {kind}')
            create_columns = ["customer_id TEXT PRIMARY KEY NOT NULL REFERENCES customers(customer_id)",
                              '"segment" TEXT NOT NULL', *definitions]
            connection.execute(f"CREATE TABLE customer_segments ({', '.join(create_columns)})")
            frame.to_sql("customer_segments", connection, if_exists="append", index=False)
