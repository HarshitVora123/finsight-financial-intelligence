-- BrightPath AI analytics, compatible with SQLite 3.25+.
-- Tables are created by scripts/run_pipeline.py. Every metric is descriptive.

-- 1) Customer-level financial summary: cash flow, debt-service ratio, and utilization.
WITH monthly_tx AS (
    SELECT customer_id,
           substr(transaction_date, 1, 7) AS month,
           SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END) AS income,
           SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END) AS outflow
    FROM transactions
    GROUP BY customer_id, substr(transaction_date, 1, 7)
), averaged AS (
    SELECT customer_id, AVG(income) AS avg_income, AVG(outflow) AS avg_outflow
    FROM monthly_tx GROUP BY customer_id
)
SELECT c.customer_id, c.monthly_income, c.existing_debt, c.minimum_monthly_payment,
       ROUND(1.0 * c.minimum_monthly_payment / NULLIF(c.monthly_income, 0), 4) AS debt_payment_to_income,
       ROUND(1.0 * c.current_credit_balance / NULLIF(c.credit_limit, 0), 4) AS utilization,
       ROUND(a.avg_income, 2) AS observed_income, ROUND(a.avg_outflow, 2) AS observed_outflow
FROM customers c LEFT JOIN averaged a USING (customer_id)
ORDER BY debt_payment_to_income DESC;

-- 2) Monthly transaction aggregation plus a windowed month-over-month change.
WITH monthly AS (
    SELECT customer_id, substr(transaction_date, 1, 7) AS month,
           SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END) AS income,
           SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END) AS expenses
    FROM transactions GROUP BY customer_id, substr(transaction_date, 1, 7)
), trend AS (
    SELECT customer_id, month, income, expenses,
           LAG(expenses) OVER (PARTITION BY customer_id ORDER BY month) AS previous_month_expenses
    FROM monthly
)
SELECT customer_id, month, ROUND(income, 2) AS income, ROUND(expenses, 2) AS expenses,
       ROUND((expenses - previous_month_expenses) / NULLIF(previous_month_expenses, 0), 4) AS spend_change
FROM trend ORDER BY customer_id, month;

-- 3) Repayment behaviour by month and customer segment.
SELECT s.segment, r.period, COUNT(*) AS payment_count,
       SUM(CASE WHEN r.on_time = 0 THEN 1 ELSE 0 END) AS late_or_missed,
       ROUND(1.0 * SUM(CASE WHEN r.on_time = 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS delinquency_rate
FROM repayments r JOIN customer_segments s USING (customer_id)
GROUP BY s.segment, r.period ORDER BY r.period, delinquency_rate DESC;

-- 4) Customer cohort analysis: missed-payment rate by first-observed repayment month.
WITH first_seen AS (
    SELECT customer_id, MIN(period) AS cohort_month FROM repayments GROUP BY customer_id
), cohort_activity AS (
    SELECT f.cohort_month, r.period,
           (CAST(substr(r.period, 1, 4) AS INTEGER) * 12 + CAST(substr(r.period, 6, 2) AS INTEGER)) -
           (CAST(substr(f.cohort_month, 1, 4) AS INTEGER) * 12 + CAST(substr(f.cohort_month, 6, 2) AS INTEGER)) AS period_number,
           r.on_time
    FROM repayments r JOIN first_seen f USING (customer_id)
)
SELECT cohort_month, period_number, COUNT(*) AS observed_payments,
       ROUND(AVG(1.0 - on_time), 4) AS missed_payment_rate
FROM cohort_activity GROUP BY cohort_month, period_number ORDER BY cohort_month, period_number;

-- 5) Cumulative amount paid per customer, using a window frame.
SELECT customer_id, period, amount_paid,
       SUM(amount_paid) OVER (PARTITION BY customer_id ORDER BY period
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_paid
FROM repayments ORDER BY customer_id, period;

-- 6) Rank customers by debt-payment burden within utilization bands.
WITH indicators AS (
    SELECT customer_id, existing_debt,
           minimum_monthly_payment / NULLIF(monthly_income, 0) AS debt_payment_to_income,
           current_credit_balance / NULLIF(credit_limit, 0) AS utilization,
           CASE WHEN current_credit_balance / NULLIF(credit_limit, 0) >= 0.70
                THEN 'higher utilization' ELSE 'lower utilization' END AS utilization_band
    FROM customers
)
SELECT customer_id, existing_debt, debt_payment_to_income, utilization, utilization_band,
       RANK() OVER (PARTITION BY utilization_band ORDER BY debt_payment_to_income DESC) AS burden_rank
FROM indicators ORDER BY utilization_band, burden_rank;

-- 7) Simulated outcome rate by discovered segment (not real repayment experience).
SELECT s.segment, COUNT(*) AS customers,
       AVG(1.0 * c.simulated_missed_payment_next_month) AS simulated_next_month_miss_rate
FROM customers c JOIN customer_segments s USING (customer_id)
GROUP BY s.segment ORDER BY simulated_next_month_miss_rate DESC;
