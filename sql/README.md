# SQL analytics

The queries in `analytics.sql` run on SQLite 3.25 or newer after `python scripts/run_pipeline.py` builds `data/brightpath.sqlite`.

- Query 1 summarizes affordability, payment burden and utilization per customer.
- Query 2 aggregates transaction cash flow monthly and uses `LAG` to compare spending with the prior month.
- Query 3 compares observed synthetic repayment misses across the data-derived customer segments.
- Query 4 follows first-observed repayment cohorts over time.
- Query 5 calculates cumulative payments with a window frame.
- Query 6 ranks repayment burden within utilization bands.
- Query 7 summarizes the generated next-month outcome by segment; it is explicitly simulated.

For a quick local check, open the database with Python's built-in sqlite3 module or use `sqlite3 data/brightpath.sqlite < sql/analytics.sql`. The file contains multiple result-producing queries, so a SQL client is convenient for inspecting each result independently.
