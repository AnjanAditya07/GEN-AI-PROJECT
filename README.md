# GenAI-Powered BI Chatbot (CSV / Excel)

Ask plain-English business questions over your CSV/Excel data and get evidence-based answers,
optional SQL transparency, and auto-generated charts — powered by Groq + DuckDB.

## How it works

```
Upload (CSV or Excel, any # of sheets)
        │
        ▼
  Cleaning layer (utils/cleaning.py)
   - standardizes column names
   - fixes types (currency strings → numbers, Excel serial dates → dates)
   - removes duplicates / empty rows
   - ALWAYS outputs standardized CSV, one per sheet
        │
        ▼
  DuckDB in-memory engine (utils/query_engine.py)
   - loads each cleaned CSV as a table
   - enforces read-only SELECT queries only (blocks INSERT/UPDATE/DROP/etc.)
        │
        ▼
  Groq API (utils/llm.py)
   - intent-aware text-to-SQL: only data questions generate SQL
   - result interpretation: SQL results → concise natural-language analyst answer
   - chart specifications are built locally only when the user explicitly requests a chart
   - runs on Groq's LPU inference — very fast responses
        │
        ▼
  Streamlit chat UI (app.py)
   - chat history with follow-up context
   - shows SQL only when requested (or when a failed query must be diagnosed)
   - renders Plotly charts only on explicit chart/graph/visualization requests
   - lets you download the cleaned CSV
```

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Groq API key** as an environment variable:
   ```bash
   export GROQ_API_KEY="gsk_..."
   ```
   (On Windows CMD: `set GROQ_API_KEY=gsk_...`; permanent: `setx GROQ_API_KEY "gsk_..."`)

   Get a free key at https://console.groq.com/keys if you don't have one.

   The default model is `openai/gpt-oss-120b`. To use a different Groq-hosted
   model, edit `MODEL` at the top of `utils/llm.py`.

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   It will open at `http://localhost:8501`.

## Using it

1. Upload one or more CSV/Excel files in the sidebar.
   - Excel files with multiple sheets are automatically split into separate queryable tables. Multi-table joins require compatible keys and may need a clearer question.
2. Each upload is auto-cleaned; expand a table in the sidebar to see exactly what was changed (renamed columns, fixed types, removed duplicates, missing-value counts).
3. Download the cleaned CSV any time via the sidebar button.
4. Ask questions in the chat box, e.g.:
   - "What are total sales by region?"
   - "Show me the top 5 customers by order amount"
   - "Now break that down by month" (follow-up — it remembers recent context)
5. SQL is generated internally only when the question requires data analysis. The SQL panel is shown only when you explicitly ask to see the SQL (or when a query fails and the app needs to expose the failed query).

## Project structure

```
bi-chatbot/
├── app.py                  # Streamlit UI + orchestration
├── requirements.txt
├── utils/
│   ├── cleaning.py          # CSV/Excel ingestion + cleaning → standardized CSV
│   ├── query_engine.py      # DuckDB loading + safe query execution
│   └── llm.py                # Groq calls: text-to-SQL, answer+chart generation
└── data/
    ├── uploads/             # raw uploaded files
    ├── cleaned/             # standardized cleaned CSVs (one per sheet)
    └── logs/                 # JSON cleaning reports per file
```

## Security notes

- The SQL execution layer only permits `SELECT`/`WITH...SELECT` statements. Any
  `INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/COPY/PRAGMA` etc. keyword is rejected
  before execution — the LLM cannot modify or exfiltrate data outside of returning
  query results.
- Query results are capped (default 5,000 rows) to avoid huge payloads being sent
  back to the LLM for interpretation.
- Data stays local: DuckDB runs in-memory in this process. Only the schema
  description and (truncated) query results are sent to the Groq API — not
  full files.

## Known limitations / next steps to extend

- **Multi-table joins**: compatible tables can be queried together when the model can infer a safe join from the schema; complex relationships may need an explicit relationship map.
- **Very large files**: everything currently loads into memory. For files with
  millions of rows, consider DuckDB's native CSV scanning (`read_csv_auto`)
  instead of loading through pandas first.
- **Auth / multi-user**: this prototype has no login system or per-user data
  isolation — add that before deploying beyond a single-user/local setup.
- **Null handling**: the cleaner flags missing values but doesn't auto-fill them
  (intentional — silent imputation can distort BI answers). Add a UI control if
  you want optional auto-fill strategies (mean/median/forward-fill).


## Analytical improvements

The analytical layer now:
- detects KPI, ranking, comparison, trend, diagnostic, driver/factor, forecasting and recommendation intent;
- detects driver wording such as why, factors, drivers, causes, contributing, affecting, underperforming, low sales and declining revenue;
- analyzes relevant business dimensions independently instead of treating the lowest states as separate factors;
- compares segment performance with overall sales/metrics and exposes segment value, sales, overall sales, difference and percentage difference;
- scores underperformance using both percentage gap and segment sales scale, with a minimum record threshold to protect tiny segments;
- uses SUM for business impact and AVG only for average-performance questions;
- supports YoY, MoM, quarterly/QoQ and before/after time comparisons;
- uses prior question/result context for follow-up questions such as “Why is it low?”;
- protects against causal wording by describing observational relationships as “associated with”;
- generates charts only when explicitly requested; uses line for trends, bar for rankings/factors, scatter for relationships, and pie/treemap where appropriate;
- displays a data-quality report covering rows removed, duplicates, missing values, standardized columns and type/format conversions.
