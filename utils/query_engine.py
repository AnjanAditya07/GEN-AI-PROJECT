"""
Query engine: loads cleaned CSVs into an in-memory DuckDB instance
and safely executes LLM-generated SQL against them.
"""

import duckdb
import pandas as pd
import re


READ_ONLY_VIOLATIONS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|CALL|INSTALL|LOAD)\b",
    re.IGNORECASE,
)

MAX_ROWS_RETURNED = 5000


class QueryEngine:
    def __init__(self):
        self.con = duckdb.connect(database=":memory:")
        self.tables = {}  # table_name -> {"columns": [...], "dtypes": {...}, "row_count": int}

    def register_table(self, table_name: str, csv_path: str):
        """Load a cleaned CSV into DuckDB as a queryable table."""
        df = pd.read_csv(csv_path)
        # Restore date/time columns that were serialized to CSV as text.
        # This is important for month/quarter/YoY/MoM analysis in DuckDB.
        for col in list(df.columns):
            name = str(col).casefold()
            if any(token in name for token in ("date", "datetime", "timestamp", "time")):
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if len(df) and parsed.notna().mean() >= 0.90:
                    df[col] = parsed
        # DuckDB can register a pandas df directly
        self.con.register(f"{table_name}_view", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM {table_name}_view')

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.tables[table_name] = {
            "columns": list(df.columns),
            "dtypes": dtypes,
            "row_count": len(df),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
        return self.tables[table_name]

    def get_schema_description(self) -> str:
        """Produces a compact schema description for the LLM.

        Sample rows are intentionally omitted from the default prompt because they
        consume TPM budget and are not required for most SQL generation.
        """
        lines = []
        for table_name, info in self.tables.items():
            lines.append(f"Table: {table_name} ({info['row_count']} rows)")
            for col in info["columns"]:
                lines.append(f"- {col} ({info['dtypes'][col]})")
        return "\n".join(lines)

    def validate_sql(self, sql: str):
        """Reject anything that isn't a read-only SELECT."""
        stripped = sql.strip()
        # Permit one optional trailing semicolon, never multiple statements.
        if ";" in stripped[:-1] or stripped.count(";") > 1:
            raise ValueError("Multiple SQL statements are not allowed.")
        stripped = stripped.rstrip(";").strip()
        if not re.match(r"^\s*(WITH|SELECT)\b", stripped, re.IGNORECASE):
            raise ValueError("Only SELECT (or WITH...SELECT) queries are allowed.")
        if re.search(r"--|/\*|\*/", stripped):
            raise ValueError("SQL comments are not allowed.")
        if READ_ONLY_VIOLATIONS.search(stripped):
            raise ValueError("Query contains a disallowed keyword. Only read-only SELECT queries are permitted.")
        return stripped

    def run_query(self, sql: str) -> pd.DataFrame:
        safe_sql = self.validate_sql(sql)
        result = self.con.execute(safe_sql).fetchdf()
        if len(result) > MAX_ROWS_RETURNED:
            result = result.head(MAX_ROWS_RETURNED)
        return result


    def get_relevant_schema_description(self, question: str, intent: dict | None = None, max_chars: int = 1900) -> str:
        """Return a compact, question-aware schema while retaining key business columns.

        Analytical questions get the dimensions needed for that analysis; simple questions
        get only likely metric/date/dimension columns. This reduces prompt size without
        hiding columns the analyst is likely to need.
        """
        q = (question or "").casefold()
        intent = intent or {}
        dimension_words = {
            "region": ("region",), "state": ("state",), "retailer": ("retailer",),
            "product": ("product",), "category": ("category",),
            "channel": ("channel", "sales_method", "sales method"),
            "time": ("date", "time", "month", "quarter", "year"),
            "customer": ("customer", "segment"),
        }
        priority = set()
        all_columns = []
        for table_name, info in self.tables.items():
            for col in info["columns"]:
                all_columns.append((table_name, col, info["dtypes"][col]))

        # Numeric/date columns are always useful to identify the metric and time axis.
        for table_name, col, dtype in all_columns:
            lc = col.casefold()
            if any(k in lc for k in ("sales", "revenue", "amount", "price", "profit", "margin", "order", "qty", "quantity", "date", "time", "year", "month", "quarter")):
                priority.add((table_name, col))

        # Explicit dimensions named in the question. Driver/factor questions need all
        # available business dimensions so the LLM can discover where the issue is concentrated.
        broad = intent.get("driver_analysis") or intent.get("recommendation_context") or intent.get("forecasting")
        for table_name, col, dtype in all_columns:
            lc = col.casefold()
            matched = any(any(alias in lc for alias in aliases) and (broad or any(alias in q for alias in aliases))
                          for aliases in dimension_words.values())
            if matched:
                priority.add((table_name, col))

        lines = []
        for table_name, info in self.tables.items():
            cols = []
            for col in info["columns"]:
                if (table_name, col) in priority:
                    cols.append(f"- {col} ({info['dtypes'][col]})")
            # Never omit every column from a table; include a compact fallback.
            if not cols:
                cols = [f"- {c} ({info['dtypes'][c]})" for c in info["columns"][:12]]
            lines.append(f"Table: {table_name} ({info['row_count']} rows)\n" + "\n".join(cols))
        text = "\n".join(lines)
        return text[:max_chars]

    def list_tables(self):
        return list(self.tables.keys())
