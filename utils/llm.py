"""LLM layer for an evidence-first BI data analyst.

Design goals:
- SQL is generated only for data questions.
- SQL generation uses plain text, not fragile structured JSON output.
- Final answers use plain text; chart specifications are built locally.
- Prompts are deliberately compact to avoid Groq TPM/JSON validation failures.
"""
import os
import re
import json
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Set the environment variable and restart Streamlit.")
    return Groq(api_key=key)


def _call_groq(system_prompt: str, user_content: str, *, max_completion_tokens: int = 700,
               temperature: float = 0.0) -> str:
    """Make a compact Groq request.

    The completion budget is intentionally bounded: Groq's TPM limit counts both
    prompt and completion tokens. A small, purpose-built call is safer than an
    unbounded request that can cross the account limit.
    """
    response = _get_client().chat.completions.create(
        model=MODEL,
        temperature=temperature,
        reasoning_effort="low",
        max_completion_tokens=max_completion_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def _compact_history(chat_history: list[dict] | None, max_turns: int = 2, max_chars: int = 900) -> str:
    if not chat_history:
        return ""
    chunks = []
    for h in chat_history[-max_turns:]:
        role = h.get("role", "")
        content = str(h.get("content", ""))[:260]
        if role == "assistant":
            q = str(h.get("question", ""))[:220]
            sql = str(h.get("sql", ""))[:220]
            prev = h.get("table")
            sample = ""
            if prev is not None:
                try:
                    sample = json.dumps(prev.head(2).to_dict(orient="records"), default=str)[:280]
                except Exception:
                    pass
            chunks.append(f"assistant:{content}\nprevious_question:{q}\nprevious_sql:{sql}\nprevious_result:{sample}")
        else:
            chunks.append(f"user:{content}")
    return "\n".join(chunks)[-max_chars:]


def generate_normal_response(question: str, chat_history: list[dict] = None) -> str:
    history = _compact_history(chat_history, max_turns=2, max_chars=700)
    system_prompt = """You are a concise business-intelligence assistant. Answer the user's question naturally.
Do not generate SQL, JSON, code, or database queries. If the user asks for a result from their uploaded data,
explain that you can analyze the uploaded dataset. Do not invent dataset facts."""
    if history:
        system_prompt += f"\nContext for follow-ups:\n{history}"
    return _call_groq(system_prompt, question, max_completion_tokens=400, temperature=0.2).strip()


def _extract_sql(text: str) -> str | None:
    """Extract one SQL statement from a plain-text model response."""
    if not text:
        return None
    text = text.strip()
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced[0].strip()
    # Remove common prefixes while preserving WITH/SELECT.
    match = re.search(r"\b(WITH|SELECT)\b", text, flags=re.I)
    if not match:
        return None
    text = text[match.start():].strip()
    # If the model appended prose after a semicolon, keep the first statement.
    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    return text


def generate_sql(question: str, schema_description: str, chat_history: list[dict] = None,
                 intent: dict | None = None) -> dict:
    """Generate one concise read-only DuckDB query as plain SQL."""
    intent = intent or {}
    schema_short = (schema_description or "")[:2000]
    history = _compact_history(chat_history, max_turns=2, max_chars=900)
    system_prompt = f"""You are a senior BI analyst writing ONE DuckDB query for the user's question.
Return ONLY SQL. No JSON, markdown, explanation, comments, or multiple statements.
SCHEMA:
{schema_short}
INTENT:{json.dumps(intent, separators=(',', ':'))}
REQUESTED_CHART_TYPE:{intent.get("requested_chart_type") or "auto"}
{('FOLLOW-UP CONTEXT:' + history) if history else ''}

Rules:
- Read-only SELECT/WITH only; use only schema columns.
- SUM = total sales/revenue/business impact; AVG = average performance; COUNT(*) = row/order volume when one row represents an order.
- Ranking must use the user's requested N and metric; do not invent a fixed LIMIT 5.
- For underperformance, benchmark a segment against the average SUM(metric) of segments in the SAME dimension, not against the company total. Keep company total separately for contribution/share when useful.
- For benchmark output use segment_sales, benchmark_sales, difference, pct_difference when relevant.
- Business impact should combine scale and shortfall; a practical score is GREATEST(0,-pct_difference)*segment_sales. Apply a minimum-volume screen for business-problem ranking when feasible (at least 10 rows and about 1% of table rows).
- Driver/factor analysis should use relevant dimensions actually present: Region, State, Retailer, Product, Category, Sales Method/Channel, and Time. Do not treat the five lowest states as five factors.
- Decline questions: establish the time trend first; then inspect where the decline is concentrated across relevant dimensions.
- Respect explicit YoY, MoM, QoQ, monthly, quarterly, before/after and date-range wording. Do not compare periods absent from the data.
- CHART REQUESTS: if the user explicitly asks for a chart/graph/visual, the query MUST return chartable rows and the chart type in the intent is authoritative.
- If REQUESTED_CHART_TYPE is line or area and the question is a trend/time question, group the relevant sales/revenue metric by the best available date/time field at a sensible grain (month for monthly/overall trend unless the user specifies otherwise), ordered chronologically. Return a time column plus a numeric sales/revenue value; never return only one grand-total row.
- If REQUESTED_CHART_TYPE is bar, return the requested dimension/ranking and numeric metric. Preserve the requested Top/Bottom N and sort direction.
- If REQUESTED_CHART_TYPE is pie or treemap, return one categorical segment column and one positive numeric contribution/value column; do not return benchmark-only columns unless they are the requested measure.
- If REQUESTED_CHART_TYPE is scatter, return two meaningful numeric measures (for example sales and orders) and include a useful dimension when available.
- If REQUESTED_CHART_TYPE is heatmap, return two categorical dimensions plus one numeric measure, suitable for a pivot/cross-tab.
- If REQUESTED_CHART_TYPE is histogram or box, return the underlying numeric measure (and a grouping dimension for box plots when relevant).
- Never generate an aggregate-only one-row result when the requested visualization needs multiple points.
- For recommendations, retrieve the evidence needed to support specific actions; never invent causes.
- For follow-ups like 'why is it low?', use previous context to resolve the referenced segment.
- If the schema cannot answer the question, return SELECT NULL AS result WHERE FALSE.
"""
    text = _call_groq(system_prompt, question, max_completion_tokens=700)
    sql = _extract_sql(text)
    return {"sql": sql, "explanation": "Generated a read-only analytical query." if sql else "The question could not be translated into a valid data query."}


def repair_sql(question: str, schema_description: str, sql: str, error_message: str,
               chat_history: list[dict] = None, intent: dict | None = None) -> dict:
    intent = intent or {}
    system_prompt = f"""Repair ONE failed DuckDB query. Return ONLY the repaired SQL, with no JSON, markdown, explanation, or comments.
SCHEMA:
{(schema_description or '')[:1900]}
INTENT:{json.dumps(intent, separators=(',', ':'))}
FAILED SQL:{sql[:2800]}
ERROR:{error_message[:800]}
Rules: preserve the user's metric and intent; use only SELECT/WITH; fix aliases, GROUP BY, date handling,
scalar benchmark references, and column names. In grouped SELECTs, use ANY_VALUE() for a scalar benchmark when needed.
"""
    text = _call_groq(system_prompt, "Repair the query.", max_completion_tokens=650)
    repaired = _extract_sql(text)
    return {"sql": repaired, "explanation": "Repaired the failed query." if repaired else "Could not repair the query."}


def _fallback_answer(question: str, df) -> str:
    if df is None or getattr(df, "empty", False):
        return "The query returned no rows, so there is no supporting evidence for a business conclusion."
    cols = list(df.columns)
    preview = df.head(3)
    lines = [f"The analysis returned {len(df):,} result row(s)."]
    for _, row in preview.iterrows():
        parts = [f"{c}: {row[c]}" for c in cols[:5]]
        lines.append("• " + "; ".join(parts))
    return "\n".join(lines)


def generate_answer_and_chart(question: str, sql: str, result_df_json: str, columns: list[str],
                              intent: dict | None = None, chart_requested: bool = False) -> dict:
    """Generate the analyst narrative only. Chart selection is deterministic/local."""
    intent = intent or {}
    chart_note = "A chart was explicitly requested; describe the result succinctly." if chart_requested else "No chart was requested."
    system_prompt = f"""You are a senior BI data analyst. Answer the user's business question using ONLY the supplied query result.
Do not invent facts, numbers, causes, or dimensions. {chart_note}
Start a complex answer with one-sentence conclusion.
Then give 2-5 concise evidence bullets with actual values from the result.
For benchmark analysis, distinguish the dimension benchmark from company total and report segment sales, benchmark,
difference and percentage difference when available.
Use SUM for business scale and AVG for average performance.
For impact ranking, explain scale + underperformance when impact_score is present.
Recommendations must directly follow the strongest evidence; if evidence does not show a decline/problem, say so.
Never claim causation from observational data: say 'associated with', 'coincides with', or 'concentrated in'.
If the result is insufficient, say exactly what is supported.
Return plain text only; no JSON, markdown tables, code, or chart specification.
QUESTION:{question}
INTENT:{json.dumps(intent, separators=(',', ':'))}
RESULT COLUMNS:{columns}"""
    # Result data is compacted before the call to protect TPM while retaining useful evidence.
    result_short = result_df_json[:6500]
    try:
        answer = _call_groq(system_prompt, f"SQL:{sql}\nRESULT:{result_short}", max_completion_tokens=800, temperature=0.1).strip()
        if not answer:
            raise ValueError("Empty analyst response")
    except Exception:
        # The app supplies a stronger local fallback if the LLM is temporarily unavailable.
        class _Rows:
            def __init__(self, payload):
                self.empty = not payload
                self._payload = payload
            def head(self, n):
                return self
            def iterrows(self):
                for i, r in enumerate(self._payload[:3]):
                    yield i, r
        try:
            payload = json.loads(result_short)
        except Exception:
            payload = []
        answer = _fallback_answer(question, _Rows(payload))
    return {"answer": answer, "chart": {"type": "none", "x": None, "y": None, "color": None,
                                         "names": None, "values": None, "z": None, "title": ""}}
