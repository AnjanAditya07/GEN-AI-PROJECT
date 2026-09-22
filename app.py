"""
GenAI-Powered Business Intelligence Chatbot
Scope: CSV / Excel files only. Cleaned data is always standardized to CSV.

Run with:  streamlit run app.py
Requires:  GROQ_API_KEY environment variable set
"""

import streamlit as st
import pandas as pd
import plotly.io as pio
import os

from utils.cleaning import process_upload
from utils.query_engine import QueryEngine
from utils.llm import generate_sql, generate_answer_and_chart, repair_sql, generate_normal_response
from utils.charts import CHART_TYPES, CHART_LABELS, build_chart, build_manual_chart, auto_spec_from_intent
from utils.intent import detect_intent
import re

st.set_page_config(page_title="Ledger — BI Chatbot", page_icon="◧", layout="wide")

UPLOAD_DIR = "data/uploads"
CLEANED_DIR = "data/cleaned"
LOG_DIR = "data/logs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


INK = "#0E1116"          
PANEL = "#141A22"        
PANEL_2 = "#1B2330"      
LINE = "#262F3B"         
TEXT = "#F2F4F7"         
MUTED = "#AEB6C4"        
BRASS = "#E3A857"     
TEAL = "#5FBFAE"      
RUST = "#D9705F"         


pio.templates["ledger"] = pio.templates["plotly_dark"]
pio.templates["ledger"].layout.update(
    paper_bgcolor=PANEL_2,
    plot_bgcolor=PANEL_2,
    font=dict(family="Inter, sans-serif", color=TEXT, size=13),
    colorway=[BRASS, TEAL, "#8E9AAF", "#D9705F", "#C9B458", "#6E8FA3"],
    margin=dict(t=48, l=16, r=16, b=16),
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    title=dict(font=dict(size=15, color=TEXT)),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
pio.templates.default = "ledger"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    code, pre, .stCode, [data-testid="stCodeBlock"] * {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .stApp {{
        background: {INK};
    }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {LINE};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.6rem;
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 0;
    }}

    /* ---------- Headings ---------- */
    h1, h2, h3 {{
        color: {TEXT};
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .lg-brand {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.15rem;
    }}
    .lg-brand-mark {{
        width: 30px; height: 30px;
        border-radius: 7px;
        background: linear-gradient(155deg, {BRASS}, #B9793A);
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; color: {INK}; font-weight: 700;
        flex-shrink: 0;
    }}
    .lg-brand-name {{
        font-size: 1.18rem; font-weight: 700; color: {TEXT};
        letter-spacing: -0.01em;
    }}
    .lg-tagline {{
        color: {MUTED}; font-size: 0.83rem; margin: 0 0 1.1rem 0; line-height: 1.4;
    }}
    .lg-section-label {{
        font-size: 0.72rem; font-weight: 600; color: {MUTED};
        text-transform: uppercase; letter-spacing: 0.06em;
        margin: 1.2rem 0 0.5rem 0;
    }}

    /* ---------- Generic surfaces ---------- */
    .lg-card {{
        background: {PANEL_2};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }}
    .lg-table-card {{
        background: {PANEL_2};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.55rem;
    }}
    .lg-table-name {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.86rem; font-weight: 600; color: {TEXT};
    }}
    .lg-table-meta {{
        font-size: 0.76rem; color: {MUTED}; margin-top: 0.15rem;
    }}
    .lg-pill {{
        display: inline-block;
        background: rgba(227, 168, 87, 0.12);
        color: {BRASS};
        border: 1px solid rgba(227, 168, 87, 0.35);
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }}

    /* ---------- API key notice ---------- */
    .lg-alert {{
        background: rgba(217, 112, 95, 0.1);
        border: 1px solid rgba(217, 112, 95, 0.4);
        color: #F0B5AA;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        background: {PANEL_2};
        color: {TEXT};
        border: 1px solid {LINE};
        border-radius: 8px;
        font-weight: 500;
        transition: border-color 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {BRASS};
        color: {BRASS};
    }}
    .stDownloadButton > button {{
        background: transparent;
        color: {TEAL};
        border: 1px solid rgba(95, 191, 174, 0.4);
        border-radius: 8px;
        font-size: 0.82rem;
    }}
    .stDownloadButton > button:hover {{
        border-color: {TEAL};
        background: rgba(95, 191, 174, 0.08);
    }}

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {{
        background: {PANEL_2};
        border: 1px dashed {LINE};
        border-radius: 10px;
    }}

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {{
        background: transparent;
        gap: 0.6rem;
    }}
    .lg-hero {{
        text-align: center;
        padding: 3.2rem 1rem 2.4rem 1rem;
    }}
    .lg-hero-mark {{
        width: 52px; height: 52px; margin: 0 auto 1rem auto;
        border-radius: 12px;
        background: linear-gradient(155deg, {BRASS}, #B9793A);
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; color: {INK}; font-weight: 700;
    }}
    .lg-hero h2 {{
        font-size: 1.5rem; margin-bottom: 0.35rem;
    }}
    .lg-hero p {{
        color: {MUTED}; font-size: 0.92rem; max-width: 480px; margin: 0 auto;
        line-height: 1.5;
    }}

    .lg-sql-label {{
        font-size: 0.75rem; color: {MUTED}; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 0.3rem;
    }}

    [data-testid="stChatInput"] {{
        border-color: {LINE};
    }}

    /* ---------- Misc ---------- */
    hr {{ border-color: {LINE}; }}
    [data-testid="stExpander"] {{
        background: {PANEL_2};
        border: 1px solid {LINE};
        border-radius: 8px;
    }}
    .lg-caption {{
        color: {MUTED}; font-size: 0.78rem;
    }}

    /* ---------- Readable text everywhere (fixes dim-gray-on-black) ---------- */
    .stApp, .stApp p, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stChatMessage"] p,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] span,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    .stButton > button p {{
        color: {TEXT} !important;
    }}
    /* Streamlit captions / helper text: keep muted but still legible */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stMarkdownContainer"] small,
    .lg-tagline, .lg-table-meta, .lg-caption {{
        color: {MUTED} !important;
    }}
    /* Dataframe / table text */
    [data-testid="stDataFrame"] * {{
        color: {TEXT} !important;
    }}
    [data-testid="stDataFrame"] {{
        background: {PANEL_2};
    }}
    /* Chat input text the user types */
    [data-testid="stChatInput"] textarea {{
        color: {TEXT} !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 1;
    }}
    /* st.status widget label */
    [data-testid="stStatusWidget"] p,
    [data-testid="stStatusWidget"] label {{
        color: {TEXT} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Session state
# ============================================================
if "engine" not in st.session_state:
    st.session_state.engine = QueryEngine()
if "tables_loaded" not in st.session_state:
    st.session_state.tables_loaded = []  # list of dicts from process_upload
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"/"assistant", "content": ...}]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

engine: QueryEngine = st.session_state.engine

ASSISTANT_AVATAR = "📊"
USER_AVATAR = "🧑‍💻"


with st.sidebar:
    st.markdown(
        """
        <div class="lg-brand">
            <div class="lg-brand-mark">◧</div>
            <div class="lg-brand-name">Ledger</div>
        </div>
        <p class="lg-tagline">Ask plain-English questions about your CSV or Excel data and get SQL-backed answers with charts.</p>
        """,
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.markdown(
            '<div class="lg-alert">⚠️ <strong>GROQ_API_KEY</strong> is not set. '
            "Add it as an environment variable — see the README for setup.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="lg-section-label">Upload data</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            already_loaded = any(t["original_filename"] == uf.name for t in st.session_state.tables_loaded)
            if already_loaded:
                continue
            with st.spinner(f"Cleaning {uf.name}…"):
                tmp_path = os.path.join(UPLOAD_DIR, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.getbuffer())

                results = process_upload(tmp_path, output_dir=CLEANED_DIR, log_dir=LOG_DIR)
                for r in results:
                    r["original_filename"] = uf.name
                    engine.register_table(r["table_name"], r["csv_path"])
                    st.session_state.tables_loaded.append(r)
            st.toast(f"Loaded {uf.name}", icon="✅")

    st.markdown('<div class="lg-section-label">Loaded tables</div>', unsafe_allow_html=True)
    if not st.session_state.tables_loaded:
        st.markdown(
            '<p class="lg-caption">Nothing here yet — upload a file above to get started.</p>',
            unsafe_allow_html=True,
        )
    else:
        for t in st.session_state.tables_loaded:
            n_actions = len(t["report"].get("actions", []))
            extra = f" · {n_actions} cleaning step{'s' if n_actions != 1 else ''}" if n_actions else ""
            st.markdown(
                f"""
                <div class="lg-table-card">
                    <div class="lg-table-name">🗂 {t['table_name']}</div>
                    <div class="lg-table-meta">{t['row_count']:,} rows · {len(t['columns'])} columns{extra}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Details"):
                report = t["report"]
                st.markdown(f"**Columns:** {', '.join(t['columns'])}")
                removed_rows = int(report.get("rows_removed_total", report.get("rows_in", 0) - report.get("rows_out", 0)))
                blank_rows = int(report.get("blank_rows_removed", 0))
                duplicate_rows = int(report.get("duplicate_rows_removed", 0))
                missing_cells = int(report.get("missing_cells", sum(report.get("null_summary", {}).values())))
                standardized_cols = int(report.get("columns_standardized", 0))
                st.markdown(
                    f"**Data-quality report:** {removed_rows:,} rows removed "
                    f"({blank_rows:,} blank · {duplicate_rows:,} duplicates) · "
                    f"{len(report.get('null_summary', {}))} columns with missing values "
                    f"({missing_cells:,} missing cells) · "
                    f"{standardized_cols:,} columns standardized · "
                    f"{len(report.get('actions', []))} cleaning/type-conversion actions"
                )
                if report["actions"]:
                    st.markdown("**Cleaning applied:**")
                    for a in t["report"]["actions"]:
                        st.caption(f"• {a}")
                with open(t["csv_path"], "rb") as f:
                    st.download_button(
                        "⬇ Download cleaned CSV",
                        f,
                        file_name=os.path.basename(t["csv_path"]),
                        mime="text/csv",
                        key=f"dl_{t['table_name']}",
                        use_container_width=True,
                    )

    if st.session_state.tables_loaded:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col_b:
            if st.button("Clear data", use_container_width=True):
                st.session_state.engine = QueryEngine()
                st.session_state.tables_loaded = []
                st.session_state.chat_history = []
                st.rerun()


if not st.session_state.tables_loaded:
    st.markdown(
        """
        <div class="lg-hero">
            <div class="lg-hero-mark">◧</div>
            <h2>Ask your data a question</h2>
            <p>Upload a CSV or Excel file in the sidebar, and Ledger will clean it,
            load it into a queryable table, and let you explore it in plain English —
            no SQL required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

n_tables = len(st.session_state.tables_loaded)
total_rows = sum(t["row_count"] for t in st.session_state.tables_loaded)
plural_tables = "s" if n_tables != 1 else ""
st.markdown(
    f'<span class="lg-pill">● {n_tables} table{plural_tables} loaded · {total_rows:,} rows ready to query</span>',
    unsafe_allow_html=True,
)

if not st.session_state.chat_history:
    first_table = st.session_state.tables_loaded[0]
    cols = first_table["columns"]
    example_qs = [
        f"Give me a summary of {first_table['table_name']}",
        f"What are the top 10 rows by {cols[-1]}?" if cols else "What's in this dataset?",
        "Are there any outliers or trends worth noting?",
    ]
    st.markdown(
        """
        <div class="lg-hero" style="padding-top:1.2rem;">
            <h2 style="font-size:1.2rem;">Data's loaded — what do you want to know?</h2>
            <p>Try one of these, or type your own question below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(len(example_qs))
    for c, eq in zip(chip_cols, example_qs):
        with c:
            if st.button(eq, key=f"chip_{eq}", use_container_width=True):
                st.session_state.pending_question = eq
                st.rerun()


CHART_PICKER_OPTIONS = ["Auto (AI decides)"] + [CHART_LABELS[t] for t in CHART_TYPES] + ["No chart"]
LABEL_TO_TYPE = {v: k for k, v in CHART_LABELS.items()}


def _protect_causation_language(text: str) -> str:
    """Keep observational BI findings from being phrased as proven causation."""
    if not text:
        return text
    for phrase, replacement in (
        ("caused the decline", "are associated with the decline"),
        ("caused this decline", "are associated with this decline"),
        ("caused the drop", "are associated with the drop"),
        ("caused this drop", "are associated with this drop"),
        ("is the cause of", "is associated with"),
        ("was the cause of", "was associated with"),
    ):
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement, text, flags=re.I)
    return text

def _is_generic_chart_followup(question: str, intent: dict) -> bool:
    """Return True when this turn is only asking to visualize the previous result.

    Examples that should reuse the previous dataframe:
      - "show it using a chart for better visualization"
      - "visualize this"
      - "display the above as a line chart"

    A request that names a fresh metric, dimension, ranking, or time grain is not
    generic and should go through SQL normally (for example,
    "show sales by region as a bar chart").
    """
    if not intent.get("chart_requested"):
        return False

    q = (question or "").casefold()


    data_terms = re.compile(
        r"\b(sales|revenue|profit|margin|orders?|units?|quantity|region|state|country|"
        r"retailer|product|category|channel|sales\s+method|month|monthly|quarter|quarterly|"
        r"year|yearly|trend|top|bottom|highest|lowest|best|worst|rank|ranking|compare|"
        r"comparison|versus|vs|average|avg|sum|total|count|forecast|predict)\b",
        re.I,
    )
    if data_terms.search(q):
        return False

    return True


def _interpret_result(question, sql, df_result, intent):
    """Generate the final evidence-based answer; return a safe fallback on LLM failure."""
    try:
        result_json = df_result.head(60).to_json(orient="records")
        return generate_answer_and_chart(
            question, sql, result_json, list(df_result.columns), intent=intent,
            chart_requested=bool(intent.get("chart_requested"))
        )
    except Exception as exc:
        msg = str(exc)
        if "413" in msg or "rate_limit" in msg.lower() or "tokens per minute" in msg.lower():
            answer = "The data query succeeded, but the AI interpretation request exceeded the current Groq token-per-minute limit. The result table is shown below; please retry for the narrative summary."
        elif "GROQ_API_KEY" in msg:
            answer = "The data query succeeded, but GROQ_API_KEY is not available for the final interpretation."
        else:
            answer = f"The data query succeeded, but I couldn't generate the narrative interpretation: `{msg[:250]}`"
        return {"answer": answer, "chart": {"type": "none"}}


def render_assistant_result(msg: dict, idx: int):
    """Renders one assistant turn. What gets shown depends on what the user
    actually asked for in that turn (see utils.intent.detect_intent):

      - Asked to see/generate the SQL query -> the "View SQL used" panel is
        shown. Otherwise it stays hidden and the reply is just the summary.
      - Asked for "only"/"just" a chart -> render ONLY the chart, nothing
        else (no text, no table, no SQL).
      - Named a particular chart type (e.g. "bar chart") -> that type is
        used/pre-selected instead of the AI's auto-picked chart.

    Shared by the history loop and the just-generated turn so both behave
    identically.
    """
    df = msg.get("table")
    chart_spec = msg.get("chart_spec") or {}
    ai_title = chart_spec.get("title") or msg.get("question") or ""
    requested_chart_type = msg.get("requested_chart_type")

    chart_requested = msg.get("chart_requested")
    if chart_requested is None:
        chart_requested = bool(chart_spec.get("type") and chart_spec.get("type") != "none")


    if msg.get("chart_only") and df is not None:
        fig = None
        

        candidates = [requested_chart_type] if requested_chart_type else [chart_spec.get("type"), *CHART_TYPES]
        for ctype in candidates:
            if not ctype or ctype == "none":
                continue
            if ctype == chart_spec.get("type") and not requested_chart_type:
                fig = build_chart(df, ctype, chart_spec, title=ai_title)
            else:
                fig = build_manual_chart(df, ctype, title=ai_title)
            if fig is not None:
                break
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_render_{idx}")
        else:
            st.caption("Couldn't build a chart for this result.")
        return

    st.markdown(_protect_causation_language(msg["content"]))

 
    if msg.get("sql") and (msg.get("wants_sql") or df is None):
        with st.expander("View SQL used", expanded=bool(msg.get("wants_sql"))):
            st.code(msg["sql"], language="sql")

    if df is None:
        return

    if not chart_requested:
        st.dataframe(df, use_container_width=True)
        n_rows = len(df)
        st.caption(f"{n_rows:,} row{'s' if n_rows != 1 else ''} returned")
        return

   
    default_label = CHART_LABELS.get(requested_chart_type, "Auto (AI decides)")
    default_index = CHART_PICKER_OPTIONS.index(default_label) if default_label in CHART_PICKER_OPTIONS else 0

    pick = st.selectbox(
        "Chart type",
        CHART_PICKER_OPTIONS,
        index=default_index,
        key=f"chart_pick_{idx}",
        label_visibility="collapsed",
    )

    fig = None
    if pick == "Auto (AI decides)":
     
        preferred = chart_spec.get("type")
        candidates = [preferred] if preferred and preferred != "none" else []
        candidates += [c for c in CHART_TYPES if c not in candidates]
        for ctype in candidates:
            if ctype == preferred and preferred != "none":
                fig = build_chart(df, ctype, chart_spec, title=ai_title)
            else:
                fig = build_manual_chart(df, ctype, title=ai_title)
            if fig is not None:
                break
    elif pick != "No chart":
        chosen_type = LABEL_TO_TYPE[pick]
        fig = build_manual_chart(df, chosen_type, title=ai_title)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"chart_render_{idx}")
    elif pick not in ("Auto (AI decides)", "No chart"):
        st.caption(f"A {pick.lower()} chart doesn't fit this result's columns.")
    elif len(df) > 0:
        st.caption("The result is non-empty, but its columns could not be mapped to a supported chart type.")

    st.dataframe(df, use_container_width=True)
    n_rows = len(df)
    st.caption(f"{n_rows:,} row{'s' if n_rows != 1 else ''} returned")


for i, msg in enumerate(st.session_state.chat_history):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_assistant_result(msg, i)
        else:
            st.markdown(_protect_causation_language(msg["content"]))

question = st.chat_input("e.g. What were total sales by region last quarter?")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    intent = detect_intent(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.status("Working on it…", expanded=False) as status:
            schema_desc = engine.get_relevant_schema_description(question, intent=intent)
            history_for_llm = []
            for m in st.session_state.chat_history[:-1][-8:]:
                item = {"role": m["role"], "content": m.get("content", "")}
                if m.get("role") == "assistant":
                    item["previous_question"] = m.get("question", "")
                    item["previous_sql"] = m.get("sql", "")
                    prev_df = m.get("table")
                    if prev_df is not None:
                        item["previous_result"] = prev_df.head(20).to_dict(orient="records")
                history_for_llm.append(item)

        
            previous_data_msg = next(
                (m for m in reversed(st.session_state.chat_history[:-1])
                 if m.get("role") == "assistant"
                 and m.get("table") is not None
                 and len(m.get("table")) > 0),
                None,
            )
            generic_chart_followup = _is_generic_chart_followup(question, intent)

            if generic_chart_followup and previous_data_msg is not None:
                status.update(label="Building the chart from the previous result…")
                df_result = previous_data_msg.get("table")
          

                chart_intent = dict(previous_data_msg.get("intent") or previous_data_msg.get("analysis_type") and {"analysis_type": previous_data_msg.get("analysis_type")} or {})
                chart_intent.update({k: v for k, v in intent.items() if v})
                chart_intent["chart_requested"] = True
                chart_intent["requested_chart_type"] = intent.get("requested_chart_type") or previous_data_msg.get("requested_chart_type")
                chart_spec = auto_spec_from_intent(
                    df_result, intent=chart_intent,
                    requested_type=chart_intent.get("requested_chart_type")
                )
                status.update(label="Done", state="complete")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Here is the requested visualization of the previous result.",
                    "sql": previous_data_msg.get("sql"),
                    "table": df_result,
                    "chart_spec": chart_spec,
                    "question": question,
                    "wants_sql": intent.get("wants_sql", False),
                    "chart_only": intent.get("chart_only", False),
                    "chart_requested": True,
                    "requested_chart_type": intent.get("requested_chart_type"),
                    "analysis_type": previous_data_msg.get("analysis_type"),
                    "intent": intent,
                })
    

            elif not intent.get("requires_data", True):
                status.update(label="Answering in natural language…")
                try:
                    answer_text = generate_normal_response(question, chat_history=history_for_llm)
                except Exception:
                    answer_text = "I can answer that in natural language. For dataset-specific questions, ask me about sales, rankings, comparisons, trends, or business drivers."
                status.update(label="Done", state="complete")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "question": question,
                    "wants_sql": False,
                    "chart_only": False,
                    "chart_requested": False,
                    "requested_chart_type": None,
                    "analysis_type": intent.get("analysis_type"),
                    "intent": intent,
                })
            else:
                status.update(label="Translating your question into SQL…")
                try:
                    sql_result = generate_sql(question, schema_desc, chat_history=history_for_llm, intent=intent)
                except Exception as exc:
                    status.update(label="Could not contact the AI service", state="error")
                    msg = str(exc)
                    if "413" in msg or "rate_limit" in msg.lower() or "tokens per minute" in msg.lower():
                        explanation = "The AI request exceeded the current Groq token-per-minute limit. The chatbot uses compact prompts; please retry in a moment."
                    elif "GROQ_API_KEY" in msg:
                        explanation = "GROQ_API_KEY is not set. Set the API key in CMD and restart Streamlit."
                    else:
                        explanation = f"I couldn't generate the data query: {msg[:300]}"
                    sql_result = {"sql": None, "explanation": explanation}
                sql = sql_result.get("sql")
    
                if not sql:
                    status.update(label="Couldn't build a query", state="error")
                    answer_text = str(sql_result.get('explanation', 'Unclear question.'))
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
                else:
                    status.update(label="Running the query against your data…")
                    try:
                        df_result = engine.run_query(sql)
                    except Exception as first_error:
            

                        status.update(label="Repairing the generated SQL…")
                        try:
                            repaired = repair_sql(
                                question, schema_desc, sql, str(first_error),
                                chat_history=history_for_llm, intent=intent
                            )
                        except Exception as exc:
                            repaired = {"sql": None, "explanation": str(exc)}
                        repaired_sql_text = repaired.get("sql")
    
                        if repaired_sql_text:
                            try:
                                df_result = engine.run_query(repaired_sql_text)
                                sql = repaired_sql_text
                            except Exception as second_error:
                                status.update(label="Query failed", state="error")
                                answer_text = (
                                    "The generated query could not be executed even after an "
                                    f"automatic SQL repair: `{second_error}`"
                                )
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": answer_text, "sql": sql}
                                )
                                df_result = None
                        else:
                            status.update(label="Query failed", state="error")
                            answer_text = f"The generated query failed to run: `{first_error}`"
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer_text, "sql": sql}
                            )
                            df_result = None
    
                        if df_result is None:
                            pass
                        else:
                            status.update(label="Interpreting the results…")
                            interp = _interpret_result(question, sql, df_result, intent)
                            answer_text = interp.get("answer", "Here's what I found.")
                            chart_spec = auto_spec_from_intent(df_result, intent=intent, requested_type=intent.get("requested_chart_type")) if intent.get("chart_requested") else {"type": "none"}
    
                            status.update(label="Done", state="complete")
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer_text,
                                "sql": sql,
                                "table": df_result,
                                "chart_spec": chart_spec,
                                "question": question,
                                "wants_sql": intent["wants_sql"],
                                "chart_only": intent["chart_only"],
                                "chart_requested": intent["chart_requested"],
                                "requested_chart_type": intent["requested_chart_type"],
                                "analysis_type": intent.get("analysis_type"),
                                "intent": intent,
                            })
                    else:
                        status.update(label="Interpreting the results…")
                        interp = _interpret_result(question, sql, df_result, intent)
                        answer_text = interp.get("answer", "Here's what I found.")
                        chart_spec = auto_spec_from_intent(df_result, intent=intent, requested_type=intent.get("requested_chart_type")) if intent.get("chart_requested") else {"type": "none"}
    
                        status.update(label="Done", state="complete")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sql": sql,
                            "table": df_result,
                            "chart_spec": chart_spec,
                            "question": question,
                            "wants_sql": intent["wants_sql"],
                            "chart_only": intent["chart_only"],
                            "chart_requested": intent["chart_requested"],
                            "requested_chart_type": intent["requested_chart_type"],
                            "analysis_type": intent.get("analysis_type"),
                            "intent": intent,
                        })


        last_idx = len(st.session_state.chat_history) - 1
        render_assistant_result(st.session_state.chat_history[last_idx], last_idx)
