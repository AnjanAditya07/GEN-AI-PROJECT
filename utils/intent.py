"""Business-question intent and chart-request detection."""
import re

# Ordered from the most specific/explicit chart requests to generic requests.
_CHART_TYPE_PATTERNS = [
    ("histogram", [r"\bhistogram\b"]),
    ("heatmap", [r"\bheat\s*-?\s*map\b"]),
    ("treemap", [r"\btree\s*-?\s*map\b"]),
    ("scatter", [r"\bscatter\s*(chart|graph|plot)?\b", r"\bscatterplot\b"]),
    ("box", [r"\bbox\s*(plot|chart)\b", r"\bboxplot\b"]),
    ("pie", [r"\bpie\s*(chart|graph|plot)?\b", r"\b(donut|doughnut)\s*(chart|graph|plot)?\b"]),
    ("area", [r"\barea\s*(chart|graph|plot)\b"]),
    ("line", [
        r"\bline\s*(chart|graph|plot)\b",
        r"\btrend\s*line\b",
        r"\b(line)\s+visual\b",
    ]),
    ("bar", [
        r"\bbar\s*(chart|graph|plot)\b",
        r"\bbar\s+visual\b",
    ]),
]
_CHART_WORD_RE = re.compile(
    r"\b(chart|charts|graph|graphs|plot|plots|visual|visuals|visuali[sz]ation|visuali[sz]e|visuali[sz]e|visualize|visualise)\b",
    re.I,
)
_ONLY_WORD_RE = re.compile(r"\b(only|just)\b", re.I)
_SQL_INTENT_RE = re.compile(
    r"\bsql\b|"
    r"\b(generate|show|write|give|see|view|display|share)\b[^.?!\n]{0,25}\bquery\b|"
    r"\bquery\b[^.?!\n]{0,20}\b(used|behind|generated|you (?:used|wrote|ran))\b|"
    r"\bwhat(?:'?s| is| was)\b[^.?!\n]{0,15}\bquery\b", re.I
)

DRIVER_TERMS = (
    "why", "factors", "drivers", "causes", "contributing", "affecting",
    "underperforming", "low sales", "declining revenue", "poor performance",
    "reason", "what is causing", "what affects",
)
FORECAST_TERMS = ("forecast", "predict", "prediction", "future sales", "next month", "next quarter")
RECOMMEND_TERMS = (
    "recommend", "recommendation", "recommendations", "suggest", "suggestion", "suggestions",
    "what should", "what can we do", "what do you suggest", "action", "actions",
    "how should we", "how can we", "how do we", "ways to improve", "way to improve",
    "improve", "improve sales", "improve revenue", "boost sales", "increase sales",
    "recover sales", "reduce decline", "address the decline", "fix the decline",
    "deal with the decline", "prevent further decline", "sales improvement",
)
RANKING_TERMS = ("top", "bottom", "highest", "lowest", "best", "worst", "rank", "ranking", "largest", "smallest")
COMPARISON_TERMS = ("compare", "comparison", "versus", "vs", "difference", "higher than", "lower than")
TREND_TERMS = (
    "trend", "over time", "monthly", "month over month", "mom", "year over year", "yoy",
    "quarterly", "quarter over quarter", "qoq", "declining", "decline", "before", "after",
    "sales trend", "revenue trend", "time series", "timeseries",
)
KPI_TERMS = ("kpi", "total sales", "total revenue", "average sales", "average revenue", "sales", "revenue", "orders", "order volume", "count", "margin", "profit")

GENERAL_LANGUAGE_PATTERNS = (
    r"^(hi|hello|hey|good morning|good afternoon|good evening)[!. ]*$",
    r"^(thanks|thank you|thx|ok|okay|great|nice)[!. ]*$",
    r"\b(what is|what are|explain|define|meaning of)\s+(?:a|an|the)?\s*(sql|kpi|bi|business intelligence|power bi|tableau|duckdb|machine learning|bar chart|line chart|pie chart|scatter plot|heatmap|treemap)\b",
    r"\b(how do you work|what can you do|what can this chatbot do|help|help me)\b",
)


def _is_general_language_question(q: str) -> bool:
    return any(re.search(pattern, q, re.I) for pattern in GENERAL_LANGUAGE_PATTERNS)


def _contains(q: str, terms) -> bool:
    return any(term in q for term in terms)


def _requested_chart_type(q: str):
    """Return an explicitly requested chart type, including natural phrasing such as
    'use a pie', 'as a bar chart', and 'show it as a line'."""
    for chart_type, patterns in _CHART_TYPE_PATTERNS:
        if any(re.search(pattern, q, re.I) for pattern in patterns):
            return chart_type

    # Natural-language variants where the word 'chart' is separated from the type.
    natural = {
        "bar": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*bar\b|\bbar\s+(?:visual|visualization)\b",
        "line": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*line\b|\bline\s+(?:visual|visualization)\b",
        "area": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*area\b",
        "scatter": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*scatter\b",
        "pie": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*(?:pie|donut|doughnut)\b",
        "histogram": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*histogram\b",
        "box": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*box(?:\s*plot)?\b",
        "heatmap": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*heat\s*-?\s*map\b",
        "treemap": r"\b(?:as|using|with|make|use|show|display|create|draw|plot)\s+(?:a|an)?\s*tree\s*-?\s*map\b",
    }
    for chart_type, pattern in natural.items():
        if re.search(pattern, q, re.I):
            return chart_type
    return None


def detect_intent(question: str) -> dict:
    q = (question or "").strip().casefold()
    # Table/column names are often snake_case (e.g. "Nike_Dataset", "total_sales").
    # Underscore counts as a "word" character in regex, so "\bdataset\b" would not
    # match inside "nike_dataset" and keyword detection would silently fail for any
    # question that references a table/column by its literal name. Normalizing
    # underscores to spaces gives every \b-based pattern below a real boundary to
    # match against, while substring ("in") checks keep working unchanged.
    q = q.replace("_", " ")
    wants_sql = bool(_SQL_INTENT_RE.search(q))
    general_language = _is_general_language_question(q)

    requested_chart_type = _requested_chart_type(q)
    chart_requested = requested_chart_type is not None or bool(_CHART_WORD_RE.search(q))
    chart_only = chart_requested and bool(_ONLY_WORD_RE.search(q))

    driver_analysis = _contains(q, DRIVER_TERMS)
    trend = _contains(q, TREND_TERMS)
    ranking = _contains(q, RANKING_TERMS)
    comparison = _contains(q, COMPARISON_TERMS)
    forecasting = _contains(q, FORECAST_TERMS)
    recommendation = _contains(q, RECOMMEND_TERMS)

    recommendation_context = recommendation and bool(re.search(
        r"\b(sales|revenue|profit|margin|orders|decline|declining|low|underperform|performance)\b", q
    ))
    if recommendation_context:
        trend = trend or bool(re.search(r"\b(decline|declining|drop|fall|decrease|low)\b", q))
        diagnostic = True
    else:
        diagnostic = driver_analysis or "diagnos" in q or "problem" in q or "underperform" in q

    data_summary = bool(re.search(r"\b(summary|summarize|describe|overview)\b", q) and re.search(r"\b(dataset|data|table|file)\b", q)) or bool(
        re.search(r"\bwhat(?:\x27s| is) in (?:the )?(dataset|data|table|file)\b", q)
    )

    if driver_analysis:
        analysis_type = "driver"
    elif forecasting:
        analysis_type = "forecast"
    elif recommendation:
        analysis_type = "recommendation"
    elif trend:
        analysis_type = "trend"
    elif ranking:
        analysis_type = "ranking"
    elif comparison:
        analysis_type = "comparison"
    elif diagnostic:
        analysis_type = "diagnostic"
    elif _contains(q, KPI_TERMS):
        analysis_type = "kpi"
    else:
        analysis_type = "kpi"

    requires_data = (not general_language) and (
        wants_sql or driver_analysis or trend or ranking or comparison or forecasting
        or recommendation or diagnostic or data_summary or chart_requested or _contains(q, KPI_TERMS)
    )

    return {
        "wants_sql": wants_sql,
        "chart_only": chart_only,
        "chart_requested": chart_requested,
        "requested_chart_type": requested_chart_type,
        "analysis_type": analysis_type,
        "driver_analysis": driver_analysis,
        "trend": trend,
        "ranking": ranking,
        "comparison": comparison,
        "forecasting": forecasting,
        "recommendation": recommendation,
        "recommendation_context": recommendation_context,
        "requires_data": requires_data,
        "general_language": general_language,
    }
