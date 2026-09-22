"""
Chart building layer.

Two entry points:
  - build_chart(df, spec)         -> render using the AI's chosen type + columns
  - build_manual_chart(df, type_) -> render a *user-picked* type, auto-inferring
                                      sensible columns from the dataframe's dtypes

Both funnel through the same per-type renderers so the two systems (AI-decided,
user-decided) never draw a given chart type differently.
"""

import pandas as pd
import plotly.express as px

# The full set of chart types the app can render. Shown in the UI's manual
# picker and accepted from the AI's chart spec.
CHART_TYPES = ["bar", "line", "area", "scatter", "pie", "histogram", "box", "heatmap", "treemap"]

CHART_LABELS = {
    "bar": "Bar",
    "line": "Line",
    "area": "Area",
    "scatter": "Scatter",
    "pie": "Pie",
    "histogram": "Histogram",
    "box": "Box plot",
    "heatmap": "Heatmap",
    "treemap": "Treemap",
}


def _cols_by_kind(df: pd.DataFrame):
    """Split columns into numeric / datetime / categorical buckets, in original order."""
    numeric, datetime_, categorical = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_.append(col)
        else:
            # Try a cheap datetime sniff for object columns that are really dates
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(20)
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if len(sample) > 0 and parsed.notna().mean() > 0.8:
                    datetime_.append(col)
                    continue
            categorical.append(col)
    return numeric, datetime_, categorical


def infer_spec_for_type(df: pd.DataFrame, chart_type: str) -> dict:
    """
    Given a dataframe and a user-chosen chart type, pick reasonable columns.
    Used for the manual "you decide" chart system, where there's no LLM spec.
    Returns a spec dict compatible with build_chart(), or {} if the shape
    genuinely doesn't support that chart type.
    """
    numeric, datetime_, categorical = _cols_by_kind(df)
    ordered_x = datetime_ + categorical  # prefer a date axis over a plain category

    spec = {"x": None, "y": None, "color": None, "names": None, "values": None, "z": None, "title": None}

    if chart_type in ("bar", "line", "area"):
        if not (ordered_x and numeric):
            return {}
        spec["x"], spec["y"] = ordered_x[0], numeric[0]

    elif chart_type == "scatter":
        # Prefer two numeric measures, but permit category/date + numeric when
        # the user explicitly asks for a scatter plot.
        if len(numeric) >= 2:
            spec["x"], spec["y"] = numeric[0], numeric[1]
            if categorical:
                spec["color"] = categorical[0]
        elif (ordered_x and numeric):
            spec["x"], spec["y"] = ordered_x[0], numeric[0]

    elif chart_type == "pie":
        if not (ordered_x and numeric):
            return {}
        spec["names"], spec["values"] = ordered_x[0], numeric[0]

    elif chart_type == "histogram":
        if not numeric:
            return {}
        spec["x"] = numeric[0]

    elif chart_type == "box":
        if not numeric:
            return {}
        spec["y"] = numeric[0]
        if categorical:
            spec["x"] = categorical[0]

    elif chart_type == "heatmap":
        if len(categorical) >= 2 and numeric:
            spec["x"], spec["y"], spec["z"] = categorical[0], categorical[1], numeric[0]
        elif len(categorical) >= 1 and numeric:
            # Explicit heatmap request with one dimension: render a one-column
            # metric matrix rather than silently changing the requested chart type.
            spec["x"], spec["z"] = categorical[0], numeric[0]
        elif len(datetime_) >= 1 and numeric:
            spec["x"], spec["z"] = datetime_[0], numeric[0]
        elif len(numeric) >= 2:
            spec["_correlation"] = True
        else:
            return {}

    elif chart_type == "treemap":
        if not (ordered_x and numeric):
            return {}
        spec["names"], spec["values"] = ordered_x[0], numeric[0]

    else:
        return {}

    return spec


def build_chart(df: pd.DataFrame, chart_type: str, spec: dict, title: str = None):
    """
    Render a Plotly figure for the given type + spec against df.
    spec keys: x, y, color, names, values, z (all optional depending on type).
    Returns a Plotly figure, or None if the spec doesn't fit the data.
    """
    if not chart_type or chart_type == "none":
        return None

    cols = set(df.columns)

    def has(*keys):
        return all(spec.get(k) and spec.get(k) in cols for k in keys)

    color = spec.get("color") if spec.get("color") in cols else None
    title = title or spec.get("title") or ""

    try:
        if chart_type == "bar" and has("x", "y"):
            fig = px.bar(df, x=spec["x"], y=spec["y"], color=color, title=title)
        elif chart_type == "line" and has("x", "y"):
            fig = px.line(df, x=spec["x"], y=spec["y"], color=color, title=title)
        elif chart_type == "area" and has("x", "y"):
            fig = px.area(df, x=spec["x"], y=spec["y"], color=color, title=title)
        elif chart_type == "scatter" and has("x", "y"):
            fig = px.scatter(df, x=spec["x"], y=spec["y"], color=color, title=title)
        elif chart_type == "pie" and has("names", "values"):
            fig = px.pie(df, names=spec["names"], values=spec["values"], title=title)
        elif chart_type == "histogram" and has("x"):
            fig = px.histogram(df, x=spec["x"], color=color, title=title)
        elif chart_type == "box" and has("y"):
            x = spec.get("x") if spec.get("x") in cols else None
            fig = px.box(df, x=x, y=spec["y"], color=color, title=title)
        elif chart_type == "heatmap":
            if spec.get("_correlation"):
                numeric_df = df.select_dtypes("number")
                if numeric_df.shape[1] < 2:
                    return None
                fig = px.imshow(numeric_df.corr(), text_auto=".2f", title=title or "Correlation heatmap")
            elif has("x", "y", "z"):
                pivot = df.pivot_table(index=spec["y"], columns=spec["x"], values=spec["z"], aggfunc="mean")
                fig = px.imshow(pivot, text_auto=".2s", title=title)
            elif spec.get("x") in cols and spec.get("z") in cols:
                small = df[[spec["x"], spec["z"]]].copy()
                small[spec["x"]] = small[spec["x"]].astype(str)
                pivot = small.groupby(spec["x"], dropna=False)[spec["z"]].mean().to_frame(name=spec["z"])
                fig = px.imshow(pivot, text_auto=".2s", title=title)
            else:
                return None
        elif chart_type == "treemap" and has("names", "values"):
            fig = px.treemap(df, path=[spec["names"]], values=spec["values"], title=title)
        else:
            return None
    except Exception:
        return None

    fig.update_layout(title_x=0.02)
    return fig


def build_manual_chart(df: pd.DataFrame, chart_type: str, title: str = None):
    """Convenience wrapper: infer columns for a user-picked type, then render."""
    spec = infer_spec_for_type(df, chart_type)
    if not spec:
        return None
    return build_chart(df, chart_type, spec, title=title)


def auto_spec_from_intent(df: pd.DataFrame, intent: dict | None = None, requested_type: str | None = None) -> dict:
    """Build a chart spec locally from the already executed result.

    This avoids another LLM/JSON call and ensures charts are produced only when
    the user explicitly requests one.
    """
    intent = intent or {}
    numeric, datetime_, categorical = _cols_by_kind(df)
    cols = list(df.columns)
    chart_type = requested_type
    # Explicit user choice always wins. Otherwise derive the visualization from
    # the business question, not merely from the dataframe shape.
    if not chart_type:
        if intent.get("trend") and (datetime_ or categorical) and numeric:
            chart_type = "line"
        elif intent.get("ranking"):
            chart_type = "bar"
        elif intent.get("comparison") and len(categorical) >= 1 and numeric:
            chart_type = "bar"
        elif intent.get("driver_analysis") or intent.get("recommendation") or intent.get("analysis_type") == "diagnostic":
            chart_type = "bar"
        elif len(numeric) >= 2:
            chart_type = "scatter"
        elif categorical and numeric:
            chart_type = "bar"
        else:
            chart_type = "bar"

    spec = infer_spec_for_type(df, chart_type)
    if not spec:
        return {"type": "none"}

    # Prefer business-relevant fields when present.
    def pick(patterns, pool):
        for p in patterns:
            for c in pool:
                if p in c.casefold():
                    return c
        return pool[0] if pool else None

    # Trend charts must use time on X and the sales/revenue metric on Y.
    # This is deliberately handled locally so a user request such as
    # "show sales trend chart" cannot accidentally select a segment column
    # or an impact score as the axis.
    if chart_type in ("line", "area") and intent.get("trend") and (datetime_ or categorical) and numeric:
        x = pick(["date", "datetime", "timestamp", "time", "month", "quarter", "year"], datetime_ + categorical)
        y = pick(["sales", "revenue", "total_sales", "segment_sales", "amount", "value", "orders", "units"], numeric)
    else:
        x = pick(["segment", "region", "state", "retailer", "product", "category", "sales_method", "channel", "date", "month", "quarter", "year"], categorical + datetime_)
        if intent.get("driver_analysis") or intent.get("recommendation"):
            y = pick(["impact_score", "segment_sales", "total_sales", "sales", "revenue", "difference", "pct_difference", "orders", "units"], numeric)
        else:
            y = pick(["total_sales", "segment_sales", "sales", "revenue", "amount", "value", "orders", "units", "impact_score", "difference", "pct_difference"], numeric)
    if chart_type in ("bar", "line", "area") and x and y:
        spec["x"], spec["y"] = x, y
    if chart_type == "pie":
        spec["names"], spec["values"] = x, y
    if chart_type == "treemap":
        spec["names"], spec["values"] = x, y

    if chart_type == "line" and intent.get("trend"):
        spec["title"] = "Sales Trend Over Time"
    elif chart_type == "bar" and intent.get("ranking"):
        spec["title"] = "Sales Ranking"
    elif chart_type == "bar" and intent.get("comparison"):
        spec["title"] = "Sales Comparison"
    elif chart_type == "bar" and (intent.get("driver_analysis") or intent.get("recommendation")):
        spec["title"] = "Sales Underperformance by Key Business Factor"
    elif chart_type == "scatter":
        spec["title"] = "Sales Relationship Analysis"
    elif chart_type == "heatmap":
        spec["title"] = "Sales Comparison Heatmap"
    elif chart_type == "pie":
        spec["title"] = "Sales Contribution"
    elif chart_type == "treemap":
        spec["title"] = "Sales Contribution by Segment"
    elif chart_type == "histogram":
        spec["title"] = "Sales Distribution"
    elif chart_type == "box":
        spec["title"] = "Sales Distribution by Segment"
    elif chart_type == "area":
        spec["title"] = "Sales Trend Over Time"
    return {"type": chart_type, **spec}
