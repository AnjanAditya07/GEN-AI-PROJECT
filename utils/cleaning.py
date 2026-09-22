"""
Data ingestion & robust cleaning layer.

Accepts CSV or Excel (any sheet count) and outputs standardized CSV files.
The cleaner is intentionally conservative: it fixes formatting/type
inconsistencies without guessing missing business values.
"""

import os
import re
import json
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd


# Common representations of missing data.
NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil",
    "-", "--", "unknown", "not available", "not_applicable",
    "not applicable", "missing", "?", "#n/a", "#na",
}

TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
FALSE_TOKENS = {"false", "f", "no", "n", "0"}

CURRENCY_RE = re.compile(
    r"^\s*[\(\+\-]?\s*(?:[$€£₹]|usd|eur|gbp|inr)\s*.*\s*[\)]?\s*$",
    re.IGNORECASE,
)


def load_any_format(filepath: str):
    """Load CSV or Excel and return {sheet_name: DataFrame}."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        # Try common encodings and separators.
        last_error = None
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                return {"Sheet1": pd.read_csv(
                    filepath,
                    encoding=enc,
                    sep=None,
                    engine="python",
                    skip_blank_lines=False,
                )}
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as exc:
                last_error = exc
        raise ValueError(f"Could not read CSV file: {filepath}") from last_error

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, sheet_name=None, engine=None)

    raise ValueError(
        f"Unsupported file type: {ext}. Only .csv, .xlsx, .xls are supported."
    )


def _normalize_text(value):
    """Normalize Unicode, whitespace, invisible characters and null tokens."""
    if pd.isna(value):
        return np.nan

    if not isinstance(value, str):
        return value

    # Normalize Unicode variants (e.g. full-width characters).
    value = unicodedata.normalize("NFKC", value)

    # Remove zero-width/invisible characters.
    value = re.sub(r"[\u200b-\u200d\ufeff]", "", value)

    # Normalize all whitespace to one regular space.
    value = re.sub(r"\s+", " ", value).strip()

    if value.lower() in NULL_TOKENS:
        return np.nan

    return value


def _clean_column_names(df):
    """Standardize column names and guarantee uniqueness."""
    new_cols = []
    seen = {}

    for col in df.columns:
        c = _normalize_text(str(col))
        if pd.isna(c) or not str(c).strip():
            c = "unnamed_col"
        else:
            c = str(c).strip().lower()
            c = re.sub(r"[^\w]+", "_", c, flags=re.UNICODE)
            c = re.sub(r"_+", "_", c).strip("_") or "unnamed_col"

        base = c
        count = seen.get(base, 0)
        if count:
            c = f"{base}_{count + 1}"
        seen[base] = count + 1
        new_cols.append(c)

    df.columns = new_cols
    return df


def _coerce_excel_dates(series):
    """Convert Excel date serials when the values strongly resemble dates."""
    if not pd.api.types.is_numeric_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return series

    # Conservative Excel serial-date range.
    in_range = non_null.between(20000, 60000).mean()
    if in_range > 0.8:
        try:
            return pd.to_datetime(series, unit="D", origin="1899-12-30")
        except (ValueError, TypeError, OverflowError):
            pass

    return series


def _numeric_candidate(series):
    """
    Convert common numeric representations:
      1,234 | $1,234.50 | ₹1,234 | (1,234.50) | 25%
    Returns (converted_series, is_percent, success_ratio).
    """
    s = series.astype("string").str.strip()

    # Do not destroy values that look like ordinary identifiers.
    cleaned = s.str.replace("\u00a0", " ", regex=False)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(r"(?i)(usd|eur|gbp|inr)\s*", "", regex=True)
    cleaned = cleaned.str.replace(r"[$€£₹]", "", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(r"^\s*\+\s*", "", regex=True)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    non_null = s.notna()

    if non_null.sum() == 0:
        return series, False, 0.0

    success = float(numeric[non_null].notna().mean())
    is_percent = bool(s[non_null].str.contains("%", regex=False).any())

    return numeric, is_percent, success


def _looks_like_identifier(series):
    """
    Avoid converting columns such as ZIP codes, employee IDs or product codes
    merely because they contain digits.
    """
    name = str(series.name).lower()
    identifier_words = (
        "id", "code", "zip", "pin", "phone", "mobile", "postal",
        "sku", "account", "invoice", "order", "customer", "employee"
    )
    if any(word in name.split("_") for word in identifier_words):
        return True

    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return False

    # Leading-zero values are usually identifiers.
    leading_zero_ratio = s.str.match(r"^0\d+$").mean()
    return leading_zero_ratio > 0.2


def _canonicalize_categorical_values(series, log):
    """
    Fix case/spacing variants without forcing arbitrary title casing.

    Example:
      'New York', ' new york ', 'NEW YORK'
    become the most frequently occurring representation: 'New York'.
    """
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return series

    original = series.copy()
    s = series.map(_normalize_text)

    non_null = s.dropna().astype(str)
    if non_null.empty:
        return s

    # Canonical key ignores case and repeated whitespace.
    keys = non_null.str.casefold().str.replace(r"\s+", " ", regex=True)

    # Select the most frequent original spelling for each normalized key.
    counts = (
        pd.DataFrame({"value": non_null.values, "key": keys.values})
        .groupby(["key", "value"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["key", "count"], ascending=[True, False])
    )
    canonical = (
        counts.drop_duplicates("key")
        .set_index("key")["value"]
        .to_dict()
    )

    result = s.map(
        lambda x: canonical.get(
            str(x).casefold().replace("  ", " ") if pd.notna(x) else x,
            x,
        )
    )

    changes = int((original.astype("string") != result.astype("string")).fillna(False).sum())
    if changes:
        log.append(
            f"Column '{series.name}': normalized {changes} inconsistent text value(s)"
        )

    return result


def _infer_and_coerce_types(df, log):
    """Normalize values and infer numeric/date/boolean types conservatively."""
    for col in df.columns:
        original_dtype = df[col].dtype
        series = df[col]

        # First normalize strings and missing-value spellings.
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            series = series.map(_normalize_text)

            # Boolean detection.
            non_null = series.dropna().astype(str).str.casefold()
            if not non_null.empty and non_null.isin(TRUE_TOKENS | FALSE_TOKENS).mean() >= 0.95:
                df[col] = series.map(
                    lambda x: (
                        True if pd.notna(x) and str(x).casefold() in TRUE_TOKENS
                        else False if pd.notna(x) and str(x).casefold() in FALSE_TOKENS
                        else np.nan
                    )
                ).astype("boolean")
                log.append(f"Column '{col}': standardized boolean values")
                continue

            # Numeric detection. Keep identifier-like fields as text.
            numeric, is_percent, success = _numeric_candidate(series)
            if success >= 0.90 and not _looks_like_identifier(series):
                df[col] = numeric
                if is_percent:
                    # Preserve the user's percentage convention as numeric percent
                    # (e.g. 25% -> 25), rather than silently changing its scale.
                    log.append(f"Column '{col}': standardized percentage values")
                else:
                    log.append(f"Column '{col}': standardized numeric values")
                continue

            # Date detection. Only convert when strongly supported.
            sample = series.dropna().astype(str)
            if not sample.empty:
                date_attempt = pd.to_datetime(
                    sample, errors="coerce", format="mixed"
                )
                if date_attempt.notna().mean() >= 0.90:
                    parsed = pd.to_datetime(
                        series, errors="coerce", format="mixed"
                    )
                    df[col] = parsed
                    log.append(f"Column '{col}': standardized date/time values")
                    continue

            # Categorical/text consistency.
            df[col] = _canonicalize_categorical_values(series, log)

        else:
            # Excel serial dates and native datetime columns.
            coerced = _coerce_excel_dates(series)
            if not coerced.equals(series):
                df[col] = coerced
                log.append(f"Column '{col}': converted Excel serial numbers to dates")

    return df


def _remove_wholly_empty(df, log):
    """Remove rows/columns that contain no usable information."""
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log.append(f"Dropped {len(empty_cols)} fully empty column(s): {empty_cols}")

    before = len(df)
    df = df.dropna(how="all")
    if before != len(df):
        log.append(f"Dropped {before - len(df)} fully empty row(s)")

    return df


def clean_dataframe(df, source_name=""):
    """
    Comprehensive, BI-safe cleaning.

    Fixes:
      - empty rows/columns
      - column-name inconsistencies
      - Unicode/invisible-character issues
      - whitespace inconsistencies
      - common missing-value spellings
      - duplicate rows (including duplicates exposed by normalization)
      - case/spacing variants in text categories
      - boolean variants
      - numeric/currency/negative/percentage formatting
      - Excel serial dates
      - mixed date/time representations
      - identifier protection against unsafe numeric coercion

    Missing business values are NOT guessed or imputed.
    """
    df = df.copy()
    log = []
    rows_before = len(df)
    cols_before = len(df.columns)

    # Track structural removals separately so the UI/report can distinguish
    # blank rows from duplicate rows.
    empty_rows_before = len(df)
    df = df.dropna(how="all")
    blank_rows_removed_initial = empty_rows_before - len(df)
    if blank_rows_removed_initial:
        log.append(f"Dropped {blank_rows_removed_initial} fully empty row(s)")

    # 1. Remove empty structure.
    df = _remove_wholly_empty(df, log)

    # 2. Standardize column names.
    old_cols = list(df.columns)
    df = _clean_column_names(df)
    renamed = {
        str(old): new
        for old, new in zip(old_cols, df.columns)
        if str(old) != str(new)
    }
    if renamed:
        log.append(
            f"Standardized {len(renamed)} column name(s): "
            f"{dict(list(renamed.items())[:5])}"
        )

    # 3. Normalize values and infer correct types.
    df = _infer_and_coerce_types(df, log)

    # 4. Remove duplicate rows AFTER normalization so formatted duplicates
    #    such as 'New York' vs ' new york ' are caught.
    dupes = int(df.duplicated().sum())
    if dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dupes} duplicate row(s)")

    # 5. Final empty cleanup in case normalization created empty/null rows.
    df = _remove_wholly_empty(df, log)

    # 6. Missing-value report (informational only).
    null_summary = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if int(df[col].isna().sum()) > 0
    }
    if null_summary:
        log.append(f"Missing values detected (not guessed/imputed): {null_summary}")

    rows_out = len(df)
    report = {
        "source": source_name,
        "timestamp": datetime.now().isoformat(),
        "rows_in": rows_before,
        "rows_out": rows_out,
        "rows_removed_total": rows_before - rows_out,
        "blank_rows_removed": blank_rows_removed_initial,
        "duplicate_rows_removed": dupes,
        "columns_in": cols_before,
        "columns_out": list(df.columns),
        "columns_standardized": len(renamed),
        "null_summary": null_summary,
        "missing_cells": int(sum(null_summary.values())),
        "actions": log,
    }

    return df, report


def process_upload(filepath, output_dir="data/cleaned", log_dir="data/logs"):
    """
    Main entry point. Takes CSV/Excel and writes one cleaned CSV per sheet.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    base_name = re.sub(r"[^\w\-]+", "_", base_name)

    sheets = load_any_format(filepath)
    results = []

    for sheet_name, raw_df in sheets.items():
        clean_name = re.sub(r"[^\w\-]+", "_", str(sheet_name))
        cleaned_df, report = clean_dataframe(
            raw_df,
            source_name=f"{base_name} / {sheet_name}",
        )

        if len(sheets) > 1:
            csv_filename = f"{base_name}__{clean_name}__cleaned.csv"
        else:
            csv_filename = f"{base_name}__cleaned.csv"

        csv_path = os.path.join(output_dir, csv_filename)
        cleaned_df.to_csv(csv_path, index=False, encoding="utf-8")

        log_path = os.path.join(
            log_dir,
            csv_filename.replace(".csv", "__log.json"),
        )
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        results.append({
            "table_name": clean_name if len(sheets) > 1 else base_name,
            "csv_path": csv_path,
            "log_path": log_path,
            "report": report,
            "row_count": len(cleaned_df),
            "columns": list(cleaned_df.columns),
        })

    return results
