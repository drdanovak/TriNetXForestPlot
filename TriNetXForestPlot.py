import csv
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import streamlit as st

plt.style.use("default")
st.set_page_config(layout="wide")
st.title("🌲 Novak's TriNetX Forest Plot Generator")

required_cols = [
    "Outcome",
    "Risk, Odds, or Hazard Ratio",
    "Effect Size (Cohen's d, approx.)",
    "Lower CI",
    "Upper CI",
]
optional_cols = ["p"]


# =========================
# Core utilities
# =========================
def compute_cohens_d(rr):
    try:
        if pd.isnull(rr):
            return np.nan
        val = float(rr)
        if val <= 0:
            return np.nan
        return np.log(val) * (np.sqrt(3) / np.pi)
    except Exception:
        return np.nan


def clean_cell(value):
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    text = text.strip('"').strip()
    return text


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "", clean_cell(value).lower())


def parse_float(value):
    text = clean_cell(value).replace(",", "")
    if text in {"", " ", "nan", "None"}:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    # Strip common p-value comparators while preserving the numeric value.
    text = re.sub(r"^(p\s*[<=>]\s*)", "", text, flags=re.IGNORECASE)
    text = text.lstrip("<>=≤≥ ")
    try:
        return float(text)
    except Exception:
        return None


def parse_p_value(value):
    """Parse p-values written as .021, 0.021, <.001, p=.03, or p < 0.001."""
    text = clean_cell(value)
    if not text:
        return None
    text = text.replace(",", "").replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"^p\s*[-_ ]*value\s*[:=]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^p\s*[:=<>≤≥]?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().lstrip("<>=≤≥ ")
    if text.startswith("."):
        text = "0" + text
    try:
        val = float(text)
        if 0 <= val <= 1:
            return val
    except Exception:
        pass
    match = re.search(r"(?<!\d)(?:0?\.\d+|1(?:\.0+)?)(?!\d)", text)
    if match:
        try:
            val = float(match.group(0))
            if 0 <= val <= 1:
                return val
        except Exception:
            return None
    return None


def parse_ci_bounds(value):
    """Return lower/upper CI bounds from cells like '0.56–0.96', '(0.56, 0.96)', or '95% CI: 0.56 - 0.96'."""
    text = clean_cell(value)
    if not text:
        return None, None
    # Remove a leading 95% if the CI label was included in the value cell.
    text_for_numbers = re.sub(r"\b95\s*%", "", text, flags=re.IGNORECASE)
    # In ratio tables, a dash/en dash between two numbers is almost always a range
    # delimiter, not a negative sign. Convert that delimiter before numeric parsing.
    text_for_numbers = re.sub(r"(?<=\d)\s*[\-–—−]\s*(?=\d|\.)", ",", text_for_numbers)
    text_for_numbers = text_for_numbers.replace("−", "-").replace("–", ",").replace("—", ",")
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text_for_numbers)
    parsed = []
    for n in numbers:
        try:
            parsed.append(float("0" + n if n.startswith(".") else n))
        except Exception:
            pass
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    return None, None


def next_nonblank_row(rows, start_idx):
    for idx in range(start_idx, len(rows)):
        if any(clean_cell(cell) for cell in rows[idx]):
            return idx
    return None


# =========================
# TriNetX parsing helpers
# =========================
def extract_section_triplet(rows, section_name):
    """
    Extract point estimate, confidence interval, and p value from a TriNetX
    Risk Ratio, Odds Ratio, or Hazard Ratio section.

    This is intentionally defensive because TriNetX exports vary by module and
    file type. Some files store lower and upper CI limits in separate columns,
    while others store the 95% CI in a single text cell such as "0.56–0.96".
    Likewise, the p-value may be labeled "p", "p-value", "p value", or may be
    the final numeric value in the section row.
    """

    section_norm = normalize_text(section_name)

    def is_section_row(row):
        first = clean_cell(next((cell for cell in row if clean_cell(cell)), ""))
        return normalize_text(first) == section_norm

    def header_role(header):
        h = normalize_text(header)
        if h in {"p", "pvalue", "pval", "probability"} or "pvalue" in h:
            return "p"
        if ("lower" in h or "low" in h or "lcl" in h) and ("ci" in h or "confidence" in h or "95" in h or "limit" in h):
            return "lower"
        if ("upper" in h or "up" in h or "ucl" in h) and ("ci" in h or "confidence" in h or "95" in h or "limit" in h):
            return "upper"
        if ("95" in h or "ci" in h or "confidenceinterval" in h or "confidence" in h) and "lower" not in h and "upper" not in h:
            return "ci"
        if h in {section_norm, "value", "estimate", "pointestimate", "ratio"} or section_norm in h:
            return "estimate"
        if "z" in h and len(h) <= 12:
            return "z"
        return "other"

    for i, row in enumerate(rows):
        if not is_section_row(row):
            continue

        header_idx = next_nonblank_row(rows, i + 1)
        data_idx = next_nonblank_row(rows, (header_idx + 1) if header_idx is not None else i + 1)
        if data_idx is None:
            return None

        headers = [clean_cell(x) for x in rows[header_idx]] if header_idx is not None else []
        values = [clean_cell(x) for x in rows[data_idx]]

        estimate = lower = upper = p_value = None

        # Header-driven extraction first. This prevents z statistics or p-values
        # from being mistaken for confidence limits when CI is provided as text.
        for h, v in zip(headers, values):
            role = header_role(h)
            if role == "estimate" and estimate is None:
                estimate = parse_float(v)
            elif role == "lower" and lower is None:
                lower = parse_float(v)
            elif role == "upper" and upper is None:
                upper = parse_float(v)
            elif role == "ci" and (lower is None or upper is None):
                parsed_lower, parsed_upper = parse_ci_bounds(v)
                if parsed_lower is not None and parsed_upper is not None:
                    lower, upper = parsed_lower, parsed_upper
            elif role == "p" and p_value is None:
                p_value = parse_p_value(v)

        # If the estimate was not labeled, use the first numeric field in the
        # data row that is not obviously a p-value-only string.
        if estimate is None:
            for v in values:
                parsed = parse_float(v)
                if parsed is not None:
                    estimate = parsed
                    break

        # If the CI was stored as a single unlabelled cell, find the first cell
        # containing two plausible numeric bounds.
        if lower is None or upper is None:
            for v in values:
                parsed_lower, parsed_upper = parse_ci_bounds(v)
                if parsed_lower is not None and parsed_upper is not None:
                    # Avoid treating an estimate/p-value pair as a CI unless the
                    # cell contains a visible range delimiter or parentheses.
                    if any(token in clean_cell(v) for token in ["-", "–", "—", ",", "(", ")", "["]):
                        lower, upper = parsed_lower, parsed_upper
                        break

        # Numeric fallback for the common wide-table format:
        # estimate, lower CI, upper CI, z, p OR estimate, lower CI, upper CI, p.
        numeric_values = [parse_float(cell) for cell in values]
        numeric_values = [x for x in numeric_values if x is not None]
        if estimate is None and len(numeric_values) >= 1:
            estimate = numeric_values[0]
        if (lower is None or upper is None) and len(numeric_values) >= 3:
            lower, upper = numeric_values[1], numeric_values[2]

        # p-value fallback. Prefer a header-labeled p-value; otherwise use the
        # final numeric field when it is a plausible p-value and not one of the
        # CI bounds. This catches TriNetX Risk Ratio and Odds Ratio sections.
        if p_value is None:
            for h, v in zip(headers, values):
                if header_role(h) == "p":
                    p_value = parse_p_value(v)
                    break
        if p_value is None and len(numeric_values) >= 4:
            candidate = numeric_values[-1]
            if 0 <= candidate <= 1:
                p_value = candidate
        if p_value is None:
            # Last resort: look for cells explicitly written as p=.021 or p < .001.
            for v in values:
                if re.search(r"\bp\s*[-_ ]*(value)?\s*[<=>:]", clean_cell(v), flags=re.IGNORECASE):
                    p_value = parse_p_value(v)
                    if p_value is not None:
                        break

        if estimate is not None and lower is not None and upper is not None:
            return {
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "p": p_value if p_value is not None else np.nan,
            }

    return None

def extract_section_value_map(rows, section_name):
    for i, row in enumerate(rows):
        first = clean_cell(next((cell for cell in row if clean_cell(cell)), ""))
        if first == section_name:
            header_idx = next_nonblank_row(rows, i + 1)
            data_idx = next_nonblank_row(rows, (header_idx + 1) if header_idx is not None else i + 1)
            if header_idx is None or data_idx is None:
                return None
            headers = [clean_cell(x) for x in rows[header_idx]]
            values = [clean_cell(x) for x in rows[data_idx]]
            out = {}
            for h, v in zip(headers, values):
                if h:
                    out[h] = v
            return out
    return None


def extract_title_from_rows(rows):
    for row in rows[:8]:
        first = clean_cell(next((cell for cell in row if clean_cell(cell)), ""))
        if first and first != "Generated by TriNetX":
            return first
    return ""


def detect_trinetx_table_type(title, raw_text):
    title_clean = clean_cell(title)
    raw_text = raw_text or ""
    if "Kaplan-Meier Table" in title_clean or "Kaplan-Meier Table" in raw_text:
        return "Kaplan-Meier"
    if "Measures of Association Table" in title_clean or "Measures of Association Table" in raw_text:
        return "Measures of Association"
    if "Hazard Ratio" in raw_text and "Log-Rank Test" in raw_text:
        return "Kaplan-Meier"
    if any(x in raw_text for x in ["Risk Ratio", "Odds Ratio"]):
        return "Measures of Association"
    return "TriNetX"


def clean_title(title):
    title = clean_cell(title)
    title = re.sub(r"\s*Measures of Association Table\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*Kaplan[- ]Meier Table\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" -_")
    return title


def clean_filename(name):
    stem = Path(name).stem
    stem = re.sub(r"(?i)_result_[a-z0-9]+_moa_table$", "", stem)
    stem = re.sub(r"(?i)_result_[a-z0-9]+_km_table$", "", stem)
    stem = re.sub(r"(?i)_moa_table$", "", stem)
    stem = re.sub(r"(?i)_km_table$", "", stem)
    stem = re.sub(r"(?i)_kaplan[-_ ]meier_table$", "", stem)
    stem = re.sub(r"(?i)_table$", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def looks_generic(label):
    text = (label or "").strip().lower()
    return bool(re.match(r"^outcome\s*\d+(\s+result.*)?$", text))


def choose_outcome_label(title, filename):
    cleaned_title = clean_title(title)
    cleaned_filename = clean_filename(filename)
    if cleaned_title and not looks_generic(cleaned_title):
        return cleaned_title
    if cleaned_filename:
        return cleaned_filename
    return cleaned_title or "Outcome"


def parse_trinetx_csv_text(file_bytes):
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    rows = list(csv.reader(io.StringIO(text)))
    return rows, text


def parse_trinetx_excel_rows(file_bytes):
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=str)
    rows = raw.fillna("").values.tolist()
    joined = "\n".join([",".join([clean_cell(cell) for cell in row]) for row in rows])
    return rows, joined


def parse_trinetx_effect_rows(rows, filename, raw_text):
    title = extract_title_from_rows(rows)
    table_type = detect_trinetx_table_type(title, raw_text)

    risk_ratio = extract_section_triplet(rows, "Risk Ratio")
    odds_ratio = extract_section_triplet(rows, "Odds Ratio")
    hazard_ratio = extract_section_triplet(rows, "Hazard Ratio")

    if not any([risk_ratio, odds_ratio, hazard_ratio]):
        return None

    log_rank = extract_section_value_map(rows, "Log-Rank Test") if table_type == "Kaplan-Meier" else None
    proportionality = extract_section_value_map(rows, "Proportionality") if table_type == "Kaplan-Meier" else None

    return {
        "Outcome": choose_outcome_label(title, filename),
        "Original Title": clean_title(title),
        "Source File": filename,
        "TriNetX Table Type": table_type,
        "Risk Ratio": risk_ratio["estimate"] if risk_ratio else np.nan,
        "RR Lower CI": risk_ratio["lower"] if risk_ratio else np.nan,
        "RR Upper CI": risk_ratio["upper"] if risk_ratio else np.nan,
        "RR p": risk_ratio.get("p", np.nan) if risk_ratio else np.nan,
        "Odds Ratio": odds_ratio["estimate"] if odds_ratio else np.nan,
        "OR Lower CI": odds_ratio["lower"] if odds_ratio else np.nan,
        "OR Upper CI": odds_ratio["upper"] if odds_ratio else np.nan,
        "OR p": odds_ratio.get("p", np.nan) if odds_ratio else np.nan,
        "Hazard Ratio": hazard_ratio["estimate"] if hazard_ratio else np.nan,
        "HR Lower CI": hazard_ratio["lower"] if hazard_ratio else np.nan,
        "HR Upper CI": hazard_ratio["upper"] if hazard_ratio else np.nan,
        "HR p": hazard_ratio.get("p", np.nan) if hazard_ratio else np.nan,
        "Log-Rank p": parse_float(log_rank.get("p")) if log_rank else np.nan,
        "PH Assumption p": parse_float(proportionality.get("p")) if proportionality else np.nan,
    }


# =========================
# Standard table handling
# =========================
def standardize_existing_forest_table(df):
    working = df.copy()

    # Preserve common p-value column names if users upload a preformatted table.
    if "p" not in working.columns:
        p_aliases = {"p", "p value", "p-value", "p_value", "pvalue", "P", "P value", "P-value"}
        for col in list(working.columns):
            if str(col).strip() in p_aliases:
                working = working.rename(columns={col: "p"})
                break

    for col in required_cols + optional_cols:
        if col not in working.columns:
            working[col] = np.nan

    numeric_cols = [
        "Risk, Odds, or Hazard Ratio",
        "Lower CI",
        "Upper CI",
        "Effect Size (Cohen's d, approx.)",
        "p",
    ]
    for col in numeric_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if working["Effect Size (Cohen's d, approx.)"].isna().all():
        working["Effect Size (Cohen's d, approx.)"] = working["Risk, Odds, or Hazard Ratio"].apply(compute_cohens_d)

    return working[required_cols + optional_cols]


# =========================
# Upload detection
# =========================
def looks_like_trinetx_text(raw_text):
    if not raw_text:
        return False
    return "Generated by TriNetX" in raw_text and any(
        x in raw_text for x in ["Risk Ratio", "Odds Ratio", "Hazard Ratio", "Kaplan-Meier Table", "Measures of Association Table"]
    )


def detect_and_load_uploaded_files(uploaded_files):
    parsed_trinetx_rows = []
    parsed_standard_tables = []
    parsing_notes = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        suffix = Path(filename).suffix.lower()

        try:
            if suffix == ".csv":
                rows, raw_text = parse_trinetx_csv_text(file_bytes)
                if looks_like_trinetx_text(raw_text):
                    parsed = parse_trinetx_effect_rows(rows, filename, raw_text)
                    if parsed is not None:
                        parsed_trinetx_rows.append(parsed)
                        continue
                standard_df = pd.read_csv(io.BytesIO(file_bytes))
                parsed_standard_tables.append(standardize_existing_forest_table(standard_df))

            elif suffix in {".xlsx", ".xls"}:
                try:
                    rows, raw_text = parse_trinetx_excel_rows(file_bytes)
                    if looks_like_trinetx_text(raw_text):
                        parsed = parse_trinetx_effect_rows(rows, filename, raw_text)
                        if parsed is not None:
                            parsed_trinetx_rows.append(parsed)
                            continue
                except Exception:
                    pass
                standard_df = pd.read_excel(io.BytesIO(file_bytes))
                parsed_standard_tables.append(standardize_existing_forest_table(standard_df))

            else:
                parsing_notes.append(f"Skipped unsupported file type: {filename}")
        except Exception as e:
            parsing_notes.append(f"Could not parse {filename}: {e}")

    return parsed_trinetx_rows, parsed_standard_tables, parsing_notes


# =========================
# Forest plot table assembly
# =========================
def choose_effect_for_row(row, preferred_measure):
    measure_priority = [preferred_measure]
    for candidate in ["Risk Ratio", "Odds Ratio", "Hazard Ratio"]:
        if candidate not in measure_priority:
            measure_priority.append(candidate)

    for candidate in measure_priority:
        if candidate == "Risk Ratio" and pd.notnull(row.get("Risk Ratio", np.nan)):
            return candidate, row["Risk Ratio"], row["RR Lower CI"], row["RR Upper CI"], row.get("RR p", np.nan)
        if candidate == "Odds Ratio" and pd.notnull(row.get("Odds Ratio", np.nan)):
            return candidate, row["Odds Ratio"], row["OR Lower CI"], row["OR Upper CI"], row.get("OR p", np.nan)
        if candidate == "Hazard Ratio" and pd.notnull(row.get("Hazard Ratio", np.nan)):
            p_value = row.get("HR p", np.nan)
            if pd.isna(p_value):
                p_value = row.get("Log-Rank p", np.nan)
            return candidate, row["Hazard Ratio"], row["HR Lower CI"], row["HR Upper CI"], p_value

    return None, np.nan, np.nan, np.nan, np.nan


def deduplicate_outcome_labels(df):
    df = df.copy()
    counts = df["Outcome"].value_counts(dropna=False)
    duplicate_labels = set(counts[counts > 1].index.tolist())

    def relabel(row):
        label = row["Outcome"]
        if label in duplicate_labels:
            effect_type = row.get("Effect Type")
            table_type = row.get("TriNetX Table Type")
            if pd.notnull(effect_type) and pd.notnull(table_type):
                return f"{label} — {effect_type} ({table_type})"
            if pd.notnull(effect_type):
                return f"{label} — {effect_type}"
        return label

    df["Outcome"] = df.apply(relabel, axis=1)
    return df


def build_plot_table_from_trinetx(parsed_rows, preferred_measure):
    out_rows = []

    for row in parsed_rows:
        chosen_measure, estimate, lower, upper, p_value = choose_effect_for_row(row, preferred_measure)
        out_rows.append({
            "Outcome": row["Outcome"],
            "Risk, Odds, or Hazard Ratio": estimate,
            "Effect Size (Cohen's d, approx.)": compute_cohens_d(estimate),
            "Lower CI": lower,
            "Upper CI": upper,
            "p": p_value,
            "Effect Type": chosen_measure,
            "TriNetX Table Type": row["TriNetX Table Type"],
            "Log-Rank p": row.get("Log-Rank p", np.nan),
            "PH Assumption p": row.get("PH Assumption p", np.nan),
            "Source File": row["Source File"],
        })

    built = pd.DataFrame(out_rows)
    if not built.empty:
        built = deduplicate_outcome_labels(built)
    return built


def infer_ratio_axis_label(df):
    if "Effect Type" in df.columns:
        effect_types = [str(x).strip() for x in df["Effect Type"].dropna().tolist() if str(x).strip()]
        unique_effect_types = list(dict.fromkeys(effect_types))
        if len(unique_effect_types) == 1:
            return unique_effect_types[0]
        if len(unique_effect_types) > 1:
            return "Effect Estimate"
    return st.session_state.get("manual_ratio_label", "Risk Ratio")


def compute_ratio_axis_limits(ci_vals, axis_padding, use_log=False):
    numeric_vals = pd.to_numeric(ci_vals, errors="coerce")
    numeric_vals = numeric_vals.replace([np.inf, -np.inf], np.nan).dropna()
    numeric_vals = numeric_vals[numeric_vals > 0]

    if numeric_vals.empty:
        return 0.5, 1.5

    raw_min = float(min(numeric_vals.min(), 1.0))
    raw_max = float(max(numeric_vals.max(), 1.0))

    if use_log:
        lower_factor = 1.0 / raw_min if raw_min > 0 else raw_max
        upper_factor = raw_max
        factor = max(lower_factor, upper_factor, 1.25)
        factor *= (1 + axis_padding / 100)
        return max(1 / factor, 1e-6), factor

    dist_left = max(1.0 - raw_min, 0.0)
    dist_right = max(raw_max - 1.0, 0.0)
    half_span = max(dist_left, dist_right)
    if half_span == 0:
        half_span = 0.25
    padded_half_span = half_span * (1 + axis_padding / 100)
    auto_min = max(0.0, 1.0 - padded_half_span)
    auto_max = 1.0 + padded_half_span
    if auto_min == auto_max:
        auto_max = auto_min + 1.0
    return auto_min, auto_max



# =========================
# Forest plot / table hybrid helpers
# =========================
def get_row_p_value(row):
    for col in ["p", "P", "p-value", "P-value", "p value", "P value", "Log-Rank p"]:
        if col in row.index:
            value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
            if pd.notnull(value):
                return value
    return np.nan


def format_p_value(value):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".replace("0.", ".", 1)


def format_number(value, decimals=2):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def format_ci(lower, upper, decimals=2):
    lower_text = format_number(lower, decimals)
    upper_text = format_number(upper, decimals)
    if not lower_text or not upper_text:
        return ""
    return f"{lower_text}–{upper_text}"


def infer_significance(effect, lower, upper, p_value, ref_line):
    p_value = pd.to_numeric(pd.Series([p_value]), errors="coerce").iloc[0]
    lower = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
    if pd.notnull(p_value):
        return p_value < 0.05
    if pd.notnull(lower) and pd.notnull(upper):
        return (upper < ref_line) or (lower > ref_line)
    return False


def ratio_column_header(axis_label):
    label = str(axis_label).lower()
    if "risk ratio" in label:
        return "RR"
    if "odds ratio" in label:
        return "OR"
    if "hazard ratio" in label:
        return "HR"
    if "effect estimate" in label:
        return "Estimate"
    return "Estimate"


def build_hybrid_display_rows(df, use_groups=True):
    display_rows = []
    inside_group = False
    for _, row in df.iterrows():
        outcome = clean_cell(row.get("Outcome", ""))
        if not outcome:
            continue
        if use_groups and outcome.startswith("##"):
            display_rows.append({"kind": "header", "label": outcome[2:].strip(), "row": None, "indented": False})
            inside_group = True
        else:
            display_rows.append({"kind": "data", "label": outcome, "row": row, "indented": inside_group})
    return display_rows


def create_forest_table_hybrid(
    df,
    plot_column,
    x_measure,
    x_axis_label,
    ref_line,
    x_min,
    x_max,
    use_groups=True,
    use_log=False,
    show_grid=False,
    plot_title="",
    font_size=12,
    point_size=9,
    line_width=1.8,
    cap_height=0.15,
    header_color="#203F99",
    significant_color="#1F3D99",
    nonsignificant_color="#666666",
):
    display_rows = build_hybrid_display_rows(df, use_groups=use_groups)
    if not display_rows:
        raise ValueError("No rows are available to plot.")

    n_rows = len(display_rows)
    fig_height = max(3.2, 0.46 * n_rows + 1.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.subplots_adjust(left=0.30, right=0.76, top=0.90, bottom=0.18)

    if use_log:
        ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(n_rows - 0.35, -0.90)
    ax.set_yticks([])

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", labelsize=max(font_size - 2, 8), colors="#444444")

    if show_grid:
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    else:
        ax.grid(False)

    ax.axvline(ref_line, color="#333333", linewidth=1.15, zorder=1)

    # Mixed transform: x as an axes fraction, y as data coordinate.
    text_transform = blended_transform_factory(ax.transAxes, ax.transData)
    left_x = -0.64
    right_est_x = 1.08
    right_ci_x = 1.28
    right_p_x = 1.53
    header_y = -0.62
    ratio_header = ratio_column_header(x_axis_label)

    ax.text(right_est_x, header_y, ratio_header, transform=text_transform, ha="center", va="center",
            fontsize=font_size, weight="bold", clip_on=False)
    ax.text(right_ci_x, header_y, "95% CI", transform=text_transform, ha="center", va="center",
            fontsize=font_size, weight="bold", clip_on=False)
    ax.text(right_p_x, header_y, r"$p$", transform=text_transform, ha="center", va="center",
            fontsize=font_size, weight="bold", clip_on=False)

    ci_line_color = "#222222"
    span = x_max - x_min
    for y, item in enumerate(display_rows):
        if item["kind"] == "header":
            # Draw the section bar across the left labels, forest plot, and right table.
            rect = Rectangle(
                (left_x - 0.02, y - 0.38),
                2.25,
                0.76,
                transform=text_transform,
                facecolor=header_color,
                edgecolor=header_color,
                clip_on=False,
                zorder=0,
            )
            ax.add_patch(rect)
            ax.text(left_x, y, item["label"], transform=text_transform, ha="left", va="center",
                    color="white", fontsize=font_size, weight="bold", clip_on=False, zorder=2)
            continue

        row = item["row"]
        label = ("   " + item["label"]) if item["indented"] else item["label"]
        ax.text(left_x, y, label, transform=text_transform, ha="left", va="center",
                fontsize=font_size, color="#222222", clip_on=False)

        if x_measure == "Effect Size (Cohen's d, approx.)":
            effect = pd.to_numeric(pd.Series([row.get("Effect Size (Cohen's d, approx.)", np.nan)]), errors="coerce").iloc[0]
            lci = compute_cohens_d(row.get("Lower CI", np.nan))
            uci = compute_cohens_d(row.get("Upper CI", np.nan))
        else:
            effect = pd.to_numeric(pd.Series([row.get("Risk, Odds, or Hazard Ratio", np.nan)]), errors="coerce").iloc[0]
            lci = pd.to_numeric(pd.Series([row.get("Lower CI", np.nan)]), errors="coerce").iloc[0]
            uci = pd.to_numeric(pd.Series([row.get("Upper CI", np.nan)]), errors="coerce").iloc[0]

        p_value = get_row_p_value(row)
        significant = infer_significance(effect, lci, uci, p_value, ref_line)
        marker_color = significant_color if significant else nonsignificant_color

        if pd.notnull(lci) and pd.notnull(uci):
            clipped_left = max(lci, x_min)
            clipped_right = min(uci, x_max)
            ax.hlines(y, xmin=clipped_left, xmax=clipped_right, color=ci_line_color,
                      linewidth=line_width, zorder=2)
            if lci >= x_min:
                ax.vlines(lci, y - cap_height, y + cap_height, color=ci_line_color,
                          linewidth=line_width, zorder=2)
            else:
                ax.annotate("", xy=(x_min, y), xytext=(x_min + 0.06 * span, y),
                            arrowprops=dict(arrowstyle="<-", color=ci_line_color,
                                            lw=line_width, shrinkA=0, shrinkB=0),
                            zorder=2)
            if uci <= x_max:
                ax.vlines(uci, y - cap_height, y + cap_height, color=ci_line_color,
                          linewidth=line_width, zorder=2)
            else:
                ax.annotate("", xy=(x_max, y), xytext=(x_max - 0.06 * span, y),
                            arrowprops=dict(arrowstyle="->", color=ci_line_color,
                                            lw=line_width, shrinkA=0, shrinkB=0),
                            zorder=2)

        if pd.notnull(effect) and x_min <= effect <= x_max:
            ax.plot(effect, y, marker="s", markersize=point_size, color=marker_color,
                    markeredgecolor=marker_color, zorder=3)

        ax.text(right_est_x, y, format_number(effect, 2), transform=text_transform, ha="center", va="center",
                fontsize=font_size, color="#222222", clip_on=False)
        ax.text(right_ci_x, y, format_ci(lci, uci, 2), transform=text_transform, ha="center", va="center",
                fontsize=font_size, color="#222222", clip_on=False)
        ax.text(right_p_x, y, format_p_value(p_value), transform=text_transform, ha="center", va="center",
                fontsize=font_size, color="#222222", clip_on=False)

    ax.set_xlabel(x_axis_label, fontsize=font_size, weight="bold", labelpad=8)
    if plot_title:
        ax.set_title(plot_title, fontsize=font_size + 2, weight="bold", pad=12)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=significant_color,
               markeredgecolor=significant_color, markersize=8, label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=nonsignificant_color,
               markeredgecolor=nonsignificant_color, markersize=8, label="Non-significant"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(-0.02, -0.36),
              frameon=False, ncol=2, fontsize=max(font_size - 2, 8), handlelength=1.0,
              columnspacing=2.5, handletextpad=0.4)

    return fig

# =========================
# App UI
# =========================
input_mode = st.radio(
    "Select data input method:", ["📤 Upload file(s)", "✍️ Manual entry"], index=0, horizontal=True
)

df = None

if input_mode == "📤 Upload file(s)":
    uploaded_files = st.file_uploader(
        "Upload one normalized forest-plot table or multiple raw TriNetX MOA / Kaplan–Meier summary tables",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        preferred_measure = st.sidebar.selectbox(
            "Preferred TriNetX estimate to plot",
            ["Risk Ratio", "Odds Ratio", "Hazard Ratio"],
            index=0,
            help="For MOA files, the app will prefer this estimate when available. KM tables contribute Hazard Ratios automatically.",
        )
        st.session_state["manual_ratio_label"] = preferred_measure

        parsed_trinetx_rows, parsed_standard_tables, parsing_notes = detect_and_load_uploaded_files(uploaded_files)

        assembled_tables = []
        if parsed_trinetx_rows:
            trinetx_df = build_plot_table_from_trinetx(parsed_trinetx_rows, preferred_measure)
            assembled_tables.append(
                trinetx_df[required_cols + optional_cols + ["Effect Type", "TriNetX Table Type", "Log-Rank p", "PH Assumption p", "Source File"]]
            )
        if parsed_standard_tables:
            for table in parsed_standard_tables:
                table = table.copy()
                if "p" not in table.columns:
                    table["p"] = np.nan
                table["Effect Type"] = np.nan
                table["TriNetX Table Type"] = np.nan
                table["Log-Rank p"] = np.nan
                table["PH Assumption p"] = np.nan
                table["Source File"] = np.nan
                assembled_tables.append(
                    table[required_cols + optional_cols + ["Effect Type", "TriNetX Table Type", "Log-Rank p", "PH Assumption p", "Source File"]]
                )

        if assembled_tables:
            combined_df = pd.concat(assembled_tables, ignore_index=True)
            st.subheader("Parsed data preview")
            st.caption(
                "You can edit outcome labels, reorder rows, or insert section headers using ## before a label. "
                "MOA uploads now try to extract the p-value attached to the selected Risk Ratio or Odds Ratio; KM uploads populate hazard ratios and preserve log-rank / proportionality p-values as metadata."
            )
            editable_cols = required_cols + optional_cols + ["Effect Type", "TriNetX Table Type", "Log-Rank p", "PH Assumption p", "Source File"]
            edited_df = st.data_editor(
                combined_df[editable_cols],
                num_rows="dynamic",
                use_container_width=True,
                key="uploaded_table_editor",
                column_config={
                    "Effect Size (Cohen's d, approx.)": st.column_config.NumberColumn(
                        "Effect Size (Cohen's d, approx.)",
                        disabled=True,
                        help="Auto-calculated as ln(RR/OR/HR) × sqrt(3)/π",
                    ),
                    "p": st.column_config.NumberColumn("p", format="%.4g"),
                    "Effect Type": st.column_config.TextColumn(disabled=True),
                    "TriNetX Table Type": st.column_config.TextColumn(disabled=True),
                    "Log-Rank p": st.column_config.NumberColumn(disabled=True, format="%.4g"),
                    "PH Assumption p": st.column_config.NumberColumn(disabled=True, format="%.4g"),
                    "Source File": st.column_config.TextColumn(disabled=True),
                },
            )
            edited_df["Effect Size (Cohen's d, approx.)"] = edited_df["Risk, Odds, or Hazard Ratio"].apply(compute_cohens_d)
            df = edited_df

            summary_bits = []
            if parsed_trinetx_rows:
                n_km = sum(1 for r in parsed_trinetx_rows if r.get("TriNetX Table Type") == "Kaplan-Meier")
                n_moa = sum(1 for r in parsed_trinetx_rows if r.get("TriNetX Table Type") == "Measures of Association")
                if n_moa:
                    summary_bits.append(f"{n_moa} MOA file(s)")
                if n_km:
                    summary_bits.append(f"{n_km} KM file(s)")
            if parsed_standard_tables:
                summary_bits.append(f"{len(parsed_standard_tables)} preformatted table(s)")
            if summary_bits:
                st.success("Parsed: " + ", ".join(summary_bits))
        else:
            st.error("No usable forest-plot data could be parsed from the uploaded files.")

        if parsing_notes:
            with st.expander("Parsing notes"):
                for note in parsing_notes:
                    st.write(f"- {note}")
else:
    default_data = pd.DataFrame({
        "Outcome": ["## Cardiovascular", "Hypertension", "Stroke", "## Metabolic", "Diabetes", "Obesity"],
        "Risk, Odds, or Hazard Ratio": [None, 1.5, 1.2, None, 0.85, 1.2],
        "Lower CI": [None, 1.2, 1.0, None, 0.7, 1.0],
        "Upper CI": [None, 1.8, 1.5, None, 1.0, 1.4],
        "p": [None, 0.002, 0.049, None, 0.073, 0.041],
    })
    default_data["Effect Size (Cohen's d, approx.)"] = default_data["Risk, Odds, or Hazard Ratio"].apply(compute_cohens_d)
    default_data = default_data[required_cols + optional_cols]

    if "manual_table" not in st.session_state:
        st.session_state.manual_table = default_data.copy()

    _, col2 = st.columns([2, 1])
    with col2:
        if st.button("🧹 Clear Table"):
            st.session_state.manual_table = pd.DataFrame({col: [None] * 6 for col in required_cols + optional_cols})

    manual_df = st.session_state.manual_table.copy()
    manual_df["Effect Size (Cohen's d, approx.)"] = manual_df["Risk, Odds, or Hazard Ratio"].apply(compute_cohens_d)
    if "p" not in manual_df.columns:
        manual_df["p"] = np.nan
    manual_df = manual_df[required_cols + optional_cols]
    st.session_state.manual_table = st.data_editor(
        manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_input_table",
        column_config={
            "Effect Size (Cohen's d, approx.)": st.column_config.NumberColumn(
                "Effect Size (Cohen's d, approx.)",
                disabled=True,
                help="Auto-calculated as ln(RR/OR/HR) × sqrt(3)/π",
            ),
            "p": st.column_config.NumberColumn("p", format="%.4g")
        },
    )
    df = st.session_state.manual_table


# =========================
# Plotting
# =========================
if df is not None:
    st.sidebar.header("⚙️ Basic Plot Settings")

    x_measure = st.sidebar.radio(
        "Plot on X-axis",
        ("Effect Size (Cohen's d, approx.)", "Risk, Odds, or Hazard Ratio"),
        index=1,
    )

    if x_measure == "Risk, Odds, or Hazard Ratio" and "Effect Type" not in df.columns:
        default_ratio_label_index = ["Risk Ratio", "Odds Ratio", "Hazard Ratio"].index(
            st.session_state.get("manual_ratio_label", "Risk Ratio")
        ) if st.session_state.get("manual_ratio_label", "Risk Ratio") in ["Risk Ratio", "Odds Ratio", "Hazard Ratio"] else 0
        st.session_state["manual_ratio_label"] = st.sidebar.selectbox(
            "Label ratio axis as",
            ["Risk Ratio", "Odds Ratio", "Hazard Ratio"],
            index=default_ratio_label_index,
            help="Use this when the uploaded table does not include an explicit effect type column.",
        )

    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = False  # Right-side numerical annotations are always shown in hybrid layout
    use_groups = st.sidebar.checkbox("Treat rows starting with '##' as section headers", value=True)

    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        table_header_color = st.color_picker("Section Header Color", "#203F99") if color_scheme == "Color" else "#111111"
        significant_color = st.color_picker("Significant Marker Color", "#1F3D99") if color_scheme == "Color" else "#111111"
        nonsignificant_color = st.color_picker("Non-significant Marker Color", "#666666")
        point_size = st.slider("Marker Size", 6, 20, 9)
        line_width = st.slider("CI Line Width", 1, 4, 2)
        font_size = st.slider("Font Size", 10, 20, 12)
        label_offset = st.slider("Label Horizontal Offset", 0.01, 0.3, 0.05)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)
        axis_padding = st.slider("X-axis Padding (%)", 2, 40, 10)
        manual_x_axis = st.checkbox(
            "Specify X-axis range manually",
            value=False,
            help="Turn this on to set the plotted X-axis minimum and maximum yourself instead of using automatic padding.",
        )
        y_axis_padding = st.slider("Y-axis Padding (Rows)", 0.0, 5.0, 1.0, step=0.5)
        cap_height = st.slider("Tick Height (for CI ends)", 0.05, 0.5, 0.18, step=0.01)
        if color_scheme == "Color":
            ci_color = st.color_picker("CI Color", "#1f77b4")
            marker_color = st.color_picker("Point Color", "#d62728")
        else:
            ci_color = "black"
            marker_color = "black"

    if x_measure == "Effect Size (Cohen's d, approx.)":
        plot_column = "Effect Size (Cohen's d, approx.)"
        ci_l = df["Lower CI"].apply(compute_cohens_d)
        ci_u = df["Upper CI"].apply(compute_cohens_d)
        ci_vals = pd.concat([ci_l.dropna(), ci_u.dropna(), df[plot_column].dropna()])
        ref_line = 0
    else:
        plot_column = "Risk, Odds, or Hazard Ratio"
        ci_l = pd.to_numeric(df["Lower CI"], errors="coerce")
        ci_u = pd.to_numeric(df["Upper CI"], errors="coerce")
        ci_vals = pd.concat([ci_l.dropna(), ci_u.dropna(), pd.to_numeric(df[plot_column], errors="coerce").dropna()])
        ref_line = 1

    if x_measure == "Risk, Odds, or Hazard Ratio":
        x_axis_label = infer_ratio_axis_label(df)
        auto_x_min, auto_x_max = compute_ratio_axis_limits(ci_vals, axis_padding, use_log=use_log)
    else:
        x_axis_label = plot_column
        auto_x_min = ci_vals.min() if not ci_vals.empty else None
        auto_x_max = ci_vals.max() if not ci_vals.empty else None

    auto_x_span = (auto_x_max - auto_x_min) if auto_x_min is not None and auto_x_max is not None else None
    default_x_pad = (auto_x_span * (axis_padding / 100)) if auto_x_span is not None and auto_x_span != 0 else 0.1

    if manual_x_axis:
        x_input_help = "Set the minimum and maximum X-axis values to control the full plotted range."
        if auto_x_min is None or auto_x_max is None:
            manual_x_min = st.sidebar.number_input("X-axis minimum", value=0.0, format="%.4f", help=x_input_help)
            manual_x_max = st.sidebar.number_input("X-axis maximum", value=2.0, format="%.4f", help=x_input_help)
        else:
            manual_x_min = st.sidebar.number_input(
                "X-axis minimum",
                value=float(auto_x_min - default_x_pad),
                format="%.4f",
                help=x_input_help,
            )
            manual_x_max = st.sidebar.number_input(
                "X-axis maximum",
                value=float(auto_x_max + default_x_pad),
                format="%.4f",
                help=x_input_help,
            )
    else:
        manual_x_min = None
        manual_x_max = None

    if st.button("📊 Generate Forest Plot"):
        if ci_vals.empty:
            st.error("No plottable effect estimates were found.")
        else:
            if manual_x_axis:
                if manual_x_min >= manual_x_max:
                    st.error("The X-axis minimum must be smaller than the X-axis maximum.")
                    st.stop()
                plot_x_min, plot_x_max = manual_x_min, manual_x_max
            else:
                if x_measure == "Risk, Odds, or Hazard Ratio":
                    plot_x_min, plot_x_max = compute_ratio_axis_limits(ci_vals, axis_padding, use_log=use_log)
                else:
                    x_min, x_max = ci_vals.min(), ci_vals.max()
                    x_pad = (x_max - x_min) * (axis_padding / 100) if x_max != x_min else 0.1
                    plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad

            try:
                fig = create_forest_table_hybrid(
                    df=df,
                    plot_column=plot_column,
                    x_measure=x_measure,
                    x_axis_label=x_axis_label,
                    ref_line=ref_line,
                    x_min=plot_x_min,
                    x_max=plot_x_max,
                    use_groups=use_groups,
                    use_log=use_log,
                    show_grid=show_grid,
                    plot_title=plot_title,
                    font_size=font_size,
                    point_size=point_size,
                    line_width=line_width,
                    cap_height=cap_height,
                    header_color=table_header_color,
                    significant_color=significant_color,
                    nonsignificant_color=nonsignificant_color,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                "📥 Download Plot as PNG",
                data=buf.getvalue(),
                file_name="forest_plot_table_hybrid.png",
                mime="image/png",
            )

else:
    st.info("Please upload file(s) or enter data manually to generate a plot.")
