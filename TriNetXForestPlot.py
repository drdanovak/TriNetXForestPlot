import io
import csv
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Use clean whitegrid style
plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(layout="wide")
st.title("🌲 Novak's TriNetX Forest Plot Generator")

REQUIRED_COLS = ["Outcome", "Effect Size", "Lower CI", "Upper CI"]


# ----------------------------
# TriNetX export parsing utils
# ----------------------------
def _clean_line(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # common TriNetX "blank" lines in CSV exports
    if s in {'" "', '""', '"  "', '"\ufeff"', '" "'}:
        return ""
    # strip surrounding quotes if whole line is quoted
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


def _to_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def _extract_ci_from_string(s: str):
    """
    Handles strings like:
      "0.61 [0.55, 0.67]"
      "0.61 (0.55, 0.67)"
      "0.61 0.55 0.67"
    Returns (effect, lci, uci) or (None, None, None).
    """
    if not s:
        return (None, None, None)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    if len(nums) >= 3:
        return (_to_float(nums[0]), _to_float(nums[1]), _to_float(nums[2]))
    return (None, None, None)


def _parse_section_effect(lines, section_name: str):
    """
    Finds one or more occurrences of a TriNetX section like:
      Risk Ratio
      Risk Ratio,95 % CI Lower,95 % CI Upper
      0.68,0.62,0.75
    Returns list of dicts with effect/lci/uci.
    """
    results = []
    n = len(lines)
    for i, line in enumerate(lines):
        if _clean_line(line).lower() == section_name.lower():
            # header line after the section title
            j = i + 1
            while j < n and _clean_line(lines[j]) == "":
                j += 1
            if j >= n:
                continue

            header_line = lines[j]
            try:
                header = next(csv.reader([header_line]))
            except Exception:
                header = [header_line]
            header = [h.strip() for h in header if h is not None]

            # value line after header
            k = j + 1
            while k < n and _clean_line(lines[k]) == "":
                k += 1
            if k >= n:
                continue

            value_line = lines[k]
            try:
                vals = next(csv.reader([value_line]))
            except Exception:
                vals = [value_line]
            vals = [v.strip() for v in vals]

            # Identify indices for effect, lower CI, upper CI
            lower_idx = None
            upper_idx = None
            for idx, h in enumerate(header):
                hl = h.lower()
                if ("ci" in hl and "lower" in hl) or ("95" in hl and "lower" in hl):
                    lower_idx = idx
                if ("ci" in hl and "upper" in hl) or ("95" in hl and "upper" in hl):
                    upper_idx = idx

            eff_idx = None
            for idx, h in enumerate(header):
                if section_name.lower().replace(" ", "") in h.lower().replace(" ", ""):
                    eff_idx = idx
                    break
            if eff_idx is None:
                eff_idx = 0

            eff = _to_float(vals[eff_idx]) if eff_idx < len(vals) else None
            lci = _to_float(vals[lower_idx]) if (lower_idx is not None and lower_idx < len(vals)) else None
            uci = _to_float(vals[upper_idx]) if (upper_idx is not None and upper_idx < len(vals)) else None

            # Fallback: sometimes everything is in one column
            if eff is None or lci is None or uci is None:
                eff2, lci2, uci2 = _extract_ci_from_string(" ".join(vals))
                eff = eff if eff is not None else eff2
                lci = lci if lci is not None else lci2
                uci = uci if uci is not None else uci2

            if eff is not None and lci is not None and uci is not None:
                results.append(
                    {"Effect Type": section_name, "Effect Size": eff, "Lower CI": lci, "Upper CI": uci}
                )

    return results


def parse_trinetx_export_text(text: str, filename: str):
    """
    Parses TriNetX export content (CSV-ish multi-section files).
    Extracts Risk Ratio / Hazard Ratio (and Odds Ratio if present).
    Returns a dataframe with at least REQUIRED_COLS plus metadata cols.
    """
    lines = text.splitlines()
    if lines:
        lines[0] = lines[0].lstrip("\ufeff")

    title = None
    for l in lines:
        cl = _clean_line(l)
        if cl:
            title = cl
            break

    base_outcome = Path(filename).stem

    extracted = []
    for section in ["Risk Ratio", "Hazard Ratio", "Odds Ratio"]:
        extracted.extend(_parse_section_effect(lines, section))

    if not extracted:
        return pd.DataFrame()

    df = pd.DataFrame(extracted)
    df.insert(0, "Outcome", base_outcome)
    df["Source"] = title or filename
    df["File"] = filename
    return df


def parse_uploaded_trinetx_file(uploaded_file):
    """
    Supports:
      - .csv: decode text and parse
      - .xlsx: flattens sheets into pseudo-lines and parse
      - .docx: best-effort (paragraphs + tables) and parse
    """
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()

    if ext == "csv":
        raw = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
        return parse_trinetx_export_text(raw, name)

    if ext == "xlsx":
        # flatten each sheet into "lines" similar to the TriNetX CSV format
        xls = pd.ExcelFile(uploaded_file)
        flat_lines = []
        for sheet in xls.sheet_names:
            sheet_df = pd.read_excel(xls, sheet_name=sheet, header=None)
            for _, r in sheet_df.iterrows():
                vals = [str(v).strip() for v in r.tolist() if pd.notnull(v) and str(v).strip() != ""]
                if vals:
                    flat_lines.append(",".join(vals))
        return parse_trinetx_export_text("\n".join(flat_lines), name)

    if ext == "docx":
        try:
            from docx import Document
        except Exception as e:
            st.error("DOCX parsing requires python-docx. Install it or export as CSV/XLSX.")
            raise e

        doc = Document(io.BytesIO(uploaded_file.getvalue()))
        lines = []

        for p in doc.paragraphs:
            t = p.text.strip()
            if t:
                lines.append(t)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    lines.append(",".join(cells))

        return parse_trinetx_export_text("\n".join(lines), name)

    return pd.DataFrame()


def insert_section_headers(df: pd.DataFrame, group_col: str, header_prefix: str = "## "):
    """
    Inserts header rows like:
      Outcome = "## <group value>"
      Effect/CI cells are None
    """
    if df.empty or group_col not in df.columns:
        return df

    out_rows = []
    for g, sub in df.groupby(group_col, sort=False):
        out_rows.append(
            {"Outcome": f"{header_prefix}{g}", "Effect Size": None, "Lower CI": None, "Upper CI": None}
        )
        out_rows.extend(sub[REQUIRED_COLS].to_dict("records"))

    return pd.DataFrame(out_rows)


# ----------------------------
# Input mode UI
# ----------------------------
input_mode = st.radio(
    "Select data input method:",
    ["📤 Upload structured file", "📄 Import TriNetX export table(s)", "✍️ Manual entry"],
    index=2,
    horizontal=True,
)

df = None

if input_mode == "📤 Upload structured file":
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if not all(col in df.columns for col in REQUIRED_COLS):
                st.error(f"Your file must include the following columns: {REQUIRED_COLS}")
                df = None
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif input_mode == "📄 Import TriNetX export table(s)":
    uploaded_files = st.file_uploader(
        "Upload one or more TriNetX export tables (CSV/XLSX/DOCX)",
        type=["csv", "xlsx", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        parsed_frames = []
        failures = []

        for f in uploaded_files:
            try:
                p = parse_uploaded_trinetx_file(f)
                if not p.empty:
                    parsed_frames.append(p)
                else:
                    failures.append((f.name, "No Risk Ratio / Hazard Ratio / Odds Ratio section found."))
            except Exception as e:
                failures.append((f.name, str(e)))

        if failures:
            with st.expander("Parsing warnings (click to expand)", expanded=False):
                for fn, msg in failures:
                    st.write(f"- {fn}: {msg}")

        if parsed_frames:
            parsed = pd.concat(parsed_frames, ignore_index=True)

            effect_types = sorted(parsed["Effect Type"].unique().tolist())
            default_keep = [t for t in ["Risk Ratio", "Hazard Ratio"] if t in effect_types]
            if not default_keep:
                default_keep = effect_types

            keep_types = st.multiselect(
                "Effect types to include in the plot table:",
                options=effect_types,
                default=default_keep,
            )

            # Filter down to plot candidates
            plot_base = parsed[parsed["Effect Type"].isin(keep_types)].copy()

            append_type_when_needed = st.checkbox(
                "Append effect type to the Outcome label when an outcome has multiple effect types",
                value=True,
            )
            if append_type_when_needed and not plot_base.empty:
                multi = plot_base.groupby("Outcome")["Effect Type"].nunique()
                multi_outcomes = set(multi[multi > 1].index)
                plot_base["Outcome"] = plot_base.apply(
                    lambda r: f"{r['Outcome']} ({r['Effect Type']})"
                    if r["Outcome"] in multi_outcomes
                    else r["Outcome"],
                    axis=1,
                )

            add_headers = st.checkbox("Insert section headers per uploaded table", value=False)
            header_grouping = st.selectbox(
                "Header grouping field",
                options=["Source", "File"],
                index=0,
                disabled=not add_headers,
            )

            # This is the editable table that drives the plot:
            plot_table = plot_base.copy()

            # Ensure required columns exist
            for c in REQUIRED_COLS:
                if c not in plot_table.columns:
                    plot_table[c] = None

            # Coerce numeric columns
            for c in ["Effect Size", "Lower CI", "Upper CI"]:
                plot_table[c] = pd.to_numeric(plot_table[c], errors="coerce")

            # If headers requested, build a REQUIRED_COLS-only table with header rows
            if add_headers:
                df_for_editor = insert_section_headers(plot_table, group_col=header_grouping)
            else:
                df_for_editor = plot_table[REQUIRED_COLS].copy()

            st.caption("You can edit labels, add rows, or insert your own '##' section headers below.")
            df = st.data_editor(
                df_for_editor,
                num_rows="dynamic",
                use_container_width=True,
                key="trinetx_import_table",
            )

            with st.expander("Raw parsed rows (includes metadata)", expanded=False):
                st.dataframe(parsed, use_container_width=True)

            # Optional: download the parsed table
            csv_bytes = plot_base.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download parsed rows as CSV",
                data=csv_bytes,
                file_name="parsed_trinetx_effects.csv",
                mime="text/csv",
            )
        else:
            st.info("No parsable TriNetX effect sections were found in the uploaded files.")

else:
    default_data = pd.DataFrame(
        {
            "Outcome": ["## Cardiovascular", "Hypertension", "Stroke", "## Metabolic", "Diabetes", "Obesity"],
            "Effect Size": [None, 1.5, 1.2, None, 0.85, 1.2],
            "Lower CI": [None, 1.2, 1.0, None, 0.7, 1.0],
            "Upper CI": [None, 1.8, 1.5, None, 1.0, 1.4],
        }
    )
    df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="manual_input_table")


# ----------------------------
# Plot controls + plot
# ----------------------------
if df is not None:
    # Ensure required columns exist even if user pasted/edited
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    # Coerce numeric columns
    for c in ["Effect Size", "Lower CI", "Upper CI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    st.sidebar.header("⚙️ Basic Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = st.sidebar.checkbox("Show Numerical Annotations", value=False)
    use_groups = st.sidebar.checkbox("Treat rows starting with '##' as section headers", value=True)

    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        point_size = st.slider("Marker Size", 6, 20, 10)
        line_width = st.slider("CI Line Width", 1, 4, 2)
        font_size = st.slider("Font Size", 10, 20, 12)
        label_offset = st.slider("Label Horizontal Offset", 0.01, 0.3, 0.05)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)
        axis_padding = st.slider("X-axis Padding (%)", 2, 40, 10)
        y_axis_padding = st.slider("Y-axis Padding (Rows)", 0.0, 5.0, 1.0, step=0.5)
        cap_height = st.slider("Tick Height (for CI ends)", 0.05, 0.5, 0.18, step=0.01)

        if color_scheme == "Color":
            ci_color = st.color_picker("CI Color", "#1f77b4")
            marker_color = st.color_picker("Point Color", "#d62728")
        else:
            ci_color = "black"
            marker_color = "black"

    if st.button("📊 Generate Forest Plot"):
        rows = []
        y_labels = []
        text_styles = []
        indent = "\u00A0" * 4
        group_mode = False

        for _, row in df.iterrows():
            outcome_val = row.get("Outcome", "")
            if use_groups and isinstance(outcome_val, str) and outcome_val.startswith("##"):
                header = outcome_val[2:].lstrip("#").strip()
                y_labels.append(header)
                text_styles.append("bold")
                rows.append(None)
                group_mode = True
            else:
                display_name = f"{indent}{outcome_val}" if group_mode else outcome_val
                y_labels.append(display_name)
                text_styles.append("normal")
                rows.append(row)

        fig, ax = plt.subplots(figsize=(10, max(2.5, len(y_labels) * 0.7)))

        # Axis limits with padding (robust to empty)
        ci_series = pd.concat([df["Lower CI"].dropna(), df["Upper CI"].dropna()], ignore_index=True)
        if ci_series.empty:
            st.error("No valid CI values found. Please check your table.")
            st.stop()

        x_min, x_max = ci_series.min(), ci_series.max()

        if use_log:
            # log scale requires positive bounds
            positive = ci_series[ci_series > 0]
            if positive.empty:
                st.error("Log scale requires positive effect sizes and CI bounds.")
                st.stop()
            x_min, x_max = positive.min(), positive.max()

        if x_min == x_max:
            x_min = x_min * 0.9
            x_max = x_max * 1.1

        x_pad = (x_max - x_min) * (axis_padding / 100)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        # Plot
        for i, row in enumerate(rows):
            if row is None:
                continue
            effect = row["Effect Size"]
            lci = row["Lower CI"]
            uci = row["Upper CI"]
            if pd.notnull(effect) and pd.notnull(lci) and pd.notnull(uci):
                ax.hlines(i, xmin=lci, xmax=uci, color=ci_color, linewidth=line_width, capstyle="round")
                ax.vlines([lci, uci], i - cap_height, i + cap_height, color=ci_color, linewidth=line_width)
                ax.plot(effect, i, "o", color=marker_color, markersize=point_size)
                if show_values:
                    label = f"{effect:.2f} [{lci:.2f}, {uci:.2f}]"
                    ax.text(uci + label_offset, i, label, va="center", fontsize=max(8, font_size - 2))

        ax.axvline(x=1, color="gray", linestyle="--", linewidth=1)

        ax.set_yticks(range(len(y_labels)))
        tick_labels = ax.set_yticklabels(y_labels)
        for tick_label, style in zip(tick_labels, text_styles):
            if style == "bold":
                tick_label.set_fontweight("bold")
            tick_label.set_fontsize(font_size)

        if use_log:
            ax.set_xscale("log")

        if show_grid:
            ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
        else:
            ax.grid(False)

        ax.set_ylim(len(y_labels) - 1 + y_axis_padding, -1 - y_axis_padding)
        ax.set_xlabel(x_axis_label, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, weight="bold")
        fig.tight_layout()

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button(
            "📥 Download Plot as PNG",
            data=buf.getvalue(),
            file_name="forest_plot.png",
            mime="image/png",
        )
else:
    st.info("Please upload a file or enter data manually to generate a plot.")
