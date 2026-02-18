import io
import csv
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(layout="wide")
st.title("🌲 Novak's TriNetX Forest Plot Generator")

REQUIRED_COLS = ["Outcome", "Effect Size", "Lower CI", "Upper CI"]
DELETE_COL = "🗑 Delete"
ORDER_COL = "↕ Order"
HEADER_COL = "🅷 Header"


# ----------------------------
# Table helpers: order + insert + move + delete + header (IMPROVED UX)
# ----------------------------
def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required cols exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    # Ensure control columns exist
    if DELETE_COL not in df.columns:
        df.insert(0, DELETE_COL, False)
    if HEADER_COL not in df.columns:
        df.insert(0, HEADER_COL, False)
    if ORDER_COL not in df.columns:
        df.insert(0, ORDER_COL, list(range(1, len(df) + 1)))

    # Normalize types
    df[DELETE_COL] = df[DELETE_COL].fillna(False).astype(bool)
    df[HEADER_COL] = df[HEADER_COL].fillna(False).astype(bool)
    df[ORDER_COL] = pd.to_numeric(df[ORDER_COL], errors="coerce")

    # Fill missing ORDER values at the end
    if df[ORDER_COL].isna().any():
        max_order = df[ORDER_COL].max(skipna=True)
        max_order = 0 if pd.isna(max_order) else float(max_order)
        na_idx = df.index[df[ORDER_COL].isna()].tolist()
        for k, idx in enumerate(na_idx, start=1):
            df.loc[idx, ORDER_COL] = max_order + k

    # Sort by ORDER (stable), then renumber sequentially
    df = df.sort_values(ORDER_COL, kind="mergesort").reset_index(drop=True)
    df[ORDER_COL] = range(1, len(df) + 1)

    # If a row is marked as header, blank out numeric columns (keeps plot logic clean)
    hdr = df[HEADER_COL].fillna(False).astype(bool)
    if hdr.any():
        df.loc[hdr, ["Effect Size", "Lower CI", "Upper CI"]] = None

    return df


def _blank_row_for(df: pd.DataFrame) -> dict:
    row = {}
    for c in df.columns:
        if c == DELETE_COL:
            row[c] = False
        elif c == HEADER_COL:
            row[c] = False
        elif c == ORDER_COL:
            row[c] = None
        elif c in ["Effect Size", "Lower CI", "Upper CI"]:
            row[c] = None
        else:
            row[c] = ""
    return row


def editable_table_with_row_ops(df_seed: pd.DataFrame, state_key: str):
    """
    Easier row tools:
      - Pick row by number (1..N) and see a live preview
      - Big action buttons (Up/Down/Insert Above/Below/Toggle Header/Delete Selected)
      - Separate "Delete checked rows" and "Clear delete checks"
      - Header row checkbox column (🅷 Header) that auto-converts rows into headers
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = _normalize_table(df_seed)

    st.session_state[state_key] = _normalize_table(st.session_state[state_key])
    df_now = st.session_state[state_key]

    st.caption("Tip: Check 🅷 Header to make a row a section header (no need to type '##'). You can still use '##' if you want.")

    edited = st.data_editor(
        df_now,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{state_key}",
        column_config={
            ORDER_COL: st.column_config.NumberColumn(
                ORDER_COL,
                help="Row order. You can type numbers here or use the row tools below.",
                step=1,
            ),
            HEADER_COL: st.column_config.CheckboxColumn(
                HEADER_COL,
                help="Treat this row as a header/section title (effect/CI ignored).",
            ),
            DELETE_COL: st.column_config.CheckboxColumn(
                DELETE_COL,
                help="Mark rows for deletion (then click 'Delete checked rows').",
            ),
            "Effect Size": st.column_config.NumberColumn("Effect Size", step=0.01, format="%.4f"),
            "Lower CI": st.column_config.NumberColumn("Lower CI", step=0.01, format="%.4f"),
            "Upper CI": st.column_config.NumberColumn("Upper CI", step=0.01, format="%.4f"),
        },
    )

    # Persist edits, then normalize (this also blanks numeric cells for header rows)
    st.session_state[state_key] = _normalize_table(edited)
    df_now = st.session_state[state_key]

    if len(df_now) == 0:
        return df_now

    # Work in view-sorted space (by ORDER_COL)
    df_view = df_now.copy()
    df_view[ORDER_COL] = pd.to_numeric(df_view[ORDER_COL], errors="coerce")
    df_view = df_view.sort_values(ORDER_COL, kind="mergesort").reset_index(drop=True)

    # Row tools UI (expanded by default, more ergonomic)
    with st.expander("Row tools (move / insert / header / delete)", expanded=False):
        top = st.columns([2.2, 3.8, 1.2, 1.2, 1.4, 1.4, 1.6])
        with top[0]:
            row_num = st.number_input(
                "Row #",
                min_value=1,
                max_value=len(df_view),
                value=1,
                step=1,
                key=f"rownum_{state_key}",
            )
            sel_idx = int(row_num) - 1

        # Preview panel
        with top[1]:
            o = df_view.loc[sel_idx, "Outcome"]
            o = "" if pd.isna(o) else str(o)
            is_hdr = bool(df_view.loc[sel_idx, HEADER_COL])
            es = df_view.loc[sel_idx, "Effect Size"]
            lci = df_view.loc[sel_idx, "Lower CI"]
            uci = df_view.loc[sel_idx, "Upper CI"]
            if is_hdr:
                st.write(f"Selected: **HEADER** → {o}")
            else:
                st.write(f"Selected: {o}")
                st.write(f"Effect/CI: {es}  |  {lci}–{uci}")

        def commit(df_commit: pd.DataFrame):
            st.session_state[state_key] = _normalize_table(df_commit)
            st.rerun()

        curr_order = float(pd.to_numeric(df_view.loc[sel_idx, ORDER_COL], errors="coerce"))

        # Move Up
        with top[2]:
            if st.button("⬆️ Up", key=f"up_{state_key}", use_container_width=True, disabled=(sel_idx <= 0)):
                above_order = float(pd.to_numeric(df_view.loc[sel_idx - 1, ORDER_COL], errors="coerce"))
                df2 = df_view.copy()
                df2.loc[sel_idx, ORDER_COL] = above_order
                df2.loc[sel_idx - 1, ORDER_COL] = curr_order
                commit(df2)

        # Move Down
        with top[3]:
            if st.button("⬇️ Down", key=f"down_{state_key}", use_container_width=True, disabled=(sel_idx >= len(df_view) - 1)):
                below_order = float(pd.to_numeric(df_view.loc[sel_idx + 1, ORDER_COL], errors="coerce"))
                df2 = df_view.copy()
                df2.loc[sel_idx, ORDER_COL] = below_order
                df2.loc[sel_idx + 1, ORDER_COL] = curr_order
                commit(df2)

        # Insert Above
        with top[4]:
            if st.button("➕ Insert above", key=f"ins_above_{state_key}", use_container_width=True):
                df2 = df_view.copy()
                new_row = _blank_row_for(df2)
                new_row[ORDER_COL] = curr_order - 0.5
                df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
                commit(df2)

        # Insert Below
        with top[5]:
            if st.button("➕ Insert below", key=f"ins_below_{state_key}", use_container_width=True):
                df2 = df_view.copy()
                new_row = _blank_row_for(df2)
                new_row[ORDER_COL] = curr_order + 0.5
                df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
                commit(df2)

        # Toggle Header for selected
        with top[6]:
            if st.button("🅷 Toggle header", key=f"toggle_hdr_{state_key}", use_container_width=True):
                df2 = df_view.copy()
                new_val = not bool(df2.loc[sel_idx, HEADER_COL])
                df2.loc[sel_idx, HEADER_COL] = new_val
                if new_val:
                    df2.loc[sel_idx, ["Effect Size", "Lower CI", "Upper CI"]] = None
                commit(df2)

        bottom = st.columns([2.0, 2.0, 3.0, 3.0])
        # Delete selected row (immediate)
        with bottom[0]:
            if st.button("🗑 Delete selected row", key=f"del_sel_{state_key}", use_container_width=True):
                df2 = df_view.copy().drop(index=sel_idx).reset_index(drop=True)
                commit(df2)

        # Delete checked rows
        with bottom[1]:
            if st.button("🗑 Delete checked rows", key=f"del_checked_{state_key}", use_container_width=True):
                df2 = df_now.copy()
                mask = df2[DELETE_COL].fillna(False).astype(bool)
                df2 = df2.loc[~mask].copy().reset_index(drop=True)
                commit(df2)

        # Clear delete checks
        with bottom[2]:
            if st.button("✅ Clear delete checks", key=f"clear_del_{state_key}", use_container_width=True):
                df2 = df_now.copy()
                df2[DELETE_COL] = False
                commit(df2)

        # Quick add header row below selected (common workflow)
        with bottom[3]:
            if st.button("➕ Add header row below", key=f"add_hdr_below_{state_key}", use_container_width=True):
                df2 = df_view.copy()
                new_row = _blank_row_for(df2)
                new_row[HEADER_COL] = True
                new_row[ORDER_COL] = curr_order + 0.5
                df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
                commit(df2)

    return st.session_state[state_key]


# ----------------------------
# TriNetX parsing utils
# ----------------------------
def _clean_line(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
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
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s or ""))
    if len(nums) >= 3:
        return (_to_float(nums[0]), _to_float(nums[1]), _to_float(nums[2]))
    return (None, None, None)


def _parse_section_effect(lines, section_name: str):
    results = []
    n = len(lines)

    for i, line in enumerate(lines):
        if _clean_line(line).lower() == section_name.lower():
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
    lines = (text or "").splitlines()
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
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()

    if ext == "csv":
        raw = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
        return parse_trinetx_export_text(raw, name)

    if ext == "xlsx":
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
        from docx import Document

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


def insert_section_headers(df: pd.DataFrame, group_col: str):
    """
    Inserts header rows into REQUIRED_COLS schema.
    Uses HEADER_COL instead of forcing '##' prefixes.
    """
    if df.empty or group_col not in df.columns:
        return df

    out_rows = []
    for g, sub in df.groupby(group_col, sort=False):
        out_rows.append(
            {
                "Outcome": str(g),
                "Effect Size": None,
                "Lower CI": None,
                "Upper CI": None,
                HEADER_COL: True,
            }
        )
        out_rows.extend(sub[REQUIRED_COLS].to_dict("records"))

    out = pd.DataFrame(out_rows)
    return out


# ----------------------------
# Input mode UI (re-ordered)
# ----------------------------
input_mode = st.radio(
    "Select data input method:",
    ["✍️ Manual entry", "📄 Import TriNetX tables", "📤 Upload structured file"],
    index=0,
    horizontal=True,
)

df = None

if input_mode == "✍️ Manual entry":
    default_data = pd.DataFrame(
        {
            "Outcome": ["Cardiovascular", "Hypertension", "Stroke", "Metabolic", "Diabetes", "Obesity"],
            "Effect Size": [None, 1.5, 1.2, None, 0.85, 1.2],
            "Lower CI": [None, 1.2, 1.0, None, 0.7, 1.0],
            "Upper CI": [None, 1.8, 1.5, None, 1.0, 1.4],
            HEADER_COL: [True, False, False, True, False, False],
        }
    )
    df = editable_table_with_row_ops(default_data, "manual_table_df")

elif input_mode == "📄 Import TriNetX tables":
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
            with st.expander("Parsing warnings", expanded=False):
                for fn, msg in failures:
                    st.write(f"- {fn}: {msg}")

        if parsed_frames:
            parsed = pd.concat(parsed_frames, ignore_index=True)

            effect_types = sorted(parsed["Effect Type"].unique().tolist())
            default_keep = [t for t in ["Risk Ratio", "Hazard Ratio"] if t in effect_types] or effect_types
            keep_types = st.multiselect("Effect types to include:", options=effect_types, default=default_keep)

            plot_base = parsed[parsed["Effect Type"].isin(keep_types)].copy()

            append_type_when_needed = st.checkbox(
                "Append effect type to Outcome label when the same outcome has multiple effect types",
                value=True,
            )
            if append_type_when_needed and not plot_base.empty:
                multi = plot_base.groupby("Outcome")["Effect Type"].nunique()
                multi_outcomes = set(multi[multi > 1].index)
                plot_base["Outcome"] = plot_base.apply(
                    lambda r: f"{r['Outcome']} ({r['Effect Type']})" if r["Outcome"] in multi_outcomes else r["Outcome"],
                    axis=1,
                )

            add_headers = st.checkbox("Insert section headers per uploaded table", value=False)
            header_grouping = st.selectbox("Header grouping field", ["Source", "File"], index=0, disabled=not add_headers)

            for c in REQUIRED_COLS:
                if c not in plot_base.columns:
                    plot_base[c] = None
            for c in ["Effect Size", "Lower CI", "Upper CI"]:
                plot_base[c] = pd.to_numeric(plot_base[c], errors="coerce")

            if add_headers:
                df_for_editor = insert_section_headers(plot_base, group_col=header_grouping)
            else:
                df_for_editor = plot_base[REQUIRED_COLS].copy()

            df = editable_table_with_row_ops(df_for_editor, "trinetx_import_table_df")

            with st.expander("Extracted Data", expanded=False):
                st.dataframe(parsed, use_container_width=True)

            st.download_button(
                "📥 Download parsed rows as CSV",
                data=plot_base.to_csv(index=False).encode("utf-8"),
                file_name="parsed_trinetx_effects.csv",
                mime="text/csv",
            )
        else:
            st.info("No parsable TriNetX effect sections were found in the uploaded files.")
    else:
        st.info("Upload one or more TriNetX export tables to parse effect sizes and confidence intervals.")

else:  # Upload structured file
    st.subheader("Upload a structured file")

    template_df = pd.DataFrame(
        {
            "Outcome": ["Cardiovascular", "Hypertension", "Stroke"],
            HEADER_COL: [True, False, False],
            "Effect Size": [None, 1.50, 1.20],
            "Lower CI": [None, 1.20, 1.00],
            "Upper CI": [None, 1.80, 1.50],
        }
    )
    st.download_button(
        "📥 Download structured CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="forest_plot_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            sig = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("structured_file_sig") != sig:
                st.session_state["structured_file_sig"] = sig
                st.session_state.pop("structured_table_df", None)

            if uploaded_file.name.endswith(".csv"):
                df_loaded = pd.read_csv(uploaded_file)
            else:
                df_loaded = pd.read_excel(uploaded_file)

            if not all(col in df_loaded.columns for col in REQUIRED_COLS):
                st.error(f"Your file must include columns: {REQUIRED_COLS}")
                df = None
            else:
                df = editable_table_with_row_ops(df_loaded, "structured_table_df")
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ----------------------------
# Plot controls + plot
# ----------------------------
if df is not None:
    # Drop delete column; keep HEADER_COL only for plot logic (not shown on axis)
    plot_df = df.drop(columns=[DELETE_COL], errors="ignore").copy()

    # Respect explicit ordering if present
    if ORDER_COL in plot_df.columns:
        plot_df[ORDER_COL] = pd.to_numeric(plot_df[ORDER_COL], errors="coerce")
        plot_df = plot_df.sort_values(ORDER_COL, kind="mergesort").reset_index(drop=True)

    # Ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in plot_df.columns:
            plot_df[c] = None
    if HEADER_COL not in plot_df.columns:
        plot_df[HEADER_COL] = False

    # Coerce numeric cols
    for c in ["Effect Size", "Lower CI", "Upper CI"]:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    st.sidebar.header("⚙️ Basic Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = st.sidebar.checkbox("Show Numerical Annotations", value=False)
    # Keep this option for backward compatibility with older "##" habit
    use_groups = st.sidebar.checkbox("Also treat rows starting with '##' as section headers", value=True)

    with st.sidebar.expander("📏 X-axis range controls", expanded=False):
        use_custom_xlim = st.checkbox("Use custom X-axis start/end", value=False)
        x_start = st.number_input("X-axis start", value=0.0, step=0.1, disabled=not use_custom_xlim)
        x_end = st.number_input("X-axis end", value=3.0, step=0.1, disabled=not use_custom_xlim)

    with st.sidebar.expander("🧱 Top headroom & layout", expanded=False):
        # Default is now 0 rows (your request)
        top_headroom_rows = st.slider("Top headroom (rows)", 0.0, 6.0, 0.0, step=0.5)
        bottom_padding_rows = st.slider("Bottom padding (rows)", 0.0, 6.0, 1.0, step=0.5)
        title_pad_pts = st.slider("Title pad (points)", 0, 40, 12, step=2)

    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        point_size = st.slider("Marker Size", 6, 20, 10)
        line_width = st.slider("CI Line Width", 1, 4, 2)
        font_size = st.slider("Font Size", 10, 20, 12)
        label_offset = st.slider("Label Horizontal Offset", 0.01, 0.3, 0.05)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)
        axis_padding = st.slider("X-axis Padding (%)", 2, 40, 10)
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

        for _, row in plot_df.iterrows():
            outcome_val = row.get("Outcome", "")
            outcome_val = "" if pd.isna(outcome_val) else str(outcome_val)

            is_header = bool(row.get(HEADER_COL, False))
            if use_groups and outcome_val.startswith("##"):
                is_header = True

            if is_header:
                header_txt = outcome_val
                if header_txt.startswith("##"):
                    header_txt = header_txt[2:].lstrip("#").strip()
                y_labels.append(header_txt.strip())
                text_styles.append("bold")
                rows.append(None)
                group_mode = True
            else:
                display_name = f"{indent}{outcome_val}" if group_mode else outcome_val
                y_labels.append(display_name)
                text_styles.append("normal")
                rows.append(row)

        fig, ax = plt.subplots(figsize=(10, max(2.5, len(y_labels) * 0.7)))

        # X limits
        if use_custom_xlim:
            if x_end <= x_start:
                st.error("Custom X-axis end must be greater than start.")
                st.stop()
            if use_log and (x_start <= 0 or x_end <= 0):
                st.error("Log scale requires positive X-axis limits.")
                st.stop()
            ax.set_xlim(x_start, x_end)
        else:
            ci_series = pd.concat([plot_df["Lower CI"].dropna(), plot_df["Upper CI"].dropna()], ignore_index=True)
            if ci_series.empty:
                st.error("No valid CI values found. Please check your table.")
                st.stop()

            if use_log:
                ci_series = ci_series[ci_series > 0]
                if ci_series.empty:
                    st.error("Log scale requires positive effect sizes and CI bounds.")
                    st.stop()

            x_min, x_max = ci_series.min(), ci_series.max()
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

        # Headroom control (top headroom default = 0 now)
        ax.set_ylim(len(y_labels) - 1 + bottom_padding_rows, -1 - top_headroom_rows)

        ax.set_xlabel(x_axis_label, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, weight="bold", pad=title_pad_pts)

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
