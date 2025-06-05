import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Use clean whitegrid style
plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(layout="wide")
st.title("🌲 Forest Plot Generator")

# Input method
input_mode = st.radio("Select data input method:", ["📤 Upload file", "✍️ Manual entry"], index=1, horizontal=True)

required_cols = ["Outcome", "Effect Size", "Lower CI", "Upper CI"]
df = None

if input_mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            if not all(col in df.columns for col in required_cols):
                st.error(f"Your file must include the following columns: {required_cols}")
                df = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    default_data = pd.DataFrame({
        "Outcome": ["## Cardiovascular", "Hypertension", "Stroke", "## Metabolic", "Diabetes", "Obesity"],
        "Effect Size": [None, 1.5, 1.2, None, 0.85, 1.2],
        "Lower CI": [None, 1.2, 1.0, None, 0.7, 1.0],
        "Upper CI": [None, 1.8, 1.5, None, 1.0, 1.4],
    })
    df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="manual_input_table")

if df is not None:
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

        for i, row in df.iterrows():
            if use_groups and isinstance(row["Outcome"], str) and row["Outcome"].startswith("##"):
                header = row["Outcome"][3:].strip()
                y_labels.append(header)
                text_styles.append("bold")
                rows.append(None)
                group_mode = True
            else:
                display_name = f"{indent}{row['Outcome']}" if group_mode else row["Outcome"]
                y_labels.append(display_name)
                text_styles.append("normal")
                rows.append(row)

        fig, ax = plt.subplots(figsize=(10, len(y_labels) * 0.7))
        valid_rows = [i for i in range(len(rows)) if rows[i] is not None]

        # Axis limits with padding
        ci_vals = pd.concat([df["Lower CI"].dropna(), df["Upper CI"].dropna()])
        x_min, x_max = ci_vals.min(), ci_vals.max()
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
                ax.hlines(i, xmin=lci, xmax=uci, color=ci_color, linewidth=line_width, capstyle='round')
                ax.plot(effect, i, 'o', color=marker_color, markersize=point_size)
                if show_values:
                    label = f"{effect:.2f} [{lci:.2f}, {uci:.2f}]"
                    ax.text(uci + label_offset, i, label, va='center', fontsize=font_size - 2)

        # Reference line at 1
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)

        # Custom tick labels with styling
        ax.set_yticks(range(len(y_labels)))
        for tick_label, style in zip(ax.set_yticklabels(y_labels), text_styles):
            if style == "bold":
                tick_label.set_fontweight("bold")
            tick_label.set_fontsize(font_size)

        if use_log:
            ax.set_xscale('log')
        if show_grid:
            ax.grid(True, axis='x', linestyle=':', linewidth=0.6)
        else:
            ax.grid(False)

        # Y-axis padding
        ax.set_ylim(len(y_labels) - 1 + y_axis_padding, -1 - y_axis_padding)

        # Labels
        ax.set_xlabel(x_axis_label, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, weight='bold')
        fig.tight_layout()

        # Display plot
        st.pyplot(fig)

        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")

else:
    st.info("Please upload a file or enter data manually to generate a plot.")
