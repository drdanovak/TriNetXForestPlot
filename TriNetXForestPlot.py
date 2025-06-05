import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Configure page
st.set_page_config(layout="wide")
st.title("🌲 Forest Plot Generator")

# Choose input method
input_mode = st.radio("Select data input method:", ["📤 Upload file", "✍️ Manual entry"], horizontal=True)

# Expected columns
required_cols = ["Outcome", "Effect Size", "Lower CI", "Upper CI"]

# Load data
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
        "Outcome": ["Hypertension", "Diabetes", "Obesity"],
        "Effect Size": [1.5, 0.85, 1.2],
        "Lower CI": [1.2, 0.7, 1.0],
        "Upper CI": [1.8, 1.0, 1.4],
    })
    df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_input_table"
    )

if df is not None:
    # Basic settings
    st.sidebar.header("⚙️ Basic Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = st.sidebar.checkbox("Show Numerical Annotations", value=False)

    # Advanced settings in expander
    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        point_size = st.slider("Marker Size", 4, 20, 10)
        line_width = st.slider("CI Line Width", 1, 5, 2)
        font_size = st.slider("Font Size", 8, 20, 12)
        axis_min = st.number_input("X-axis Min (optional)", value=0.0, step=0.1)
        axis_max = st.number_input("X-axis Max (optional)", value=2.5, step=0.1)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)

        # Color overrides
        if color_scheme == "Color":
            ci_color = st.color_picker("CI Line Color", "#1f77b4")
            marker_color = st.color_picker("Marker Color", "#d62728")
        else:
            ci_color = "black"
            marker_color = "black"

    if st.button("📊 Generate Forest Plot"):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.6))
        y_pos = range(len(df))

        for i, row in df.iterrows():
            ax.plot([row["Lower CI"], row["Upper CI"]], [i, i],
                    color=ci_color, linewidth=line_width)
            ax.plot(row["Effect Size"], i, 'o',
                    color=marker_color, markersize=point_size)

            if show_values:
                label = f"{row['Effect Size']} [{row['Lower CI']}, {row['Upper CI']}]"
                ax.text(row["Upper CI"] + 0.05, i, label, va='center', fontsize=font_size - 2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Outcome"], fontsize=font_size)
        ax.axvline(1, color='grey', linestyle='--')
        ax.set_xlabel(x_axis_label, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2)
        if show_grid:
            ax.grid(True, axis='x', linestyle=':', alpha=0.7)
        if use_log:
            ax.set_xscale("log")
        ax.set_xlim(left=axis_min, right=axis_max)
        ax.invert_yaxis()
        fig.tight_layout()

        st.pyplot(fig)

        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")

else:
    st.info("Please upload a file or enter data manually to generate a plot.")
