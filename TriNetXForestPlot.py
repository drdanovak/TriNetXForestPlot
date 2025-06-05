import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(layout="wide")
st.title("🌲 Forest Plot Generator")

# Input method
input_mode = st.radio("Select data input method:", ["📤 Upload file", "✍️ Manual entry"], horizontal=True)

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
        "Outcome": ["Hypertension", "Diabetes", "Obesity"],
        "Effect Size": [1.5, 0.85, 1.2],
        "Lower CI": [1.2, 0.7, 1.0],
        "Upper CI": [1.8, 1.0, 1.4],
    })
    df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="manual_input_table")

if df is not None:
    st.sidebar.header("⚙️ Basic Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = st.sidebar.checkbox("Show Numerical Annotations", value=False)

    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        point_size = st.slider("Marker Size", 6, 20, 10)
        line_width = st.slider("CI Line Width", 1, 4, 2)
        font_size = st.slider("Font Size", 10, 20, 12)
        label_offset = st.slider("Label Horizontal Offset", 0.01, 0.3, 0.05)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)
        axis_padding = st.slider("X-axis Padding (%)", 2, 40, 10)

        if color_scheme == "Color":
            ci_color = st.color_picker("CI Color", "#1f77b4")
            marker_color = st.color_picker("Point Color", "#d62728")
        else:
            ci_color = "black"
            marker_color = "black"

    if st.button("📊 Generate Forest Plot"):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.7))
        y_pos = range(len(df))

        # Plot points and CIs
        for i, row in df.iterrows():
            ax.hlines(y=i, xmin=row["Lower CI"], xmax=row["Upper CI"],
                      color=ci_color, linewidth=line_width, capstyle='round')
            ax.plot(row["Effect Size"], i, 'o',
                    color=marker_color, markersize=point_size)
            if show_values:
                annotation = f'{row["Effect Size"]:.2f} [{row["Lower CI"]:.2f}, {row["Upper CI"]:.2f}]'
                ax.text(row["Upper CI"] + label_offset, i, annotation,
                        va='center', fontsize=font_size - 2)

        # Aesthetic and axis setup
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["Outcome"], fontsize=font_size)
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel(x_axis_label, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, weight='bold')

        if show_grid:
            ax.grid(True, axis='x', linestyle=':', linewidth=0.6)
        else:
            ax.grid(False)

        if use_log:
            ax.set_xscale('log')

        # Auto axis limits with padding
        all_ci = pd.concat([df["Lower CI"], df["Upper CI"]])
        x_min = all_ci.min()
        x_max = all_ci.max()
        pad = (x_max - x_min) * (axis_padding / 100)
        ax.set_xlim(x_min - pad, x_max + pad)

        ax.invert_yaxis()
        fig.tight_layout()

        st.pyplot(fig)

        # PNG Download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")

else:
    st.info("Please upload a file or enter data manually to generate a plot.")
