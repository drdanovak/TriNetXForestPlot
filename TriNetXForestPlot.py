import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Configure page
st.set_page_config(layout="wide")
st.title("🌲 Forest Plot Generator")

# Choose input method
input_mode = st.radio("Select data input method:", ["📤 Upload file", "✍️ Manual entry"], horizontal=True)

# Define the expected column names
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
    # Sidebar customization
    st.sidebar.header("⚙️ Customize Plot")
    color_scheme = st.sidebar.selectbox("Color Scheme", ["Color", "Black & White"])
    plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)

    if st.button("📊 Generate Forest Plot"):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.6))
        y_pos = range(len(df))

        for i, row in df.iterrows():
            ax.plot([row["Lower CI"], row["Upper CI"]], [i, i],
                    color='black' if color_scheme == "Black & White" else 'blue', linewidth=1.5)
            ax.plot(row["Effect Size"], i, 'o',
                    color='black' if color_scheme == "Black & White" else 'red')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Outcome"])
        ax.axvline(1, color='grey', linestyle='--')
        ax.set_xlabel(x_axis_label)
        ax.set_title(plot_title)
        if show_grid:
            ax.grid(True, axis='x', linestyle=':', alpha=0.7)
        ax.invert_yaxis()
        fig.tight_layout()

        st.pyplot(fig)

        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")
else:
    st.info("Please upload a file or enter data manually to generate a plot.")
