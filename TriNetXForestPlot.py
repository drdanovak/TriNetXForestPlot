import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# App config
st.set_page_config(layout="wide")
st.title("🌲 Forest Plot Generator")

# Upload file
uploaded_file = st.file_uploader("📂 Upload your Excel or CSV file", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

# Read file
if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Validate input
required_cols = ["Outcome", "Effect Size", "Lower CI", "Upper CI"]
if not all(col in df.columns for col in required_cols):
    st.error(f"❌ Your file must include the following columns: {required_cols}")
    st.stop()

# Customization options
st.sidebar.header("⚙️ Customize Plot")
color_scheme = st.sidebar.selectbox("Color Scheme", ["Color", "Black & White"])
plot_title = st.sidebar.text_input("Plot Title", value="Forest Plot")
x_axis_label = st.sidebar.text_input("X-axis Label", value="Effect Size (RR / OR / HR)")
show_grid = st.sidebar.checkbox("Show Grid", value=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, len(df) * 0.6))
y_pos = range(len(df))

# Plot horizontal lines for CI
for i, row in df.iterrows():
    ax.plot([row["Lower CI"], row["Upper CI"]], [i, i], color='black' if color_scheme == "Black & White" else 'blue', linewidth=1.5)
    ax.plot(row["Effect Size"], i, 'o', color='black' if color_scheme == "Black & White" else 'red')

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(df["Outcome"])
ax.axvline(1, color='grey', linestyle='--')
ax.set_xlabel(x_axis_label)
ax.set_title(plot_title)
if show_grid:
    ax.grid(True, axis='x', linestyle=':', alpha=0.7)
ax.invert_yaxis()
fig.tight_layout()

# Display plot
st.pyplot(fig)

# Download PNG
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300)
st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")
