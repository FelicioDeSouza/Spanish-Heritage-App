import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------------------
# Page setup
# ---------------------
st.set_page_config(page_title="Spanish Heritage App", layout="wide")
st.title("Spanish Heritage Timeline")
st.write("Created by **Felicio De Souza**")

# ---------------------
# Load data
# ---------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spanish_heritage.csv")
    except FileNotFoundError:
        st.error("The data file `spanish_heritage.csv` is missing.")
        return pd.DataFrame()

    # Clean up columns
    df.columns = df.columns.str.strip().str.lower()

    # Ensure 'year' column is numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).astype({"year": "int"})
    else:
        st.warning("No 'year' column found in dataset.")
    return df

df = load_data()

# ---------------------
# Sidebar filters
# ---------------------
if not df.empty:
    years = sorted(df["year"].unique())
    categories = df["category"].unique() if "category" in df.columns else []

    year_selected = st.sidebar.selectbox("Select Year", options=["All"] + list(years))
    category_selected = st.sidebar.selectbox("Select Category", options=["All"] + list(categories))

    # Filtering logic
    filtered_df = df.copy()
    if year_selected != "All":
        filtered_df = filtered_df[filtered_df["year"] == year_selected]
    if category_selected != "All" and "category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["category"] == category_selected]

    # ---------------------
    # Display data
    # ---------------------
    if filtered_df.empty:
        st.warning("No data available for this selection. Showing full dataset instead.")
        st.dataframe(df)
        display_df = df
    else:
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)
        display_df = filtered_df

        # ---------------------
        # Graph
        # ---------------------
        if "year" in display_df.columns:
            st.subheader("Data Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            display_df["year"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_title("Events by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count of Events")
            st.pyplot(fig)
        else:
            fig = None

    # ---------------------
    # Printer-friendly PDF
    # ---------------------
    st.subheader("Printer-Friendly PDF Download")

    def create_pdf(dataframe, fig_obj=None):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 50, "Spanish Heritage Timeline (Printer-Friendly)")
        c.setFont("Helvetica", 10)
        c.drawString(72, height - 70, f"Created by Felicio De Souza")

        # Add data table
        text = c.beginText(72, height - 100)
        text.setFont("Helvetica", 9)
        if dataframe.empty:
            text.textLine("No data available.")
        else:
            data_str = dataframe.to_string(index=False)
            for line in data_str.split("\n"):
                text.textLine(line)
        c.drawText(text)

        # Add graph if available
        if fig_obj is not None:
            img_buffer = BytesIO()
            fig_obj.savefig(img_buffer, format="png", bbox_inches="tight")
            img_buffer.seek(0)
            image = ImageReader(img_buffer)
            c.showPage()
            c.drawImage(image, 72, 200, width=width-144, preserveAspectRatio=True, mask="auto")

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf(display_df, fig if "fig" in locals() else None)

    st.download_button(
        label="📥 Download Printer-Friendly View (PDF)",
        data=pdf_buffer,
        file_name="spanish_heritage_timeline.pdf",
        mime="application/pdf"
    )

else:
    st.info("No data loaded. Please add `spanish_heritage.csv` to the app folder.")


