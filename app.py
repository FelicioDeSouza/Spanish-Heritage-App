import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    else:
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

        # ---------------------
        # Graph
        # ---------------------
        if "year" in filtered_df.columns:
            st.subheader("Data Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            filtered_df["year"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_title("Events by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count of Events")
            st.pyplot(fig)

    # ---------------------
    # Printer-friendly view
    # ---------------------
    st.subheader("Printer-Friendly View")
    with st.expander("Open Printer-Friendly View"):
        st.write(filtered_df if not filtered_df.empty else df)

else:
    st.info("No data loaded. Please add `spanish_heritage.csv` to the app folder.")

