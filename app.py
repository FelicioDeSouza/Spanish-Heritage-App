import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml

st.title("Historical Regression Explorer — Latin countries (World Bank data)")
st.markdown("### Created by Felicio De Souza")

# Load World Bank dataset
@st.cache_data
def load_data():
    data = fetch_openml(data_id=41021, as_frame=True)  # World Bank Development Indicators
    return data.frame

df = load_data()

# Wealthiest Latin American countries (by GDP size)
latin_countries = ["Brazil", "Mexico", "Argentina"]

# Indicators (restricted to those with reasonably complete data)
categories = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Birth rate": "SP.DYN.CBRT.IN"
}

# Sidebar controls
st.sidebar.header("Options")
category_choice = st.sidebar.selectbox("Select category", list(categories.keys()))
year_step = st.sidebar.slider("Graph increment (years)", 1, 10, 5)
extrapolate_years = st.sidebar.slider("Extrapolate into the future (years)", 0, 50, 10)
multi_country = st.sidebar.multiselect("Select countries", latin_countries, default=["Brazil"])
printer_friendly = st.sidebar.checkbox("Printer Friendly View")

code = categories[category_choice]
data_filtered = df[df["country"].isin(multi_country) & (df["indicator"] == code)]

if data_filtered.empty:
    st.warning("No data available for this selection.")
else:
    pivot_df = data_filtered.pivot(index="year", columns="country", values="value").dropna()
    pivot_df.index = pivot_df.index.astype(int)

    st.subheader("Raw Data (editable)")
    edited_df = st.data_editor(pivot_df)

    years = edited_df.index.values.reshape(-1, 1)

    plt.figure(figsize=(10,6))
    report = ""  # For printer-friendly text

    for country in edited_df.columns:
        values = edited_df[country].values
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(years)
        model = LinearRegression().fit(X_poly, values)
        y_pred = model.predict(X_poly)

        # Extrapolation
        last_year = years.max()
        future_years = np.arange(last_year+1, last_year+extrapolate_years+1).reshape(-1,1)
        future_pred = model.predict(poly.transform(future_years))

        # Plot data and regression
        plt.scatter(years, values, label=f"{country} Data")
        plt.plot(years, y_pred, label=f"{country} Regression")
        if extrapolate_years > 0:
            plt.plot(future_years, future_pred, "--", label=f"{country} Extrapolation")

        # Display regression equation
        coeffs = model.coef_
        intercept = model.intercept_
        st.markdown(f"**{country} Regression Equation:** y = {intercept:.2f} + " +
                    " + ".join([f"{coeff:.2e}·x^{i}" for i, coeff in enumerate(coeffs[1:], start=1)]))

        # Add to printer friendly report
        report += f"\n{country} — {category_choice}\n"
        report += f"Data years: {edited_df.index.min()} to {edited_df.index.max()}\n"
        report += f"Last observed value: {edited_df[country].iloc[-1]} in {edited_df.index.max()}\n"
        if extrapolate_years > 0:
            report += f"Projection {extrapolate_years} years ahead included.\n"

    plt.xlabel("Year")
    plt.ylabel(category_choice)
    plt.title(f"{category_choice} over time")
    plt.legend()
    st.subheader('Graph of data and polynomial regression model')
    st.pyplot(plt)

    # Printer Friendly View
    if printer_friendly:
        st.subheader("Printer Friendly Report")
        st.text(report)

        # Download button
        st.download_button(
            label="Download Printer Friendly Report",
            data=report,
            file_name="printer_friendly_report.txt",
            mime="text/plain"
        )
