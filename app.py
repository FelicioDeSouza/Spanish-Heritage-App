import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import wbgapi as wb

st.title("Historical Regression Explorer — Latin countries (World Bank data)")
st.markdown("### Created by Felicio De Souza")

# Wealthiest Latin American countries
latin_countries = {"Brazil": "BRA", "Mexico": "MEX", "Argentina": "ARG"}

# Indicators with World Bank codes
categories = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Birth rate": "SP.DYN.CBRT.IN"
}

# Sidebar
st.sidebar.header("Options")
category_choice = st.sidebar.selectbox("Select category", list(categories.keys()))
year_step = st.sidebar.slider("Graph increment (years)", 1, 10, 5)
extrapolate_years = st.sidebar.slider("Extrapolate into the future (years)", 0, 50, 10)
multi_country = st.sidebar.multiselect("Select countries", list(latin_countries.keys()), default=["Brazil"])
printer_friendly = st.sidebar.checkbox("Printer Friendly View")

# Fetch data from World Bank
indicator = categories[category_choice]
df = wb.data.DataFrame(indicator, latin_countries.values(), mrv=70)  # most recent 70 years
df = df.T.reset_index().rename(columns={"index": "year"})
df["year"] = df["year"].astype(int)

# Keep only selected countries
df = df[["year"] + [latin_countries[c] for c in multi_country]]
df = df.rename(columns={v: k for k, v in latin_countries.items()})

if df.empty:
    st.warning("No data available for this selection.")
else:
    df = df.dropna()
    df = df.set_index("year")

    st.subheader("Raw Data (editable)")
    edited_df = st.data_editor(df)

    years = edited_df.index.values.reshape(-1, 1)

    plt.figure(figsize=(10,6))
    report = ""

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

        # Plot
        plt.scatter(years, values, label=f"{country} Data")
        plt.plot(years, y_pred, label=f"{country} Regression")
        if extrapolate_years > 0:
            plt.plot(future_years, future_pred, "--", label=f"{country} Extrapolation")

        # Equation
        coeffs = model.coef_
        intercept = model.intercept_
        st.markdown(f"**{country} Regression Equation:** y = {intercept:.2f} + " +
                    " + ".join([f"{coeff:.2e}·x^{i}" for i, coeff in enumerate(coeffs[1:], start=1)]))

        # Add to report
        report += f"\n{country} — {category_choice}\n"
        report += f"Years: {edited_df.index.min()}–{edited_df.index.max()}\n"
        report += f"Last observed: {edited_df[country].iloc[-1]} in {edited_df.index.max()}\n"
        if extrapolate_years > 0:
            report += f"Projection {extrapolate_years} years ahead included.\n"

    plt.xlabel("Year")
    plt.ylabel(category_choice)
    plt.title(f"{category_choice} over time")
    plt.legend()
    st.subheader("Graph of data and polynomial regression model")
    st.pyplot(plt)

    if printer_friendly:
        st.subheader("Printer Friendly Report")
        st.text(report)
        st.download_button(
            label="Download Printer Friendly Report",
            data=report,
            file_name="printer_friendly_report.txt",
            mime="text/plain"
        )
