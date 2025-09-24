import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Latin Countries Regression App", layout="wide")
st.title("Historical Regression Explorer — Latin Countries")
st.markdown("### Created by Felicio De Souza")

# ---------------------
# Generate sample dataset (70 years)
# ---------------------
years = np.arange(1950, 2020 + 1)
countries = ["Brazil", "Mexico", "Argentina"]
categories = ["Population","Unemployment rate","Education level","Life expectancy",
              "Average wealth","Average income","Birth rate","Immigration out","Murder rate"]

data_rows = []
np.random.seed(42)

for country in countries:
    for cat in categories:
        base = 1000 if cat=="Population" else 50
        trend = np.linspace(base, base*2, len(years))
        noise = np.random.normal(0, base*0.05, len(years))
        values = trend + noise
        if cat=="Education level":
            values = np.clip(values / 100, 0, 25)
        elif cat=="Life expectancy":
            values = np.clip(values / 10 + 50, 50, 90)
        elif cat=="Unemployment rate" or cat=="Murder rate":
            values = np.clip(values / 10, 0, 30)
        elif cat=="Birth rate":
            values = np.clip(values / 10, 10, 50)
        elif cat=="Average income" or cat=="Average wealth":
            values = np.clip(values, 1000, 100000)
        elif cat=="Immigration out":
            values = np.clip(values / 10, 0, 10000)

        for y, v in zip(years, values):
            data_rows.append({"Year": int(y), "Country": country, "Category": cat, "Value": float(round(v,2))})

df = pd.DataFrame(data_rows)

# ---------------------
# Sidebar filters
# ---------------------
st.sidebar.header("Options")
category_choice = st.sidebar.selectbox("Select Category", categories)
country_choice = st.sidebar.multiselect("Select Countries", countries, default=["Brazil"])
extrapolate_years = st.sidebar.slider("Extrapolate into the future (years)", 0, 10, 5)
printer_friendly = st.sidebar.checkbox("Printer-Friendly View")
st.sidebar.markdown("---")
st.sidebar.subheader("Average Rate of Change")
year_start = st.sidebar.number_input("Start Year", int(years.min()), int(years.max()), int(years.min()))
year_end = st.sidebar.number_input("End Year", int(years.min()), int(years.max()), int(years.max()))

# ---------------------
# Filter data
# ---------------------
filtered_df = df[(df["Category"]==category_choice) & (df["Country"].isin(country_choice))]

if filtered_df.empty:
    st.warning("No data available for this selection.")
else:
    st.subheader("Raw Data Table")
    pivot_df = filtered_df.pivot(index="Year", columns="Country", values="Value")
    st.dataframe(pivot_df)

    # ---------------------
    # Regression and graph
    # ---------------------
    st.subheader("Polynomial Regression & Graph")
    fig, ax = plt.subplots(figsize=(10,5))
    report_text = ""

    for country in country_choice:
        country_df = filtered_df[filtered_df["Country"]==country].sort_values("Year")
        X = country_df["Year"].values.reshape(-1,1)
        y = country_df["Value"].values

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # Regression predictions
        y_pred = model.predict(X_poly)
        # Extrapolation
        if extrapolate_years > 0:
            future_years = np.arange(X.max()+1, X.max()+extrapolate_years+1).reshape(-1,1)
            y_future = model.predict(poly.transform(future_years))
            ax.plot(future_years, y_future, "--", label=f"{country} Extrapolation")
        else:
            future_years = np.array([])

        ax.scatter(X, y, label=f"{country} Data")
        ax.plot(X, y_pred, label=f"{country} Regression")

        # Regression equation
        coeffs = model.coef_
        intercept = model.intercept_
        eq = f"{intercept:.2f} + " + " + ".join([f"{c:.2e}·x^{i}" for i,c in enumerate(coeffs[1:],1)])
        st.markdown(f"**{country} Regression Equation:** y = {eq}")

        # ---------------------
        # Function analysis and textual explanations
        # ---------------------
        def f(x): return model.predict(poly.transform(np.array(x).reshape(-1,1)))[0] if np.isscalar(x) else model.predict(poly.transform(np.array(x).reshape(-1,1)))

        x_dense = np.linspace(X.min(), X.max()+extrapolate_years, 500)
        y_dense = f(x_dense)
        dy_dx = np.gradient(y_dense, x_dense)

        # Max/Min
        max_idx = np.argmax(y_dense)
        min_idx = np.argmin(y_dense)
        max_year, max_val = int(x_dense[max_idx]), y_dense[max_idx]
        min_year, min_val = int(x_dense[min_idx]), y_dense[min_idx]

        # Fastest increase/decrease
        max_slope_idx = np.argmax(dy_dx)
        min_slope_idx = np.argmin(dy_dx)
        fast_inc_year, fast_inc_val = int(x_dense[max_slope_idx]), y_dense[max_slope_idx]
        fast_dec_year, fast_dec_val = int(x_dense[min_slope_idx]), y_dense[min_slope_idx]

        # Average rate of change
        if year_end > year_start:
            y1 = f(year_start)
            y2 = f(year_end)
            avg_rate = (y2 - y1)/(year_end - year_start)
        else:
            avg_rate = None

        # Construct textual analysis
        text = f"**Analysis for {country}: {category_choice}**\n"
        text += f"- Local maximum in {max_year}, value ≈ {max_val:.2f}\n"
        text += f"- Local minimum in {min_year}, value ≈ {min_val:.2f}\n"
        text += f"- Fastest increase in {fast_inc_year}, value ≈ {fast_inc_val:.2f}\n"
        text += f"- Fastest decrease in {fast_dec_year}, value ≈ {fast_dec_val:.2f}\n"
        if avg_rate is not None:
            text += f"- Average rate of change between {int(year_start)} and {int(year_end)}: {avg_rate:.2f} per year\n"
        if extrapolate_years>0:
            text += f"- Prediction in {int(X.max()+extrapolate_years)}: {f(X.max()+extrapolate_years):.2f}\n"
        st.markdown(text)
        report_text += text + "\n"

    ax.set_xlabel("Year")
    ax.set_ylabel(category_choice)
    ax.set_title(f"{category_choice} over time")
    ax.legend()
    st.pyplot(fig)

    # ---------------------
    # Printer-Friendly PDF
    # ---------------------
    st.subheader("Download Printer-Friendly PDF")
    def create_pdf(dataframe, fig_obj, report_text):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height-50, "Historical Regression Explorer — Latin Countries")
        c.setFont("Helvetica", 10)
        c.drawString(72, height-70, "Created by Felicio De Souza")

        # Report text
        text_obj = c.beginText(72, height-100)
        text_obj.setFont("Helvetica", 9)
        for line in report_text.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)

        # Table
        table_str = dataframe.to_string(index=False)
        text2 = c.beginText(72, height-250)
        text2.setFont("Helvetica", 9)
        for line in table_str.split("\n"):
            text2.textLine(line)
        c.drawText(text2)

        # Graph
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

    pdf_buffer = create_pdf(filtered_df, fig, report_text)

    st.download_button(
        label="📥 Download Printer-Friendly PDF",
        data=pdf_buffer,
        file_name="latin_countries_full_analysis.pdf",
        mime="application/pdf"
    )
