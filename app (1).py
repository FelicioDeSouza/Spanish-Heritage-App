
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
import json

st.set_page_config(page_title="Historical Regression Explorer", layout="wide")

st.title("Historical Regression Explorer — Latin countries (World Bank data)")

st.markdown("""
This app fetches historical series from the World Bank (years usually available from 1960 onward) for several Latin American countries, fits a polynomial regression (degree ≥ 3), and performs function analysis (extrema, increasing/decreasing intervals, fastest change, extrapolation, interpolation, and comparison).
**Note:** many international series are available from 1960 on; that means ~60+ years of data (not the full 70 years) depending on the indicator. Sources: World Bank / FRED.  
See the app footer for full source links.
""")

# --- Configuration / available countries and indicators ---
COUNTRIES = {
    "Chile": "CL",
    "Panama": "PA",
    "Uruguay": "UY"
}

INDICATORS = {
    "Population": "SP.POP.TOTL",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Birth rate (crude, per 1000)": "SP.DYN.CBRT.IN",
    "GDP per capita (current US$) — average income proxy": "NY.GDP.PCAP.CD",
    "Unemployment rate (% of labor force) — may be sparse": "SL.UEM.TOTL.ZS",
    "Murder rate (homicides per 100k) — may be sparse": "VC.IHR.PSRC.P5"  # placeholder; may not exist for all years
}

st.sidebar.header("Data & model options")
country_sel = st.sidebar.multiselect("Select country (one or more):", list(COUNTRIES.keys()), default=["Chile"])
indicator_sel = st.sidebar.selectbox("Select category (indicator):", list(INDICATORS.keys()))
degree = st.sidebar.slider("Polynomial degree (≥3)", min_value=3, max_value=8, value=3)
step = st.sidebar.slider("Plot increments (years between plotted ticks/points)", min_value=1, max_value=10, value=1)
extrapolate_years = st.sidebar.slider("Extrapolate how many years into the future?", 0, 50, 10)
show_extrapolated = st.sidebar.checkbox("Show extrapolated part in different color", value=True)

st.sidebar.markdown("---")
st.sidebar.write("If you have a CSV with year,value columns you can upload it and compare it with the World Bank data:")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# --- Helper: fetch World Bank data ---
@st.cache_data(show_spinner=False)
def fetch_wb(country_code, indicator):
    # World Bank API: returns JSON pages
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=2000"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, list) or len(data) < 2:
        return None
    records = data[1]
    rows = []
    for rec in records:
        year = rec.get("date")
        val = rec.get("value")
        if val is None:
            continue
        try:
            rows.append({"year": int(year), "value": float(val)})
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("year")
    return df

# --- Load data for selected countries ---
datasets = {}
for c in country_sel:
    code = COUNTRIES[c]
    ind = INDICATORS[indicator_sel]
    df = fetch_wb(code, ind)
    if df is None or df.empty:
        st.warning(f"No data available from World Bank API for {c} / {indicator_sel}. You can upload your own CSV to compare or try another indicator.")
    else:
        datasets[c] = df

# Load uploaded CSV if provided
uploaded_df = None
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        if {"year","value"}.issubset(uploaded_df.columns):
            uploaded_df = uploaded_df[["year","value"]].dropna().sort_values("year")
            datasets["Uploaded data"] = uploaded_df
        else:
            st.sidebar.error("Uploaded CSV must contain 'year' and 'value' columns.")
            uploaded_df = None
    except Exception as e:
        st.sidebar.error("Could not read uploaded CSV: " + str(e))
        uploaded_df = None

if not datasets:
    st.stop()

# --- Display raw data table (editable) for the first selected country ---
first_country = list(datasets.keys())[0]
st.subheader(f"Raw data (editable) — {first_country} / {indicator_sel}")
editable_df = st.data_editor(datasets[first_country].reset_index(drop=True), num_rows="dynamic")
# If user edits table, use that as source for that country
datasets[first_country] = editable_df.sort_values("year").reset_index(drop=True)

# --- Modeling utilities ---
def fit_polynomial(years, values, deg):
    # Fit poly in numeric year (e.g., 1960 -> 1960). Use numpy.polyfit for simplicity and stable evaluation.
    coeffs = np.polyfit(years, values, deg)
    # Return coeffs highest->lowest
    return coeffs

def poly_eval(coeffs, x):
    return np.polyval(coeffs, x)

def poly_derivative(coeffs):
    # coeffs highest->lowest
    deg = len(coeffs)-1
    der = np.array([coeffs[i]*(deg-i) for i in range(len(coeffs)-1)])
    return der

def format_equation(coeffs, var_name="x"):
    terms = []
    deg = len(coeffs)-1
    for i,c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        coef = f"{c:.6g}"
        if power == 0:
            terms.append(f"{coef}")
        elif power == 1:
            terms.append(f"{coef}*{var_name}")
        else:
            terms.append(f"{coef}*{var_name}^{power}")
    if not terms:
        return "f(x)=0"
    return "f(x) = " + " + ".join(terms)

st.sidebar.markdown("---")
st.sidebar.write("Model & analysis actions")
if st.sidebar.button("Fit model and analyze"):
    results = {}
    fig, ax = plt.subplots(figsize=(10,6))
    # Determine global year range including extrapolation
    all_years = np.concatenate([df["year"].values for df in datasets.values()])
    min_year, max_year = int(all_years.min()), int(all_years.max())
    plot_years = np.arange(min_year, max_year+1, step)
    # extend for extrapolation
    ext_max = max_year + extrapolate_years
    full_years = np.arange(min_year, ext_max+1, step)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i,(label,df) in enumerate(datasets.items()):
        if df is None or df.empty:
            continue
        years = df["year"].values
        vals = df["value"].values
        # Fit
        try:
            coeffs = fit_polynomial(years, vals, degree)
        except np.RankWarning:
            coeffs = np.polyfit(years, vals, degree)
        # Evaluate model on plot_years and full_years
        y_plot = poly_eval(coeffs, plot_years)
        y_full = poly_eval(coeffs, full_years)
        # Plot scatter
        ax.scatter(years, vals, label=f"{label} (data)", alpha=0.7)
        # Plot fitted curve for in-sample (up to max_year)
        ax.plot(plot_years, poly_eval(coeffs, plot_years), linestyle='-', label=f"{label} (fit)", color=colors[i%len(colors)])
        # Plot extrapolated portion if requested
        if show_extrapolated and extrapolate_years>0:
            # in-sample end index
            in_sample_mask = full_years<=max_year
            extrap_mask = full_years>max_year
            ax.plot(full_years[in_sample_mask], y_full[in_sample_mask], color=colors[i%len(colors)])
            ax.plot(full_years[extrap_mask], y_full[extrap_mask], linestyle='--', color=colors[i%len(colors)], alpha=0.8)
        # Save result
        results[label] = {"coeffs": coeffs, "years": years, "vals": vals, "min_year": min_year, "max_year": max_year}
    ax.set_xlabel("Year")
    ax.set_ylabel(indicator_sel)
    ax.legend()
    st.pyplot(fig)

    # Show equations and analysis
    st.header("Model equations & function analysis")
    for label,res in results.items():
        coeffs = res["coeffs"]
        st.subheader(label)
        st.code(format_equation(coeffs, var_name="t"))
        # Domain & range (approx from data and model)
        dom_min, dom_max = int(res["min_year"]), int(res["max_year"])+extrapolate_years
        st.write(f"Domain used (years): {dom_min} to {dom_max}.")
        # Range from model over domain
        domain_grid = np.linspace(dom_min, dom_max, 1000)
        vals_grid = poly_eval(coeffs, domain_grid)
        rng_min, rng_max = float(np.nanmin(vals_grid)), float(np.nanmax(vals_grid))
        st.write(f"Approximate range of the model on the shown domain: {rng_min:.3f} to {rng_max:.3f} ({indicator_sel}).")
        # Derivative analysis
        der = poly_derivative(coeffs)
        der2 = poly_derivative(der)
        # find critical points (roots of derivative) numerically
        # use numpy.roots on derivative coefficients
        crit_points = np.roots(der)
        # keep real roots within domain
        real_crit = [float(r.real) for r in crit_points if abs(r.imag)<1e-6 and dom_min-5 <= r.real <= dom_max+5]
        st.write("Critical points (where derivative = 0) within/near domain (years): ", real_crit if real_crit else "None found")
        # evaluate second derivative at critical points to classify
        for cp in real_crit:
            sd_val = np.polyval(der2, cp)
            fval = np.polyval(coeffs, cp)
            kind = "inflection or higher-order" 
            if sd_val > 1e-8:
                kind = "local minimum"
            elif sd_val < -1e-8:
                kind = "local maximum"
            st.write(f"- Year ~ {cp:.3f}: f({int(cp)}) ≈ {fval:.3f} → {kind}.")
        # When function increases/decreases: examine derivative on grid
        der_vals = np.polyval(der, domain_grid)
        increasing_mask = der_vals > 0
        # find first and last intervals where increasing
        inc_ranges = []
        dec_ranges = []
        # compress masks into intervals
        def masks_to_intervals(grid, mask):
            intervals=[]
            start=None
            for x,m in zip(grid, mask):
                if m and start is None:
                    start = x
                if not m and start is not None:
                    intervals.append((start, prev))
                    start=None
                prev = x
            if start is not None:
                intervals.append((start, prev))
            return intervals
        inc_intervals = masks_to_intervals(domain_grid, increasing_mask)
        dec_intervals = masks_to_intervals(domain_grid, ~increasing_mask)
        st.write("Approximate intervals where the model is increasing (years):", [ (int(a),int(b)) for a,b in inc_intervals[:5] ] if inc_intervals else "None found")
        st.write("Approximate intervals where the model is decreasing (years):", [ (int(a),int(b)) for a,b in dec_intervals[:5] ] if dec_intervals else "None found")
        # fastest increase / decrease: maxima/minima of derivative
        max_der_idx = np.nanargmax(der_vals)
        min_der_idx = np.nanargmin(der_vals)
        year_max_inc = domain_grid[max_der_idx]; year_max_dec = domain_grid[min_der_idx]
        st.write(f"The model is increasing fastest at year ~ {year_max_inc:.1f} (derivative ≈ {der_vals[max_der_idx]:.4f}).")
        st.write(f"The model is decreasing fastest at year ~ {year_max_dec:.1f} (derivative ≈ {der_vals[min_der_idx]:.4f}).")
        # Conjectures: very generic — user must inspect
        st.markdown("**Conjectures for significant changes**")
        st.write("Look at known historical events, policy changes, economic crises, or demographic shifts around the years where the model shows large changes. Example conjectures: major migration waves, economic booms or recessions, large public-health events, or changes in data collection methods.")
        # Extrapolation example: predict at year beyond max_year
        future_year = dom_max + 10
        fut_val = float(np.polyval(coeffs, future_year))
        st.write(f"Extrapolation example: According to the model, {indicator_sel} for {label} is predicted to be {fut_val:.3f} in year {future_year}. (Treat with caution — extrapolation uncertainty is high.)")
        st.markdown("---")

    # --- Interpolation / extrapolation tool ---
    st.header("Interpolation / Extrapolation and rate-of-change tools")
    sel_label = st.selectbox("Choose which fitted country/model to query:", list(results.keys()))
    model = results[sel_label]
    coeffs = model["coeffs"]
    query_year = st.number_input("Enter a year to estimate f(year):", min_value=1900, max_value=2100, value=int(model["max_year"]+5))
    est = float(np.polyval(coeffs, query_year))
    st.write(f"Model estimate: f({int(query_year)}) = {est:.6f} ({indicator_sel})")
    st.write("Note whether this year is within the data range (interpolation) or beyond it (extrapolation):",
             "extrapolation" if query_year>model["max_year"] else "interpolation")

    y1 = st.number_input("Average rate of change - start year:", min_value=1900, max_value=2100, value=int(model["min_year"]))
    y2 = st.number_input("Average rate of change - end year:", min_value=1900, max_value=2100, value=int(model["max_year"]))
    if y2 != y1:
        val1 = float(np.polyval(coeffs, y1)); val2 = float(np.polyval(coeffs, y2))
        avg_rate = (val2 - val1)/(y2 - y1)
        st.write(f"Average rate of change from {int(y1)} to {int(y2)}: {avg_rate:.6f} ({indicator_sel} per year)")

    # Printer-friendly download: simple HTML
    if st.button("Generate printer-friendly HTML report"):
        html_parts = []
        html_parts.append(f"<h1>Regression report — {indicator_sel}</h1>")
        for label,res in results.items():
            html_parts.append(f"<h2>{label}</h2>")
            html_parts.append(f"<pre>{format_equation(res['coeffs'],'t')}</pre>")
        full_html = "<html><body>" + "\n".join(html_parts) + "</body></html>"
        b = full_html.encode('utf-8')
        b64 = base64.b64encode(b).decode()
        href = f"data:text/html;base64,{b64}"
        st.markdown(f"[Download printer-friendly HTML report]({href})", unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Data sources:** World Bank API (indicators) — e.g., population (SP.POP.TOTL), life expectancy (SP.DYN.LE00.IN), birth rate (SP.DYN.CBRT.IN), GDP per capita (NY.GDP.PCAP.CD). These series are typically available from 1960 onward. See https://data.worldbank.org for more.")
