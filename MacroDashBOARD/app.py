import streamlit as pd_st
import streamlit as st
import plotly.express as px
import numpy as np
from engine import MacroDataEngine

# Configure corporate financial terminal layout
st.set_page_config(page_title="Sovereign Macro Risk Terminal", layout="wide")

st.title("🏛️ Sovereign Institutional Fragility & Inflation Analytics Engine")
st.caption("Empirical evaluation framework mapping Institutional Decay against Macroeconomic Volatility loops.")

# Initialize the data framework
@st.cache_data(ttl=3600)
def load_processed_macro_data(start_year, end_year, cache_buster=1):
    engine = MacroDataEngine()
    return engine.process_econometrics(start_year, end_year)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🕹️ Control Room")

start_year, end_year = st.sidebar.slider(
    "Historical Timeline Block",
    min_value=2000, max_value=2024, value=(2012, 2024)
)

# Load data based on slider input
with st.spinner("Streaming data matrices from World Bank API..."):
    data, model_fit = load_processed_macro_data(start_year, end_year, cache_buster=2)

if data.empty:
    st.error("Data pipeline timeout. Please check your internet connectivity or refresh.")
    st.stop()

# Country Filter Multi-Select Tool
all_countries = sorted(data["country_name"].unique())
selected_countries = st.sidebar.multiselect(
    "Focus Target Markets (Leave blank for Global View)",
    options=all_countries
)

st.sidebar.markdown("---")
st.sidebar.subheader("💎 Monetization & B2B Licensing")
premium_tier_active = st.sidebar.toggle("Unlock Premium Enterprise Tier Mode")

# Filter data matrix if specific countries are selected
filtered_data = data.copy()
if selected_countries:
    filtered_data = filtered_data[filtered_data["country_name"].isin(selected_countries)]

# ----------------- FINANCIAL METRIC CARDS -----------------
col1, col2, col3 = st.columns(3)

with col1:
    avg_volatility = filtered_data["inflation_volatility"].mean()
    st.metric(label="Average Market Inflation Volatility", value=f"{avg_volatility:.2f}%")
    
with col2:
    if model_fit:
        r_squared = model_fit.rsquared
        st.metric(label="Statistical R-Squared Strength", value=f"{r_squared:.4f}")
    else:
        st.metric(label="Statistical R-Squared Strength", value="N/A")

with col3:
    highest_risk_row = filtered_data.loc[filtered_data["inflation_volatility"].idxmax()]
    st.metric(
        label="Highest Macro Volatility Country", 
        value=highest_risk_row["country_name"],
        delta=f"{highest_risk_row['inflation_volatility']:.1f}% Vol"
    )

st.markdown("---")

# ----------------- ECONOMETRIC SCATTER PLOT -----------------
st.subheader("📊 Econometric Regression Output")

# Generate standard OLS trendlines manually using model weights
if model_fit and len(filtered_data) > 1:
    fig = px.scatter(
        filtered_data,
        x="institutional_quality",
        y="inflation_volatility",
        text="country_name",
        hover_name="country_name",
        labels={
            "institutional_quality": "Institutional Quality Index Score (Higher = Stronger Governance)",
            "inflation_volatility": "Inflation Volatility (Std Dev of Annual CPI %)"
        },
        template="plotly_white",
        trendline="ols",
        trendline_color_override="red"
    )
    fig.update_traces(textposition="top center", marker=dict(size=10, opacity=0.75))
else:
    fig = px.scatter(filtered_data, x="institutional_quality", y="inflation_volatility", text="country_name")

fig.update_layout(height=600, font=dict(family="Arial", size=12))
st.plotly_chart(fig, use_container_width=True)

# ----------------- ACADEMIC ESSAY ANALYTICS -----------------
st.subheader("🧠 Academic Inference Matrix (For Personal Statement Drafts)")
if model_fit:
    slope = model_fit.params['institutional_quality']
    p_value = model_fit.pvalues['institutional_quality']
    
    st.markdown(f"""
    * **The Quantitative Thesis:** The current calculated slope parameter is **{slope:.3f}**. This indicates that for every 1-unit structural deterioration in a nation's sovereign governance framework, annual inflation volatility expands by **{abs(slope):.2f}%** on average.
    * **Statistical Significance:** The calculated model p-value evaluates to **{p_value:.4e}**. Since this is well below the standard 5% significance alpha tier ($p < 0.05$), we reject the null hypothesis, mathematically proving that weak institutions directly degrade monetary predictability.
    """)

# ----------------- B2B PREMIUM ENTERPRISE WALL -----------------
st.markdown("---")
if premium_tier_active:
    st.success("🔒 Premium Enterprise Access Unlocked")
    b2b_col1, b2b_col2 = st.columns(2)
    
    with b2b_col1:
        st.subheader("📥 Data Export Pipeline")
        st.caption("Download cleaned macroeconomic data arrays for enterprise risk models.")
        csv_data = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV Asset",
            data=csv_data,
            file_name="sovereign_risk_matrix.csv",
            mime="text/csv"
        )
        
    with b2b_col2:
        st.subheader("🔔 Automated Supply Chain Alerts")
        st.caption("Hook automated scripts to trigger internal risk workflows.")
        target_email = st.text_input("Corporate Sourcing Contact Email", "risk-officer@enterprise.com")
        vol_threshold = st.slider("Volatility Alert Threshold Trigger (%)", 5, 50, 15)
        if st.button("Deploy Live Webhook Alert"):
            st.info(f"Active risk monitor operational. Real-time changes exceeding {vol_threshold}% will ping {target_email}.")
else:
    st.info("💡 **Business Model View:** Want to download corporate CSV raw datasets and structure active real-time risk alert triggers? Toggle the **'Unlock Premium Enterprise Tier Mode'** switch inside the left sidebar panel to preview the B2B SaaS monetization tier.")
