import pandas as pd
import requests

class MacroDataEngine:
    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2/country"

    def fetch_indicator_data(self, indicator_code, start_year=2010, end_year=2024):
        """Fetches international data using clean, single-page API requests."""
        # Requesting data for a specific group of prominent global economies
        target_countries = "usa;gbr;deu;fra;jpn;can;aus;bra;chn;mex"
        url = f"{self.base_url}/{target_countries}/indicator/{indicator_code}"
        params = {
            "date": f"{start_year}:{end_year}",
            "format": "json",
            "per_page": 1000
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()

            # Safeguard parsing loop against missing API elements
            if not isinstance(raw_data, list) or len(raw_data) < 2:
                return pd.DataFrame()

            records = raw_data[1]
            if not isinstance(records, list):
                return pd.DataFrame()

            parsed_rows = []
            for item in records:
                if not isinstance(item, dict):
                    continue
                c_data = item.get("country", {})
                c_id = c_data.get("id")
                if c_id:
                    parsed_rows.append({
                        "country_iso3": c_id.upper(),
                        "country_name": c_data.get("value", "Unknown"),
                        "year": int(item["date"]) if item.get("date") else start_year,
                        "value": item["value"] if item.get("value") is not None else None
                    })
            return pd.DataFrame(parsed_rows)
        except Exception as e:
            print(f"Error fetching live indicator {indicator_code}: {e}")
            return pd.DataFrame()

    def process_econometrics(self, start_year=2010, end_year=2024):
        """Aggregates active metrics and builds linear regression output."""
        import statsmodels.api as sm
        import numpy as np
        
        # Use two active, globally supported live indicator keys
        df_gov = self.fetch_indicator_data("NY.GDP.MKTP.CD", start_year, end_year)
        df_inf = self.fetch_indicator_data("FP.CPI.TOTL.ZG", start_year, end_year)

        if df_gov.empty or df_inf.empty:
            return pd.DataFrame(), None

        # Clean and group elements
        gov_clean = df_gov.dropna(subset=["value"]).groupby(["country_name", "country_iso3"])["value"].mean().reset_index()
        gov_clean.rename(columns={"value": "institutional_quality"}, inplace=True)
        gov_clean["institutional_quality"] = np.log10(gov_clean["institutional_quality"] + 1)

        inf_clean = df_inf.dropna(subset=["value"]).groupby(["country_name", "country_iso3"])["value"].std().reset_index()
        inf_clean.rename(columns={"value": "inflation_volatility"}, inplace=True)

        merged_df = pd.merge(gov_clean, inf_clean, on=["country_name", "country_iso3"], how="inner")
        merged_df = merged_df[merged_df["inflation_volatility"] < 50]

        regression_results = None
        if len(merged_df) > 2:
            X = merged_df["institutional_quality"]
            Y = merged_df["inflation_volatility"]
            X_with_constant = sm.add_constant(X)
            model = sm.OLS(Y, X_with_constant)
            regression_results = model.fit()

        return merged_df, regression_results
