from linearmodels import PanelOLS
import pandas as pd

from data.analysis import MONTH_PRCPT_DICT, MONTH_TAVG_DICT, PANEL_OUTPUT_PATH, TABLES_DIR

def run_panel_regression_monthly(df: pd.DataFrame, exog_vars_names: list) -> PanelOLS:
    df = df.copy()
    df.set_index(["region_name", "year"], inplace=True)

    dep_var_names = ["yield_lb_acre"]

    exog_vars = df[exog_vars_names]
    dep_var = df[dep_var_names]

    model = PanelOLS(dep_var, exog_vars, entity_effects=True, time_effects=True)

    # cluster standard errors by entity (region_name) because differences are likely correlated within regions over time
    results = model.fit(cov_type="clustered", cluster_entity=True)

    print(results.summary)

    return results



if __name__ == "__main__":

    print("Running panel regression...")

    monthly_panel_df = pd.read_csv(TABLES_DIR / "pivoted_monthly_weather.csv")


    exog_vars_names = list(MONTH_TAVG_DICT.values()) + list(MONTH_PRCPT_DICT.values())
    run_panel_regression_monthly(monthly_panel_df, exog_vars_names)

    exog_vars_names = ["prcp_total", "tavg_mean"]
    run_panel_regression_monthly(monthly_panel_df, exog_vars_names)

