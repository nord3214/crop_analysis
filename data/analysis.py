from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.parse_data import item_names

COMBINED_DATA_PATH = "data/final/combined_dataset.csv"
PANEL_OUTPUT_PATH = "data/final/county_year_panel.csv"
FIGURES_DIR = Path("data/output/figures")
TABLES_DIR = Path("data/output/tables")

CROP_COLUMNS = ["County ANSI"] + list(item_names.values())

# NOAA nClimGrid-Daily values are Celsius / mm.
EXTREME_HEAT_THRESHOLD_C = 32.0  # ~90F
HEAVY_RAIN_THRESHOLD_MM = 25.0  # ~1 inch/day


def load_combined_data(path: str = COMBINED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df[df["yield_lb_acre"].notna()]
    return df


def build_growing_season_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per county-year: deduplicated crop values + aggregated growing-season weather."""
    crop = (
        df.groupby(["region_name", "year"])[CROP_COLUMNS]
        .first()
        .reset_index()
    )

    weather = (
        df.groupby(["region_name", "year"])
        .agg(
            tavg_mean=("tavg", "mean"),
            tmax_mean=("tmax", "mean"),
            tmin_mean=("tmin", "mean"),
            prcp_total=("prcp", "sum"),
            extreme_heat_days=("tmax", lambda s: (s > EXTREME_HEAT_THRESHOLD_C).sum()),
            heavy_rain_days=("prcp", lambda s: (s > HEAVY_RAIN_THRESHOLD_MM).sum()),
        )
        .reset_index()
    )

    return crop.merge(weather, on=["region_name", "year"], how="inner")


def build_monthly_weather(df: pd.DataFrame) -> pd.DataFrame:
    """One row per county-year-month, used to check whether timing within the season matters."""
    return (
        df.groupby(["region_name", "year", "month"])
        .agg(
            tavg_mean=("tavg", "mean"),
            prcp_total=("prcp", "sum"),
        )
        .reset_index()
    )

def statistics_by_county(panel: pd.DataFrame) -> pd.DataFrame:

    stat_cols = ["number of years with yield data per county", "mean yield (lb/acre)", "std dev of yield (lb/acre)"]

    stats = (
        panel.groupby("region_name")["yield_lb_acre"]
        .agg(
            number_of_years="count",
            mean_yield="mean",
            std_dev_yield="std",
        )
        .rename(columns={
            "number_of_years": stat_cols[0],
            "mean_yield": stat_cols[1],
            "std_dev_yield": stat_cols[2],
        })
    )

    stats.to_csv(TABLES_DIR / "statistics_by_county.csv")


def statistics_overview(panel: pd.DataFrame) -> pd.DataFrame:
    year_min = int(panel["year"].min())
    year_max = int(panel["year"].max())
    n_counties = panel["region_name"].nunique()
    n_possible_years = year_max - year_min + 1

    overview = pd.DataFrame(
        {
            "total county-year observations": [len(panel)],
            "number of counties": [n_counties],
            "year range": [f"{year_min}-{year_max}"],
            "panel balance (%)": [round(100 * len(panel) / (n_counties * n_possible_years), 1)],
        }
    )

    overview.to_csv(TABLES_DIR / "statistics_overview.csv", index=False)
    return overview


def summarize(panel: pd.DataFrame, tables_dir: Path = TABLES_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    stats_cols = ["yield_lb_acre", "prcp_total", "tavg_mean", "extreme_heat_days", "heavy_rain_days"]
    summary = panel[stats_cols].describe()
    summary.to_csv(tables_dir / "summary_statistics.csv")

    corr = panel[stats_cols].corr()
    corr.to_csv(tables_dir / "correlation_matrix.csv")

    print(summary)
    print(corr)
    return summary, corr


def plot_histograms(panel: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    histograms = [
        ("yield_lb_acre", "Peanut yield (lb/acre)", "hist_yield.png"),
        ("prcp_total", "Growing-season total precipitation (mm)", "hist_prcp_total.png"),
        ("tavg_mean", "Growing-season mean temperature (C)", "hist_tavg_mean.png"),
    ]
    for col, label, fname in histograms:
        plt.figure(figsize=(8, 5))
        plt.hist(panel[col].dropna(), bins=20, edgecolor="black")
        plt.xlabel(label)
        plt.ylabel("Count of county-years")
        plt.title(f"Distribution of {label}")
        plt.tight_layout()
        plt.savefig(figures_dir / fname)
        plt.close()


def plot_scatter_with_fit(
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    fname: str,
    figures_dir: Path = FIGURES_DIR,
    degree: int = 1,
) -> None:
    data = panel[[x_col, y_col]].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(data[x_col], data[y_col], alpha=0.6)

    coeffs = np.polyfit(data[x_col], data[y_col], degree)
    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 200)
    plt.plot(x_line, np.polyval(coeffs, x_line), color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.tight_layout()
    plt.savefig(figures_dir / fname)
    plt.close()


def plot_boxplot_by_county(panel: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    counties = sorted(panel["region_name"].unique())
    data = [panel.loc[panel["region_name"] == county, "yield_lb_acre"].dropna() for county in counties]

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, tick_labels=counties)
    plt.xticks(rotation=90)
    plt.ylabel("Yield (lb/acre)")
    plt.title("Peanut yield by county")
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplot_yield_by_county.png")
    plt.close()


def plot_monthly_timing(
    panel: pd.DataFrame, monthly: pd.DataFrame, figures_dir: Path = FIGURES_DIR
) -> pd.DataFrame:
    """Correlation of yield with each month's weather, to see which months matter most."""
    merged = monthly.merge(
        panel[["region_name", "year", "yield_lb_acre"]], on=["region_name", "year"], how="inner"
    )

    corr_by_month = (
        merged.groupby("month")
        .apply(
            lambda g: pd.Series(
                {
                    "temp_corr": g["tavg_mean"].corr(g["yield_lb_acre"]),
                    "prcp_corr": g["prcp_total"].corr(g["yield_lb_acre"]),
                }
            )
        )
        .reset_index()
    )

    x = corr_by_month["month"]
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, corr_by_month["temp_corr"], width, label="Temp corr. with yield")
    plt.bar(x + width / 2, corr_by_month["prcp_corr"], width, label="Precip corr. with yield")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Month")
    plt.ylabel("Correlation with yield")
    plt.title("Monthly weather correlation with peanut yield")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_correlation_with_yield.png")
    plt.close()

    return corr_by_month


if __name__ == "__main__":
    complete_df = load_combined_data()

    panel = build_growing_season_panel(complete_df)
    panel.to_csv(PANEL_OUTPUT_PATH, index=False)

    monthly_weather = build_monthly_weather(complete_df)

    summarize(panel)
    plot_histograms(panel)
    plot_scatter_with_fit(
        panel, "prcp_total", "yield_lb_acre",
        "Total precipitation (mm)", "Yield (lb/acre)",
        "scatter_yield_vs_prcp.png", degree=2,
    )
    plot_scatter_with_fit(
        panel, "tavg_mean", "yield_lb_acre",
        "Mean temperature (C)", "Yield (lb/acre)",
        "scatter_yield_vs_tavg.png", degree=1,
    )
    plot_scatter_with_fit(
        panel, "extreme_heat_days", "yield_lb_acre",
        "Extreme heat days (>32C)", "Yield (lb/acre)",
        "scatter_yield_vs_extreme_heat.png", degree=1,
    )
    plot_boxplot_by_county(panel)
    plot_monthly_timing(panel, monthly_weather)

    statistics_by_county(panel)
    statistics_overview(panel)
