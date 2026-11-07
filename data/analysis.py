import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from data.parse_data import item_names

COMBINED_DATA_PATH = "data/final/combined_dataset.csv"
PANEL_OUTPUT_PATH = "data/final/county_year_panel.csv"
FIGURES_DIR = Path("data/output/figures")
TABLES_DIR = Path("data/output/tables")

MONTH_TAVG_DICT = {i: f"tavg_{calendar.month_abbr[i].lower()}" for i in range(4, 11)}
MONTH_PRCPT_DICT = {i: f"prcp_{calendar.month_abbr[i].lower()}" for i in range(4, 11)}



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


def plot_heatmap_yield_by_county_year(panel: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    """County x year grid of yield, so cross-county and cross-year patterns show up in one figure."""
    pivot = panel.pivot_table(index="region_name", columns="year", values="yield_lb_acre", aggfunc="first")
    pivot = pivot.sort_index()

    data = np.ma.masked_invalid(pivot.values)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(data, cmap=cmap, edgecolors="white", linewidth=0.3)

    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()

    ax.set_xlabel("Year")
    ax.set_ylabel("County")
    ax.set_title("Peanut yield by county and year (gray = no data)")

    fig.colorbar(mesh, ax=ax, label="Yield (lb/acre)")
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_yield_county_year.png")
    plt.close(fig)


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

def pivot_monthly_weather(monthly: pd.DataFrame, crop_panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly weather data to have one row per county-year, with columns for each month's weather data"""
    pivoted_tavg = monthly.pivot(
        index=["region_name", "year"],
        columns="month",
        values="tavg_mean"
    )

    pivoted_tavg.rename(columns=MONTH_TAVG_DICT, inplace=True)

    pivoted_prcp = monthly.pivot(
        index=["region_name", "year"],
        columns="month",
        values="prcp_total"
    )
    pivoted_prcp.rename(columns=MONTH_PRCPT_DICT, inplace=True)

    pivoted_long = pd.concat([pivoted_tavg, pivoted_prcp], axis=1).reset_index()
    pivoted_with_crop = pd.DataFrame.merge(pivoted_long, crop_panel, on=["region_name", "year"], how="inner")
    
    return pivoted_with_crop


if __name__ == "__main__":
    complete_df = load_combined_data()

    panel = build_growing_season_panel(complete_df)
    panel.to_csv(PANEL_OUTPUT_PATH, index=False)

    monthly_weather = build_monthly_weather(complete_df)

    pivoted_monthly_weather = pivot_monthly_weather(monthly_weather, panel)
    pivoted_monthly_weather.to_csv(TABLES_DIR / "pivoted_monthly_weather.csv", index=False)

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
    plot_heatmap_yield_by_county_year(panel)
    plot_monthly_timing(panel, monthly_weather)

    statistics_by_county(panel)
    statistics_overview(panel)
