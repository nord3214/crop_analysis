from datetime import datetime
import os
import pandas as pd

item_names = {
    "PEANUTS - ACRES HARVESTED": "acres_harvested",
    "PEANUTS - ACRES PLANTED": "acres_planted",
    "PEANUTS - PRODUCTION, MEASURED IN LB": "production_lb",
    "PEANUTS - YIELD, MEASURED IN LB / ACRE": "yield_lb_acre",
}


def create_date_col(row):
        date = f"{row['year']}-{row['month']}-{row['day'].split('_')[1]}"
        return datetime.strptime(date, "%Y-%m-%d")


def melt_df(df) -> pd.DataFrame:
    """
    Melt the dataframe to have a long format with columns: region_type, region_code, region_name, year, month, variable, day, value
    """
    # remove all columns that start with "day_" and have all values equal to -999.99 (months that have less than 31 days)
    for col in df.columns:
        if col.startswith("day_"):
            if (df[col] == -999.99).all():
                df.drop(columns=[col], inplace=True)


    df_melted = df.melt(
        id_vars=["region_type",
            "region_code",
            "region_name",
            "year",
            "month",
            "variable"],
        var_name="day",
        value_name="value"
    )



    df_melted['date'] = df_melted.apply(create_date_col, axis=1)

    df_melted.drop(columns=["year", "month", "day", "region_type", "region_code", "variable"], inplace=True)

    df_melted["region_name"] = df_melted["region_name"].str.replace("FL: ", "", regex=False)

    return df_melted




def parse_noaa_data():

    florida_cty_col_pattern = "FL: "

    cols_name = [
    "region_type",
    "region_code",
    "region_name",
    "year",
    "month",
    "variable" # changes between PRCP, TAVG, TMAX, TMIN
] + [f"day_{i}" for i in range(1, 32)]



    file_path = "data/raw/noaa"
    file_path_parsed = "data/final/"
    os.makedirs(file_path_parsed, exist_ok=True)

    filenames = os.listdir(file_path)

    for var in ["prcp", "tavg", "tmax", "tmin"]:
        df_long = pd.DataFrame()

        for filename in filenames:
            if filename.endswith(".csv"):

                if f"{var}-" in filename:
                    # print(f"Processing: {filename}")
                    df = pd.read_csv(os.path.join(file_path, filename), header=None, names= cols_name)
                    # print(df.columns)   
                    boolean_mask2 = df["region_name"].str.startswith(florida_cty_col_pattern)
                    df = df[boolean_mask2]
                    # print(df.head())
                    df = melt_df(df)
                    df_long = pd.concat([df_long, df], axis=0)
                    print(f"Processed: {filename}")
        df_long.index = pd.MultiIndex.from_frame(df_long[["region_name", "date"]])
        df_long = df_long.drop(columns=["region_name", "date"])
        df_long.to_csv(os.path.join(file_path_parsed, "florida_counties_"  + var + ".csv"), index=True)
        
        print(f"Saved: florida_counties_{var}")



def parse_peanut_data():
    nass_raw = pd.read_csv(
        "data/raw/nass/peanut_data_florida_per_cty.csv",
        dtype=str,
    )





    df = nass_raw[
        nass_raw["Data Item"].isin(item_names)
        & (nass_raw["Period"].str.upper() == "YEAR")
        & (nass_raw["Domain"].str.upper() == "TOTAL")
        & ~nass_raw["County"].str.upper().str.contains("OTHER", na=False)
    ].copy()

    df["Value"] = pd.to_numeric(
        df["Value"].str.replace(",", "", regex=False),
        errors="coerce",
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Data Item"] = df["Data Item"].map(item_names)

    wide = (
        df.pivot_table(
            index=["County", "County ANSI", "Year"],
            columns="Data Item",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .sort_values(["County", "Year"])
    )

    wide.to_csv("data/final/CropYieldPeanutsFlorida_parsed.csv", index=False)

    return wide

def combine_datasets():
    noaa_dir = "data/final/"
    nass_file = "data/final/CropYieldPeanutsFlorida_parsed.csv"

    noaa_files = [f for f in os.listdir(noaa_dir) if f.startswith("florida_counties_") and f.endswith(".csv")]

    combined_df = pd.DataFrame()

    for noaa_file in noaa_files:
        var = noaa_file.split("_")[-1].replace(".csv", "")
        df_noaa = pd.read_csv(os.path.join(noaa_dir, noaa_file), index_col=[0, 1])
        df_noaa = df_noaa.rename(columns={"value": var})
        if combined_df.empty:
            combined_df = df_noaa
        else:
            combined_df = combined_df.join(df_noaa, how="outer")

    
    df_nass = pd.read_csv(nass_file)
    df_nass = df_nass.rename(columns={"County": "region_name", "Year": "year"})
    combined_df.reset_index(inplace=True)
    combined_df["region_name"] = combined_df["region_name"].str.replace("FL: ", "", regex=False).str.replace(" County", "", regex=False).str.upper()
    combined_df["date"] = pd.to_datetime(combined_df["date"])
    combined_df["year"] = combined_df["date"].dt.year

    final_combined_df = pd.merge(combined_df, df_nass, left_on=["region_name", "year"], right_on=["region_name", "year"], how="left")
    final_combined_df = final_combined_df.drop(columns=["year"])
    final_combined_df.to_csv("data/final/combined_dataset.csv", index=False)
    print("Combined dataset saved to data/final/combined_dataset.csv")


if __name__ == "__main__":

    # parse_noaa_data()
    # parse_peanut_data()

    combine_datasets()