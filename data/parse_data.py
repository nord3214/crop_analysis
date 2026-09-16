from datetime import datetime
import os
import pandas as pd


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


cols_name = [
    "region_type",
    "region_code",
    "region_name",
    "year",
    "month",
    "variable" # changes between PRCP, TAVG, TMAX, TMIN
] + [f"day_{i}" for i in range(1, 32)]


florida_cty_col_pattern = "FL: "



file_path = "noaa_weather/"
file_path_parsed = "noaa_weather_parsed/"
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


