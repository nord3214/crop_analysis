import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.dates as mdates

### display peanut yield

df = pd.read_csv('data/test/CropYieldPeanutsFlorida.csv', delimiter=';')
df['Value'] = pd.to_numeric(df['Value'].str.replace(',', ''), errors='coerce')

print(df.describe())
print(df.head())

# df_county_mask = df['County'] == 'JACKSON'
df_year_mask = df['Year'] > 2000 
df_2020 = df[df_year_mask]
first_five_counties = df_2020['County'].dropna().unique()[:5]
df_first_five = df_2020[df_2020['County'].isin(first_five_counties)]

print(df.columns)

for county, county_data in df_first_five.groupby('County'):
    plt.plot(
        county_data['Year'],
        county_data['Value'],
        marker='o',
        label=county
        )

plt.xlabel('Year')
plt.ylabel('PEANUTS - YIELD, MEASURED IN LB / ACRE')
plt.title('Peanut Yield by first five County')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




# display precipitation and temperature

relative_path = 'noaa_weather_parsed'
csv_types = ['prcp', 'tavg', 'tmin', 'tmax']
current_path = os.getcwd()
df_types = {}
for _idx, csv_type in enumerate(csv_types):
    df = pd.read_csv(os.path.join(current_path, relative_path, f'florida_counties_{csv_type}.csv'))
    df["date"] = pd.to_datetime(df["date"])
    df_types[csv_type] = df[df["region_name"] == "Polk County"].sort_values("date")

prcp_df = df_types["prcp"]
tavg_df = df_types["tavg"]

fig, temperature_axis = plt.subplots(figsize=(11, 6))
precipitation_axis = temperature_axis.twinx()

temperature_line = temperature_axis.plot(
    tavg_df["date"],
    tavg_df["value"],
    color="tab:red",
    marker="o",
    label="Average temperature",
)

prcp_monthly = (
    prcp_df
    .assign(month=prcp_df["date"].dt.to_period("M"))
    .groupby("month", as_index=False)["value"]
    .sum()
)

prcp_monthly["date"] = prcp_monthly["month"].dt.to_timestamp()

precipitation_bars = precipitation_axis.bar(
    prcp_monthly["date"],
    prcp_monthly["value"],
    width=12,
    label="Monthly precipitation",
)

temperature_axis.set_xlabel("Month")
temperature_axis.set_ylabel("Average temperature")
precipitation_axis.set_ylabel("Precipitation")
temperature_axis.set_title("Polk County Weather in 2020")
temperature_axis.xaxis.set_major_locator(mdates.MonthLocator())
temperature_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
temperature_axis.grid(axis="y", alpha=0.3)

handles = temperature_line + [precipitation_bars]
temperature_axis.legend(handles, [handle.get_label() for handle in handles], loc="upper left")
fig.tight_layout()
plt.show()