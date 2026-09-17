from pathlib import Path
import time
import requests

base_url = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/averages"

years = range(1991, 2025)
months = range(4, 11)  # April to October for growing season
variables = ["prcp", "tavg", "tmax", "tmin"]

output_dir = Path("data/raw/noaa")
output_dir.mkdir(exist_ok=True)

for year in years:
    for month in months:
        ym = f"{year}{month:02d}"

        for var in variables:
            filename = f"{var}-{ym}-cty-scaled.csv"
            url = f"{base_url}/{year}/{filename}"

            response = requests.get(url)
            response.raise_for_status()

            time.sleep(3)  

            filepath = output_dir / filename
            filepath.write_bytes(response.content)

            print(f"Downloaded: {filename}")