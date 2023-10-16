import pathlib
import pandas as pd
import numpy as np
import pytz
import holidays
from astral import LocationInfo
from astral.sun import sun, elevation, azimuth
from tqdm import tqdm
from settings import data_folder
import os

MY_TIME_ZONE = pytz.timezone("Europe/Zurich")
CANTONS = [
    "AG",
    "AI",
    "AR",
    "BE",
    "BL",
    "BS",
    "FR",
    "GE",
    "GL",
    "GR",
    "JU",
    "LU",
    "NE",
    "NW",
    "OW",
    "SG",
    "SH",
    "SO",
    "SZ",
    "TG",
    "TI",
    "UR",
    "VD",
    "VS",
    "ZG",
    "ZH",
]

CONSO_COL = [
    "Summe endverbrauchte Energie Regelblock Schweiz\nTotal energy consumed by end users in the Swiss controlblock"
]
LAT_BERN, LONG_BERN = 46.947456, 7.451123
VERBOSE = False
SUN_OFFSET = 30


def load_processed_meteo(folder, start="2020", end="2022-08-31", timezone=MY_TIME_ZONE):
    """Return all weather data in a single dataframe

    Args:
        folder (str): folder name
        start (str, optional): starting date. Defaults to "2020".
        end (str, optional): ending date. Defaults to "2022-08-31".
        timezone (tz, optional): timezone. Defaults to MY_TIME_ZONE.

    Returns:
        Dataframe, dict: weather data, column name
    """
    meteo = []
    cols_name = {}
    split_index = 0
    for p in pathlib.Path(folder).glob("*.parquet"):
        name_feat = str(p).split("/")[-1].split(".")[0]
        data = pd.read_parquet(p)
        data = data[start:end]
        data = data.reindex(columns=sorted(data.columns))
        data.columns = data.columns.map(lambda x: f"{name_feat}_" + x)
        cl = name_feat
        # Remove leading numbers
        if name_feat.startswith("2m_"):
            cl = name_feat[3:]
        elif name_feat.startswith("10m_"):
            cl = name_feat[4:]
        cols_name[cl] = np.arange(split_index, split_index + data.shape[1])
        split_index += data.shape[1]
        meteo.append(data)
    return pd.concat(meteo, axis=1), cols_name



def solar_position(grid):
    # Return the sun position ("sunrise", "sunset", "solar_elevation", "azimuth")
    city = LocationInfo("Bern", "Switzerland", "Europe/Zurich", LAT_BERN, LONG_BERN)
    sun_up = []
    for date in tqdm(grid.index, desc="Get sun coordinates", disable=not VERBOSE):
        s = sun(city.observer, date=date, tzinfo=city.timezone)
        elev = elevation(city.observer, date, True)
        azim = azimuth(city.observer, date)
        sun_up.append([date, s["sunrise"], s["sunset"], elev, azim])

    sun_up = pd.DataFrame(
        sun_up, columns=["date", "sunrise", "sunset", "solar_elevation", "azimuth"]
    )
    time_day = sun_up["date"]
    lower = sun_up["sunrise"] + pd.DateOffset(minutes=SUN_OFFSET)
    upper = sun_up["sunset"] - pd.DateOffset(minutes=SUN_OFFSET)
    sun_up["sun_up"] = (time_day >= lower) & (sun_up["date"] <= upper)
    return sun_up.set_index("date")


def build_temporal(grid):
    """Build time variables based on grid

    Args:
        grid (DataFrame): time indexed dataframe

    Returns:
        DataFRame: time variables
    """
    hours = grid.index.hour
    weekday = grid.index.weekday
    weekend = (weekday > 4).astype(int)
    hourweek = hours + weekday * 24
    t = grid.index.astype(int)
    hours_weekend = hours * weekend

    # Normalize
    dayofyear = grid.index.dayofyear
    day = grid.index.day
    month = grid.index.month
    # dayofyear /= 366
    # hourweek /= 168
    # weekday /= 7
    # hours /= 24
    min_t, max_t = 1514761200000000000, 1672524000000000000
    t = (t - min_t) / (max_t - min_t)
    week = grid.index.isocalendar().week.reset_index(drop=True)
    features = pd.concat(
        [
            pd.Series(x)
            for x in [
                hours,
                hourweek,
                hours_weekend,
                weekday,
                weekend,
                dayofyear,
                t,
                week,
                day,
                month,
            ]
        ],
        axis=1,
    ).reset_index(drop=True)
    columns = [
        "hours",
        "hourweek",
        "hours_weekend",
        "weekday",
        "weekend",
        "dayofyear",
        "t",
        "week",
        "day",
        "month",
    ]
    features.columns = columns
    features.index = grid.index
    return features

def load_calendar(start, end, use_ref, hyperparameter):
    """Build calendar variables

    Args:
        start (str): starting date
        end (str): ending date
        hyperparameter (dict): hyperparameter

    Returns:
        Tuple(DataFrame): returns 3 dataframes (hourly resolution):
                            1) calendar variables (day of week, hour, hour of the week, etc...)
                            2) holidays per canton
                            3) name of holidays per canton
    """
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    start_, end_ = start.tz_localize(MY_TIME_ZONE), pd.to_datetime(
        str(end.year + 1)
    ).tz_localize(MY_TIME_ZONE)
    # Index
    grid = pd.DataFrame({"time": [start_, end_]}).set_index("time").asfreq("1h")
    grid["date"] = grid.index.date
    grid = grid.iloc[:-1]
    if use_ref:
        # Use 2022 as reference
        start_ = pd.to_datetime("2022")
        end_ = pd.to_datetime("2022")
        start_, end_ = start_.tz_localize(MY_TIME_ZONE), pd.to_datetime(
            str(end_.year + 1)
        ).tz_localize(MY_TIME_ZONE)
        # Create index for 2022
        grid_ = pd.DataFrame({"time": [start_, end_]}).set_index("time").asfreq("1h")
        grid_ = grid_.iloc[:-1]
        feat_temp = build_temporal(grid_)
        # Convert to UTC for merging years
        # (but need +1 to correctly match days)
        feat_temp.index = feat_temp.index.tz_convert("utc") + pd.Timedelta(hours=1)
        grid_utc = grid.copy()
        grid_utc.index = grid_utc.index.tz_convert("utc") + pd.Timedelta(hours=1)

        # Replacing calendar by the 2022 calendar
        grid_utc["month_"] = grid_utc.index.month
        grid_utc["day_"] = grid_utc.index.day
        grid_utc["hour_"] = grid_utc.index.hour

        feat_temp["month_"] = feat_temp.index.month
        feat_temp["day_"] = feat_temp.index.day
        feat_temp["hour_"] = feat_temp.index.hour

        # matching years base on "month","day","hour"
        # drop 29 th of Feb if necessary
        feat_temp = (
            grid_utc.reset_index()
            .merge(feat_temp, on=["month_", "day_", "hour_"])
            .set_index("time")
            .drop(columns=["month_", "day_", "hour_", "date"])
        )  # .duplicated()
        feat_temp.index = feat_temp.index - pd.Timedelta(hours=1)
        # Reset the timezone
        feat_temp.index = feat_temp.index.tz_convert(MY_TIME_ZONE)
        # Add the column t that was removed
        feat_temp["t"] = build_temporal(grid)["t"]
        # Sort index
        feat_temp = feat_temp.sort_index()
    else:
        feat_temp = build_temporal(grid)
    feat_solar = solar_position(grid)
    # Don't use these columns
    calendar_cycles = pd.merge(
        feat_temp, feat_solar, left_index=True, right_index=True
    ).drop(columns=["sunrise", "sunset"])

    # official_holidays = utils.fill_pont(
    #     official_holidays, calendar_cycles.weekend, hyperparameter
    # )
    return calendar_cycles


def get_list_stations():
    """Find the closest meteo stations from the DNO location:
    1) less than 5 km from boundary
    2) or 3 smallest distance

    Args:
        dno (str): DNO name

    Returns:
        list: list of meteo stations
    """
    distance = pd.read_csv(
        os.path.join(data_folder, "weights", "distance.csv"), index_col=0
    )
    return distance.columns.tolist()



def get_covariates(start, end, use_ref=False):
    """Return input variables for model

    Args:
        start (str): starting date
        end (str): ending date

    Returns:
        DataFrame: Covariates
    """
    if VERBOSE:
        print("Getting calendar...")
    try:
        calendar_cycles = load_calendar(
            start, end, use_ref, hyperparameter=None
        )
    except Exception as e:
        raise ValueError(f"Cannot load calendar on period [{start},{end}]") from e
    # get list of weather stations
    closest_stn = get_list_stations()

    # Compute a single column specifying if holiday or not and the name of the holiday
    
    calendar_data = calendar_cycles
    if VERBOSE:
        print("Getting meteo...")
    try:
        meteo, _ = load_processed_meteo(
            os.path.join(data_folder, "meteoswiss"),
            start=start,
            end=end,
        )
    except Exception as e:
        raise ValueError(f"Cannot load meteo data on period [{start},{end}]") from e

    meteo = meteo[[x for x in meteo.columns if x.split("_")[-1] in closest_stn]]

    # weights for the average, based on how many buildings are in a certain area
    number_building = pd.read_csv(
        os.path.join(data_folder, "weights", "building_per_area.csv")
    ).set_index("closest_stn")
    number_building = number_building.loc[closest_stn]
    number_building /= number_building.sum()
    nb_build = number_building.T
    nb_build = nb_build.reindex(columns=sorted(nb_build.columns))

    # Average meteo data based on number of building in each region
    meteo_avg = []
    cols_name_meteo = np.unique(
        ["_".join(col.split("_")[:-1]) for col in meteo.columns]
    )
    for x in cols_name_meteo:
        cols = [c for c in meteo.columns if c.startswith(x)]
        meteo_avg.append(
            (meteo[cols] * nb_build.to_numpy().reshape(1, -1)).sum(axis=1).to_frame(x)
        )

    meteo = pd.concat(meteo_avg, axis=1)
    # Gathering all data in a single dataframe
    X = pd.concat((calendar_data, meteo), axis=1)
    return X

def set_time(col, timezone):
    """Set time to correct time zone

    Args:
        col (pd.Series): column
        timezone: timezone

    Returns:
        pd.Series:
    """
    return pd.to_datetime(col, utc=True).dt.tz_convert(timezone)


def load_national_consumption(start="2020", end="2022-08-31", col=None, timezone=MY_TIME_ZONE):
    filename=os.path.join(data_folder, "swissgrid", "consumption_all_h.csv")
    if col is not None:
        consumption_all_h = pd.read_csv(filename, usecols=[col, "date"])
    else:
        consumption_all_h = pd.read_csv(filename)
    consumption_all_h["date"] = set_time(consumption_all_h["date"], timezone)
    consumption_all_h = consumption_all_h.set_index("date")
    if start is not None and end is not None:
        consumption_all_h = consumption_all_h[start:end]
    if end is None:
        consumption_all_h = consumption_all_h[start:]
    return consumption_all_h