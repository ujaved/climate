import streamlit as st
from geopy.geocoders import Nominatim
from geopy.location import Location 
import requests
from http import HTTPStatus
from pyproj import Geod
import geopandas as gpd
from requests.exceptions import Timeout
import xarray as xr
import pandas as pd
import numpy as np

NOT_FOUND = "Not found"

@st.cache_data
def get_location_from_lat_lon(lat: float, lon: float):
    geolocator = Nominatim(user_agent="climsight")
    location = geolocator.reverse((lat, lon), language="en")
    return location.raw["address"]

@st.cache_data
def get_geo_location(location_str: str) -> Location:
    geolocator = Nominatim(user_agent="climsight")
    return geolocator.geocode(location_str, language="en")

@st.cache_data
def get_elevation_from_api(lat: float, lon: float) -> str:
    url = f"https://api.opentopodata.org/v1/etopo1?locations={lat},{lon}"
    response = requests.get(url)
    if response.status_code != HTTPStatus.OK:
        return NOT_FOUND
    data = response.json()
    return data["results"][0]["elevation"]

@st.cache_data
def fetch_land_use(lat: float, lon: float) -> str:
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    is_in({lat},{lon})->.a;
    area.a["landuse"];
    out tags;
    """
    response = requests.get(overpass_url, params={"data": overpass_query})
    if response.status_code != HTTPStatus.OK:
        return NOT_FOUND
    data = response.json()
    if data["elements"]:
        return data["elements"][0]["tags"]["landuse"]
    return NOT_FOUND

@st.cache_data
def get_soil_from_api(lat: float, lon: float):
    """
    Retrieves the soil type at a given latitude and longitude using the ISRIC SoilGrids API.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.

    Returns:
    str: The name of the World Reference Base (WRB) soil class at the given location.
    """
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=5"
        response = requests.get(url, timeout=2)
        if response.status_code != HTTPStatus.OK:
            return NOT_FOUND
        data = response.json()
        return data["wrb_class_name"]
    except Timeout:
        return NOT_FOUND
    
@st.cache_data
def fetch_biodiversity(lat: float, lon: float):
    """
    Fetches biodiversity data for a given longitude and latitude using the GBIF API.

    Args:
    - lon (float): The longitude of the location to fetch biodiversity data for.
    - lat (float): The latitude of the location to fetch biodiversity data for.

    Returns:
    - data (dict): A dictionary containing the biodiversity data for the specified location.
    """
    gbif_api_url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "decimalLatitude": lat,
        "decimalLongitude": lon,
    }
    response = requests.get(gbif_api_url, params=params)
    if response.status_code != HTTPStatus.OK:
        return NOT_FOUND
    biodiv = response.json()
    if len(biodiv['results']) == 0:
        return NOT_FOUND
    biodiv_set = {record['genericName'] for record in biodiv['results'] if 'genericName' in record and record.get('taxonRank') != 'UNRANKED'}
    return ', '.join(list(biodiv_set))


@st.cache_data
def closest_shore_distance(lat: float, lon: float, coastline_shapefile: str) -> float:
    """
    Calculates the closest distance between a given point (lat, lon) and the nearest point on the coastline.

    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        coastline_shapefile (str): Path to the shapefile containing the coastline data

    Returns:
        float: The closest distance between the point and the coastline, in meters.
    """
    geod = Geod(ellps="WGS84")
    min_distance = float("inf")

    coastlines = gpd.read_file(coastline_shapefile)

    for _, row in coastlines.iterrows():
        geom = row["geometry"]
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for coastal_point in line.coords:
                    _, _, distance = geod.inv(
                        lon, lat, coastal_point[0], coastal_point[1]
                    )
                    min_distance = min(min_distance, distance)
        else:  # Assuming LineString
            for coastal_point in geom.coords:
                _, _, distance = geod.inv(lon, lat, coastal_point[0], coastal_point[1])
                min_distance = min(min_distance, distance)

    return min_distance


def convert_to_mm_per_month(monthly_precip_kg_m2_s1):
    days_in_months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return monthly_precip_kg_m2_s1 * 60 * 60 * 24 * days_in_months



def extract_climate_data(lat: float, lon: float, hist: xr.Dataset, future: xr.Dataset) -> tuple[pd.DataFrame, dict]:
    """
    Extracts climate data for a given latitude and longitude from historical and future datasets.

    Args:
    - lat (float): Latitude of the location to extract data for.
    - lon (float): Longitude of the location to extract data for.
    - hist (xarray.Dataset): Historical climate dataset.
    - future (xarray.Dataset): Future climate dataset.

    Returns:
    - df (pandas.DataFrame): DataFrame containing present day and future temperature, precipitation, and wind speed data for each month of the year.
    - data_dict (dict): Dictionary containing string representations of the extracted climate data.
    """
    hist_temp = hist.sel(lat=lat, lon=lon, method="nearest")["tas"].values - 273.15
    hist_temp_str = np.array2string(hist_temp.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_pr = hist.sel(lat=lat, lon=lon, method="nearest")["pr"].values
    hist_pr = convert_to_mm_per_month(hist_pr)

    hist_pr_str = np.array2string(hist_pr.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_uas = hist.sel(lat=lat, lon=lon, method="nearest")["uas"].values
    hist_uas_str = np.array2string(hist_uas.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_vas = hist.sel(lat=lat, lon=lon, method="nearest")["vas"].values
    hist_vas_str = np.array2string(hist_vas.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    future_temp = future.sel(lat=lat, lon=lon, method="nearest")["tas"].values - 273.15
    future_temp_str = np.array2string(
        future_temp.ravel(), precision=3, max_line_width=100
    )[1:-1]

    future_pr = future.sel(lat=lat, lon=lon, method="nearest")["pr"].values
    future_pr = convert_to_mm_per_month(future_pr)
    future_pr_str = np.array2string(future_pr.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    future_uas = future.sel(lat=lat, lon=lon, method="nearest")["uas"].values
    future_uas_str = np.array2string(
        future_uas.ravel(), precision=3, max_line_width=100
    )[1:-1]

    future_vas = future.sel(lat=lat, lon=lon, method="nearest")["vas"].values
    future_vas_str = np.array2string(
        future_vas.ravel(), precision=3, max_line_width=100
    )[1:-1]
    df = pd.DataFrame(
        {
            "Present day Temperature": hist_temp[0, 0, :],
            "Future Temperature": future_temp[0, 0, :],
            "Present day Precipitation": hist_pr[0, 0, :],
            "Future Precipitation": future_pr[0, 0, :],
            "Present day Wind speed": np.hypot(hist_uas[0, 0, :], hist_vas[0, 0, :]),
            "Future Wind speed": np.hypot(future_uas[0, 0, :], future_vas[0, 0, :]),
            "Month": range(1, 13),
        }
    )
    data_dict = {
        "hist_temp": hist_temp_str,
        "hist_pr": hist_pr_str,
        "hist_uas": hist_uas_str,
        "hist_vas": hist_vas_str,
        "future_temp": future_temp_str,
        "future_pr": future_pr_str,
        "future_uas": future_uas_str,
        "future_vas": future_vas_str,
    }
    return df, data_dict