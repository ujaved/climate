import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
from chatbot import OpenAIChatbot, LLamaChatbot, ClimateGPTChatbot
from session import Session
from utils import get_geo_location


data_path = "./data/"
# load natural hazard data from Socioeconomic Data and Applications Center (sedac), based on EM-DAT 
haz_path = './data/natural_hazards/pend-gdis-1960-2018-disasterlocations.csv'
# load population data from UN World Population Prospects 2022
pop_path = './data/population/WPP2022_Demographic_Indicators_Medium.csv'

distance_from_event = 5.0 # frame within which natural hazard events shall be considered [in km]

year_step = 10 # indicates over how many years the population data is being averaged
start_year = 1980 # time period that one is interested in (min: 1950, max: 2100)
end_year = None # same for end year (min: 1950, max: 2100)


@st.cache_data
def load_data():
    hist = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_historical*.nc", compat="override")
    future = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_ssp585*.nc", compat="override")
    return hist, future


@st.cache_data
def load_nat_haz_data(haz_path):
    """
    Load natural hazard data from a CSV file and filter relevant columns.

    Args:
    - haz_path (str): File path to the CSV file containing natural hazard data.

    Returns:
    - pandas.DataFrame: Dataset with selected columns ('country', 'year', 'geolocation', 'disastertype', 'latitude', 'longitude').
    """

    haz_dat = pd.read_csv(haz_path)

    # reduce data set to only contain relevant columns
    columns_to_keep = ['country', 'year', 'geolocation', 'disastertype', 'latitude', 'longitude']
    haz_dat = haz_dat.loc[:, columns_to_keep]

    return(haz_dat)

@st.cache_data
def filter_events_within_square(lat, lon, haz_path, distance_from_event):
    """
    Filter events within a square of given distance from the center point.

    Args:
    - lat (float): Latitude of the center point (rounded to 3 decimal places)
    - lon (float): Longitude of the center point (rounded to 3 decimal places)
    - haz_dat (pandas.DataFrame): Original dataset.
    - distance_from_event (float): Distance in kilometers to form a square.

    Returns:
    - pandas.DataFrame: Reduced dataset containing only events within the square.
    """

    haz_dat = load_nat_haz_data(haz_path)

    # Calculate the boundaries of the square
    lat_min, lat_max = lat - (distance_from_event / 111), lat + (distance_from_event / 111)
    lon_min, lon_max = lon - (distance_from_event / (111 * np.cos(np.radians(lat)))), lon + (distance_from_event / (111 * np.cos(np.radians(lat))))

    # Filter events within the square
    filtered_haz_dat = haz_dat[
        (haz_dat['latitude'] >= lat_min) & (haz_dat['latitude'] <= lat_max) &
        (haz_dat['longitude'] >= lon_min) & (haz_dat['longitude'] <= lon_max)
    ]

    prompt_haz_dat = filtered_haz_dat.drop(columns=['country', 'geolocation', 'latitude', 'longitude'])

    return filtered_haz_dat, prompt_haz_dat

@st.cache_data
def plot_disaster_counts(filtered_events):
    """
    Plot the number of different disaster types over a time period for the selected location (within 5km radius).

    Args:
    - filtered_events: Only those natural hazard events that were within a 5 km (or whatever other value is set for distance_from_event) radius of the clicked location.
    Returns:
    - figure: bar plot with results
    """
    if not filtered_events.empty:
        # Group by 'year' and 'disastertype' and count occurrences
        disaster_counts = filtered_events.groupby(['year', 'disastertype']).size().unstack(fill_value=0)
        place = filtered_events['geolocation'].unique()

        # create figure and axes
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Plotting the bar chart
        disaster_counts.plot(kind='bar', stacked=False, ax=ax, figsize=(10,6), colormap='viridis')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Count of different disaster types in ' + place[0] + ' over time')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Disaster Type')

        return fig
    else:
        return None

def get_population(pop_path, country):
    """
    Extracts population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - red_pop_data (pandas.DataFrame): reduced DataFrame containing present day and future values for only the following variables:
        - TPopulation1July (as of 1 July, thousands)
        - PopDensity (as of 1 July, persons per square km)
        - PopGrowthRate (percentage)
        - LEx (Life Expactancy at Birth, both sexes, in years)
        - NetMigrations (Number of Migrants, thousands)    
    """
    pop_dat = pd.read_csv(pop_path)

    unique_locations = pop_dat['Location'].unique()
    my_location = country

    # check if data is available for the country that we are currently investigating
    if my_location in unique_locations:
        country_data = pop_dat[pop_dat['Location'] == country]
        red_pop_data = country_data[['Time', 'TPopulation1July', 'PopDensity', 'PopGrowthRate', 'LEx', 'NetMigrations']]
        return red_pop_data
    else:
        print("No population data available for " + country + ".")
        return None
    
def plot_population(pop_path, country):
    """
    Plots population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - plot: visual representation of the data distribution    
    """
    reduced_pop_data = get_population(pop_path, country)
    
    today = datetime.date.today()
    current_year = today.year

    if reduced_pop_data is not None and not reduced_pop_data.empty:
        fig, ax1 = plt.subplots(figsize=(10,6))
        plt.grid()

        # Total population data
        ax1.plot(reduced_pop_data['Time'], reduced_pop_data['TPopulation1July'], label='Total Population', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('People in thousands', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # life expectancy
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(('axes', 1.1))
        ax2.bar(reduced_pop_data['Time'], reduced_pop_data['LEx'], label='Life Expectancy', color='purple', alpha=0.1)
        ax2.set_ylabel('Life Expectancy in years', color='purple', )
        ax2.tick_params(axis='y', labelcolor='purple')

        # population growth data
        ax3 = ax1.twinx()
        ax3.plot(reduced_pop_data['Time'], reduced_pop_data['PopGrowthRate'], label='Population Growth Rate', color='green')
        ax3.set_ylabel('Population growth rate in %', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Net Migrations
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(('axes', 1.2))
        ax4.plot(reduced_pop_data['Time'], reduced_pop_data['NetMigrations'], label='Net Migrations', color='black', linestyle='dotted')
        ax4.set_ylabel('Net Migrations in thousands', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.axvline(x=current_year, color='orange', linestyle='--', label=current_year)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines+lines2+lines3+lines4, labels+labels2+labels3+labels4, loc='center right')

        plt.title(('Population changes in ' + country))
        return fig 
    else:
        return None
    
def calc_mean(years, dataset):
    """
    Calculates the mean of every column of a dataframe over a given time period and returns those means.

    Parameters:
    years (int): The time period that one is interested in to be averaged.
    dataset (pandas data frame): The corresponding data set. It has to have a column called 'Time' in datetime format.

    Returns:
    pandas data frame: A data frame with the means calculated for the given time span.
    """
    years = str(years) + 'Y'
    dataset.set_index('Time', inplace=True) # Set the 'Time' column as the index
    numeric_columns = dataset.select_dtypes(include='number')
    dataset = numeric_columns.resample(years).mean() # Resample the numeric data in x-year intervals and calculate the mean
    dataset.reset_index(inplace=True) # Reset the index to have 'Time' as a regular column
    dataset['Time'] = dataset['Time'].dt.year # and normal year format
    
    return dataset

def x_year_mean_population(pop_path, country, year_step=1, start_year=None, end_year=None):
    """
    Returns a reduced data set with the means calculated for every column over a given time span

    Parameters:
    pop_path (string): Path where the data is stored.
    country (string): The country which has been clicked on the map by the user.
    year_step (int): How many years shall be aggregated.
    start_year (int): The year from which onward population data is considered.
    end_year (int): The year until which population data is considered.

    Returns:
    pandas data frame: A data frame containing the mean population data values for a given time period.
    """
    # Check if start_year and end_year are within the allowed range
    if (start_year is not None and (start_year < 1950 or start_year > 2100)) or \
       (end_year is not None and (end_year < 1950 or end_year > 2100)):
        print("Warning: Start and end years must be between 1950 and 2100.")
        return None
    
    population_xY_mean = get_population(pop_path, country)
    if population_xY_mean is None:
        print(f"No population data available for {country}.")
        return None
    column_to_remove = ['LEx', 'NetMigrations'] # change here if less / other columns are wanted
    

    if not population_xY_mean.empty:
        population_xY_mean = population_xY_mean.drop(columns=column_to_remove)

        population_xY_mean['Time'] = pd.to_datetime(population_xY_mean['Time'], format='%Y')

        # Filter data based on start_year and end_year
        if start_year is not None:
            start_year = max(min(start_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year >= start_year]
        if end_year is not None:
            end_year = max(min(end_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year <= end_year]

        # Subdivide data into two data frames. One that contains the last complete x-year period (z-times the year_step) and the rest (modulo). For each data set the mean is calculated.
        modulo_years = len(population_xY_mean['Time']) % year_step 
        lastFullPeriodYear = population_xY_mean['Time'].dt.year.iloc[-1] - modulo_years  
        FullPeriod = population_xY_mean[population_xY_mean['Time'].dt.year <= lastFullPeriodYear]
        RestPeriod = population_xY_mean[population_xY_mean['Time'].dt.year > lastFullPeriodYear]

        # calculate mean for each period
        FullPeriodMean = calc_mean(year_step, FullPeriod)
        RestPeriodMean = calc_mean(modulo_years - 1, RestPeriod)
        RestPeriodMean = RestPeriodMean.iloc[1:] # drop first row as it will be same as last one of FullPeriodMean

        combinedMean  = pd.concat([FullPeriodMean, RestPeriodMean], ignore_index=True) # combine back into one data set

        new_column_names = {
            'TPopulation1July': 'TotalPopulationAsOf1July',
            'PopDensity': 'PopulationDensity',
            'PopGrowthRate': 'PopulationGrowthRate',  
        }
        combinedMean.rename(columns=new_column_names, inplace=True)

        return combinedMean
    
    else:
        return None
        
    
def location_cb():
    if not st.session_state.location:
        return
    loc = get_geo_location(st.session_state.location)
    st.session_state.latitude = loc.latitude
    st.session_state.longitude = loc.longitude
    st.session_state.location_display_name = loc.raw['display_name']
    
def llm_selection_cb():
    st.session_state.disable_llm_selection = True
    
    if st.session_state.llm == "llama":
        st.session_state.session.chatbot = LLamaChatbot(model_id="llama", temperature=0.01)
    #elif st.session_state.llm == "climategpt":
    #    st.session_state.session.chatbot = ClimateGPTChatbot(endpoint="https://adapting-foxhound-neatly.ngrok-free.app/completion", max_new_tokens=500, poll_interval=5)
        
    
def render_map():
    
    lat_default = 52.5240
    lon_default = 13.3700
    
    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
         st.session_state.latitude = lat_default
         st.session_state.longitude = lon_default
         st.session_state.location_display_name = ""
         
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=10)
    with st.sidebar:
        st.markdown(f"**{st.session_state.location_display_name}**")
        st.text_input("Location", key="location", on_change=location_cb)
        col1, col2 = st.columns(2)
        col1.metric("Latitude", round(st.session_state.latitude, 4))
        col2.metric("Longitude", round(st.session_state.longitude, 4))
        map_data = st_folium(m)
        clicked_coords = map_data["last_clicked"]
        if clicked_coords:
            st.session_state.latitude = clicked_coords["lat"]
            st.session_state.longitude = clicked_coords["lng"]

WELCOME_MSG = 'Welcome to Climate Risk Assessment!!'

def main():
    
    st.set_page_config(page_title="Climate Risk Assessment", page_icon=":globe_with_meridians:", layout="wide")
    
    if 'disable_llm_selection' not in st.session_state:
        st.session_state.disable_llm_selection = False
    st.sidebar.radio("llm", ["gpt-4", "llama"],
                     key="llm", on_change=llm_selection_cb, disabled=st.session_state.disable_llm_selection)
    
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MSG)
    if 'session' not in st.session_state: 
        st.session_state.session = Session(chatbot=OpenAIChatbot(model_id="gpt-4-1106-preview", temperature=0))
        
    tabs = st.tabs(["chat", "charts"])
    st.session_state.session.chat_tab = tabs[0]
    st.session_state.session.charts_tab = tabs[1]
    st.session_state.session.render()
    
    render_map()
    if 'first_user_input' not in st.session_state:
         st.session_state.first_user_input = False
         
    hist, future = load_data()
    st.session_state.hist = hist
    st.session_state.future = future

    haz_fig = None 
    population_plot = None
    #haz_fig = plot_disaster_counts(filtered_events_square)
    #population_plot = plot_population(pop_path, country)

    if haz_fig is not None or population_plot is not None:
        st.subheader("Additional information", divider='rainbow')

    # Natural Hazards
    if haz_fig is not None:
        st.markdown("**Natural hazards:**")
        st.pyplot(haz_fig)
        with st.expander("Source"):
            st.markdown('''
                *The GDIS data descriptor*  
                Rosvold, E.L., Buhaug, H. GDIS, a global dataset of geocoded disaster locations. Sci Data 8,
                61 (2021). https://doi.org/10.1038/s41597-021-00846-6  
                *The GDIS dataset*  
                Rosvold, E. and H. Buhaug. 2021. Geocoded disaster (GDIS) dataset. Palisades, NY: NASA
                Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/zz3b-8y61.
                Accessed DAY MONTH YEAR.  
                *The EM-DAT dataset*  
                Guha-Sapir, Debarati, Below, Regina, & Hoyois, Philippe (2014). EM-DAT: International
                disaster database. Centre for Research on the Epidemiology of Disasters (CRED).
            ''')

    # Population Data
    if population_plot is not None:
        st.markdown("**Population Data:**")
        st.pyplot(population_plot)
        with st.expander("Source"):
            st.markdown('''
            United Nations, Department of Economic and Social Affairs, Population Division (2022). World Population Prospects 2022, Online Edition. 
            Accessible at: https://population.un.org/wpp/Download/Standard/CSV/.
            ''')
    
    
if __name__ == "__main__":
    main()