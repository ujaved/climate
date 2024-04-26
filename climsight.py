import streamlit as st
import xarray as xr
import pandas as pd
from streamlit_folium import st_folium
import folium
from chatbot import OpenAIChatbot, LLamaChatbot
from session import Session
from utils import get_geo_location


data_path = "./data/"
distance_from_event = 5.0 # frame within which natural hazard events shall be considered [in km]

year_step = 10 # indicates over how many years the population data is being averaged
start_year = 1980 # time period that one is interested in (min: 1950, max: 2100)
end_year = None # same for end year (min: 1950, max: 2100)


@st.cache_data
def load_data():
    hist = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_historical*.nc", compat="override")
    future = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_ssp585*.nc", compat="override")
    return hist, future
        
    
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
    
    lat_default = 51.4779
    lon_default =  -0.0015
    
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

WELCOME_MSG = 'Welcome to Local Climate Risk Assessment!!'

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
    
    
if __name__ == "__main__":
    main()