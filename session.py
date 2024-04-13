from dataclasses import dataclass, field
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from chatbot import Chatbot
import pandas as pd
from utils import get_elevation_from_api, fetch_land_use, get_soil_from_api, fetch_biodiversity, closest_shore_distance, extract_climate_data


coastline_shapefile = "./data/natural_earth/coastlines/ne_50m_coastline.shp"

content_message = "{user_input} \n \
      Location: latitude = {lat}, longitude = {lon} \
      Adress: {address} \
      Distance to the closest coastline: {distance_to_coastline} \
      Elevation above sea level: {elevation} \
      Current land use: {current_land_use} \
      Current soil type: {current_soil_type} \
      Occuring species: {occuring_species} \
      Current mean monthly temperature for each month: {hist_temp} \
      Future monthly temperatures for each month at the location: {future_temp}\
      Curent precipitation flux (mm/month): {hist_pr} \
      Future precipitation flux (mm/month): {future_pr} \
      Curent u wind component (in m/s): {hist_uas} \
      Future u wind component (in m/s): {future_uas} \
      Curent v wind component (in m/s): {hist_vas} \
      Future v wind component (in m/s): {future_vas} \
      "    
     # Natural hazards: {nat_hazards} \
     # Population data: {population} \
      
climate_gpt_msg_template = "Given the following information about address {address}: \n \
      Location: latitude = {lat}, longitude = {lon} \
      Adress: {address} \
      Distance to the closest coastline: {distance_to_coastline} \
      Elevation above sea level: {elevation} \
      Current land use: {current_land_use} \
      Current soil type: {current_soil_type} \
      Occuring species: {occuring_species} \
      Current mean monthly temperature for each month: {hist_temp} \
      Future monthly temperatures for each month at the location: {future_temp}\
      Curent precipitation flux (mm/month): {hist_pr} \
      Future precipitation flux (mm/month): {future_pr} \
      Curent u wind component (in m/s): {hist_uas} \
      Future u wind component (in m/s): {future_uas} \
      Curent v wind component (in m/s): {hist_vas} \
      Future v wind component (in m/s): {future_vas} \n \
      The climate change risk assessment for the human activity {user_input} is \
      "
    
def get_env_climate_info():
      
    hist=st.session_state.hist 
    future=st.session_state.future
    
    with st.spinner("Getting information..."):
        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures.append(executor.submit(get_elevation_from_api, st.session_state.latitude, st.session_state.longitude))
            futures.append(executor.submit(fetch_land_use, st.session_state.latitude, st.session_state.longitude))
            futures.append(executor.submit(get_soil_from_api, st.session_state.latitude, st.session_state.longitude))
            futures.append(executor.submit(fetch_biodiversity, st.session_state.latitude, st.session_state.longitude))
            futures.append(executor.submit(closest_shore_distance, st.session_state.latitude, st.session_state.longitude, coastline_shapefile))
            futures.append(executor.submit(extract_climate_data, st.session_state.latitude, st.session_state.longitude, hist, future))
        
        location_env_info = {}
        location_env_info["elevation"] = ("Elevation above sea level", f"{futures[0].result()} m")
        location_env_info["current_land_use"] = ("Current land use", f"{futures[1].result()}")
        location_env_info["current_soil_type"] = ("Current Soil type", f"{futures[2].result()}")
        location_env_info["occuring_species"] = ("Occuring Species", f"{futures[3].result()}")
        location_env_info["distance_to_coastline"] = ("Distance to the closest coastline", f"{round(futures[4].result(), 2)} m")
        df, climate_data_info = futures[5].result()
        
        return location_env_info, climate_data_info, df

def plot_chart(df: pd.DataFrame):
    if df is None:
        return 
    st.text("Near surface temperature [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]")
    st.line_chart(
            df,
            x="Month",
            y=["Present day Temperature", "Future Temperature"],
            color=["#d62728", "#2ca02c"],
        )
    st.text("Precipitation [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]")
    st.line_chart(
            df,
            x="Month",
            y=["Present day Precipitation", "Future Precipitation"],
            color=["#d62728", "#2ca02c"],
    )
    st.text("Wind speed [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]")
    st.line_chart(
            df,
            x="Month",
            y=["Present day Wind speed", "Future Wind speed"],
            color=["#d62728", "#2ca02c"],
    )
     

def user_input_callback():
    user_input = st.session_state.user_input 
    if len(user_input) == 0:
        return
    st.session_state.disable_llm_selection = True
    
    session = st.session_state.session
    with session.chat_tab:
        with st.chat_message('user'):
            st.markdown(user_input)
    session.messages.append({'role': 'user', 'content': user_input})
    if not st.session_state.first_user_input:
        location_env_info, climate_data_info, df = get_env_climate_info()
        session.charts_df = df
            
        location_env_info_str = ""
        for v in location_env_info.values():
            location_env_info_str += f"**{v[0]}:** {v[1]}\n\n"
        
        with session.chat_tab:
            with st.chat_message('assistant'):
                st.markdown(location_env_info_str)
        session.messages.append({'role': 'assistant', 'content': location_env_info_str})
        
        # build the first user prompt for the llm
        prompt_dict = {k: v[1] for k, v in location_env_info.items()}
        prompt_dict["user_input"] = user_input
        prompt_dict["lat"] = st.session_state.latitude
        prompt_dict["lon"] = st.session_state.longitude
        prompt_dict["address"] = st.session_state.location_display_name
        prompt_dict.update(climate_data_info)
        
        user_input = content_message.format(**prompt_dict)
        st.session_state.first_user_input = True
        
        if st.session_state.llm == "climategpt":
            user_input = climate_gpt_msg_template.format(**prompt_dict)
            
    with session.chat_tab:
        with st.chat_message('assistant'):
            response = session.chatbot.response(user_input)
            session.messages.append({'role': 'assistant', 'content': response})

@dataclass
class Session:
    chatbot: Chatbot
    messages: list[str] = field(default_factory=lambda: [{'role': 'assistant', 'content': "Describe the activity you would like to evaluate for this location."}])
    charts_df: pd.DataFrame = None
    chat_tab: any = None
    charts_tab: any = None

    def render(self): 
        with self.chat_tab:
            for m in self.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
        st.chat_input('user input', key="user_input", on_submit=user_input_callback)
        
        with self.charts_tab:
            plot_chart(self.charts_df)
