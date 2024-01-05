import streamlit as st
import pandas as pd
from utils import make_grid, update_analysis

st.set_page_config(
    page_title="Analysis Dashboard",
    page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# add keys to session state to be populated later
if 'aggregate_df' not in st.session_state:
    st.session_state.aggregate_df =  pd.DataFrame(
        columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"]
    )   
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0
if 'person_count' not in st.session_state:
    st.session_state.person_count = 0   
if 'conf_data' not in st.session_state:
    st.session_state.conf_data = pd.DataFrame()
if 'counts_data' not in st.session_state:
    st.session_state.counts_data = pd.DataFrame()
if 'performance' not in st.session_state:
    st.session_state.performance = pd.DataFrame()        

# create page grid
st.title("Analysis Dashboard")
st.divider()
page_grid = make_grid(1,1)

# st.text(st.session_state.aggregate_df_list)
# st.text(len(st.session_state.aggregate_df_list))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP RIGHT QUADRANT: 1/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
refresh_button = page_grid[0][0].button(label="Refresh Analysis Results", use_container_width=True)
page_grid[0][0].markdown("### Vehicle Counts")
with page_grid[0][0]:
    metrics_grid = make_grid(1,4)
    # TODO: make these dynamic
    metrics_grid[0][1].metric(label="*Today*", value=1567, delta="37%")
    metrics_grid[0][2].metric(label="*Last Hour*", value=35, delta="-78%")
page_grid[0][0].markdown("### Predictions Charts")
with page_grid[0][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Class Prediction Counts over Time*")  
counts_placeholder = page_grid[0][0].empty()
with page_grid[0][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Confidence Scores over Time*")    
confidence_placeholder = page_grid[0][0].empty()
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOTTOM RIGHT QUADRANT: 2/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
page_grid[0][0].markdown("### Latency & Performance Statistics")
with page_grid[0][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Mean Latencies (ms) & FPS*")    
performance_placeholder = page_grid[0][0].empty()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS REFRESH: updates multiple parts of page if pressed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if refresh_button:
    update_analysis(st.session_state["aggregate_df"], st.session_state)
    # confidence_df, counts_df, performance_data = update_analysis(st.session_state["aggregate_df"])
    # # class aggregated counts
    # print(st.session_state)
    counts_placeholder.area_chart(st.session_state.counts_data, width=300, height=300)
    # # confidence score
    confidence_placeholder.line_chart(st.session_state.conf_data, width=300, height=300)
    performance_placeholder.dataframe(st.session_state.performance, hide_index=True, use_container_width=True)
