import pandas as pd
import streamlit as st
from utils import process_frames, check_streaming_url, make_grid, update_analysis, NAMES

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# APP CONFIGURATION & INSTANCE VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# app configuration
st.set_page_config(
    page_title="Operations Dashboard",
    page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.markdown("## AI-Powered Smart City Dashboard")

# add keys to session state to be populated later
if 'aggregate_df' not in st.session_state:
    st.session_state.aggregate_df =  pd.DataFrame(
        columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"]
    )   
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0
if 'person_count' not in st.session_state:
    st.session_state.person_count = 0    
if 'stream_url' not in st.session_state:
    st.session_state.stream_url = None
if 'model_meta_mapping' not in st.session_state:
    st.session_state.model_meta_mapping = {
        "YOLOv8 Object Detection": {
            "id": "avlilexzvu",
            "version": "1.0.0"
        },
        "YOLOv8 Object Detection (OpenVINO Optmized)": {
            "id": "dyywxasii6",
            "version": "1.0.0"
        }
    }          

# video streaming
LIVE_URL = "https://www.youtube.com/watch?v=nyjd_019-aw"
STREAM_URL = st.session_state.stream_url

# modzy config
# yolov8
# MODEL_ID = "avlilexzvu" #demo
# # MODEL_ID = "mk8halxkx6" #dev
# MODEL_VERSION = "1.0.0"
# yolos
# MODEL_ID = "ccsvw2eofe"
# MODEL_VERSION = "0.0.1"

# create app grid
st.title("AI-Powered Smart City Dashboard")
st.divider()
page_grid = make_grid(1,2,gap="medium")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP RIGHT QUADRANT: video filter configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[0][1].markdown("### Video Configuration")
selected_model = page_grid[0][1].selectbox(
    "*Available Model Options*", 
    # options=["YOLOv8 Object Detection", "YOLOS Tiny Object Detection", "DETR ResNet-50 Object Detection", "DETR ResNet-50 Object Detection"],
    options=["YOLOv8 Object Detection", "YOLOv8 Object Detection (OpenVINO Optmized)"], 
    index=0
)
st.selectbox("", [], )
page_grid[0][1].divider()
confidence_threshold = page_grid[0][1].slider(
    "*Confidence threshold*", 0.0, 1.0, 0.15, 0.05, key="slider_value"
)
page_grid[0][1].text(" ")
class_filters = page_grid[0][1].multiselect(
    label = "*Select which class labels to view*",
    options = NAMES,
    default = ['person', 'bicycle', 'car', 'motorcycle', 'traffic light', 'stop sign', 'truck', 'dog', 'parking meter', 'fire hydrant', 'bench', 'bird'],
    key = "class_filters"
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOTTOM RIGHT QUADRANT: video feed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[0][0].markdown("### Video Feed")
frame_placeholder = page_grid[0][0].empty() # same video size --> see how i can format this
with page_grid[0][0]:    
    tl_subgrid = make_grid(1,4)
    start_button_pressed = tl_subgrid[0][0].button("Start Feed", use_container_width=True)
    stop_button_pressed = tl_subgrid[0][1].button("Stop Feed", use_container_width=True)
    overlay_button_on = tl_subgrid[0][2].button("Overlay On", use_container_width=True)
    overlay_button_off = tl_subgrid[0][3].button("Overlay Off", use_container_width=True)

check_streaming_url(LIVE_URL, STREAM_URL, st.session_state)
process_frames(st.session_state.stream_url, 'localhost', 55000, frame_placeholder, stop_button_pressed, overlay_button_on, overlay_button_off, st.session_state, selected_model)    


