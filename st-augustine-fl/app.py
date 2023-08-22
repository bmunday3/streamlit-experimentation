import cv2
import time
import json
import traceback
import pandas as pd
import threading
import numpy as np
import streamlit as st
from utils import process_frames, check_streaming_url, make_grid, update_analysis, NAMES
# from utils import StreamerUtils
from modzy import EdgeClient
from modzy.edge import InputSource

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

if 'aggregate_df' not in st.session_state:
    st.session_state.aggregate_df =  pd.DataFrame(
        columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"]
    )         

global stream_url
stream_url = None

# video streaming
LIVE_URL = "https://www.youtube.com/watch?v=YLSELFy-iHQ"

# modzy config
# yolov8
# MODEL_ID = "avlilexzvu" #demo
MODEL_ID = "mk8halxkx6" #dev
MODEL_VERSION = "1.0.0"
# yolos
# MODEL_ID = "ccsvw2eofe"
# MODEL_VERSION = "0.0.1"

# create app grid
st.title("St. George Street Customer Traffic")
page_grid = make_grid(2,2,gap="medium")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOTTOM LEFT QUADRANT: video filter configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[1][0].markdown("### Video Configuration")
page_grid[1][0].selectbox(
    "*Available Model Options*", 
    options=["YOLOv8 Object Detection", "YOLOS Tiny Object Detection", "DETR ResNet-50 Object Detection", "DETR ResNet-50 Object Detection"], index=0)
confidence_threshold = page_grid[1][0].slider(
    "*Confidence threshold*", 0.0, 1.0, 0.15, 0.05, key="slider_value"
)
class_filters = page_grid[1][0].multiselect(
    label = "*Select which class labels to view*",
    options = NAMES,
    default = ['person', 'bicycle', 'stop sign', 'umbrella', 'clock'],
    key = "class_filters"
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP RIGHT QUADRANT: 1/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[0][1].markdown("### Analysis")
page_grid[0][1].markdown("##### Pedestrian Counts")
with page_grid[0][1]:
    metrics_grid = make_grid(1,4)
    # TODO: make these dynamic
    metrics_grid[0][1].metric(label="*Today*", value=1567, delta="37%")
    metrics_grid[0][2].metric(label="*Last Hour*", value=35, delta="-78%")
page_grid[0][1].markdown("##### Predictions Charts")
charts_placeholder = page_grid[0][1].empty()
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOTTOM RIGHT QUADRANT: 2/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
page_grid[1][1].markdown("##### Latency & Performance Statistics")
with page_grid[1][1]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][1].markdown("*Mean Latencies (ms) & FPS*")    
performance_placeholder = page_grid[1][1].empty()
refresh_button = page_grid[1][1].button(label="Refresh Analysis Results", use_container_width=True)
if refresh_button:
    # REPLACE WITH AGGREGATE DF
    # test_df = pd.read_csv("data_agg.csv")
    # print(aggregate_frame_df.shape)
    confidence_df, counts_df, performance_data = update_analysis(st.session_state["aggregate_df"])
    # chart_data, performance_data = update_analysis(test_df)
    # people counts
    # confidence score
    charts_placeholder.line_chart(confidence_df, width=300, height=300)
    performance_placeholder.dataframe(performance_data, hide_index=True, use_container_width=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP LEFT QUADRANT: video feed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
page_grid[0][0].markdown("### Video Feed")
frame_placeholder = page_grid[0][0].empty() # same video size --> see how i can format this
with page_grid[0][0]:    
    tl_subgrid = make_grid(1,4)
    start_button_pressed = tl_subgrid[0][0].button("Start Feed", use_container_width=True)
    stop_button_pressed = tl_subgrid[0][1].button("Stop Feed", use_container_width=True)
    overlay_button_on = tl_subgrid[0][2].button("Overlay On", use_container_width=True)
    overlay_button_off = tl_subgrid[0][3].button("Overlay Off", use_container_width=True)
LIVE_URL = "https://www.youtube.com/watch?v=YLSELFy-iHQ"
streaming_url = check_streaming_url(LIVE_URL, stream_url)
process_frames(streaming_url, 'localhost', 55000, frame_placeholder, stop_button_pressed, overlay_button_on, overlay_button_off, st.session_state, MODEL_ID, MODEL_VERSION)    







# # col 1 - video streaming (streamlit code embedded within process_frames function
# col1.markdown("### Video Feed")
# # URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692307243/ei/yzreZJXDB_eblu8Pz5yUmAU/ip/66.211.86.66/id/YLSELFy-iHQ.1/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/rr5---sn-vgqsknzs.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9fw9JgGoPC77IKDn1ZpeiZmm5EYg/vprv/1/playlist_type/DVR/initcwndbps/521250/mh/BT/mm/44/mn/sn-vgqsknzs/ms/lva/mv/m/mvi/5/pl/24/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000023/mt/1692285404/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRgIhANmg9gPqJ2Rgu3c1A41AL3mB4NtRbQQdoyxxd_1Ijl7XAiEApJO4p_0D2t6NkOWMEbMP9s6liMrHiddGPSnifCk0tp4%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIgFhMZ3ooVRTg4mA0G18kynFeXEyVuW822Tfib7EVC0ewCIQDm5UaK61J64TzRANFvPadCAfHgPCgJc8Lk7icAgXU8MQ%3D%3D/playlist/index.m3u8"
# # process_frames(URL, 'localhost', 55000, frame_placeholder, stop_button_pressed, overlay_button_pressed, MODEL_ID, MODEL_VERSION)

# # container 1 - buttons underneath video
# container1 = col1.container()
# stop_button_pressed = container1.button("Stop Feed")
# overlay_button_pressed = container1.button("Overlay Predictions")




# # session state testing
# col1.divider()
# col1.write('Session State: ')
# col1.write(st.session_state)

# pounds = col1.number_input("pounds", key="lbs")
# kilogram = col1.number_input("kilos", key="kg")

