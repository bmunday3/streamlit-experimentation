import cv2
import time
import json
import traceback
import threading
import numpy as np
import streamlit as st
from utils import xywh2xyxy, process_frames, make_grid
from modzy import EdgeClient
from modzy.edge import InputSource

# """
# CONFIGURATION & APP INSTANCE VARIABLES
# """
# app configuration
st.set_page_config(
    page_title="Operations Dashboard",
    page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# create app grid
st.title("St. George Street Customer Traffic")
# page_grid = make_grid(2,(3,1,1))
# page_grid[0][0].write("top left")
# page_grid[0][1].write("top right")
# page_grid[1][0].write("bottom left")
# page_grid[1][1].write("bottom right")
# page_grid[0][0].write("uhh???")
# page_grid[0][2].write("test1")
# page_grid[1][2].write("test2")
# page_grid[2][0].write("yes")

# mygrid = make_grid(3,(2,4,4))
mygrid = make_grid(3,(4,1,1,1))

mygrid[0][0].markdown("""
#### World population could be 8 billion by end of 2022
""")
mygrid[0][0].slider(
    "maybe?", 0.0, 1.0, 0.9, 0.05,
)

with mygrid[0][0]:
    subgrid = make_grid(2,2)
    subgrid[0][0].write("please work?")
    subgrid[0][1].write("please work?")
    subgrid[1][1].write("please work?")
    subgrid[1][0].write("please work?")


mygrid[0][1].write("""
Since 1975 the world has been growing by billion people every 12 years. 
It passed 7 billion in 2011 and, by the end of 2022, 
there will be 8 billion people in the world. 
But, the growth rate is below 1%, less than half its peak rate of growth - of 2.3% - in the 1960s.
""")
mygrid[0][2].write("""
Since 1975 the world has been growing by billion people every 12 years. 
It passed 7 billion in 2011 and, by the end of 2022, 
there will be 8 billion people in the world. 
But, the growth rate is below 1%, less than half its peak rate of growth - of 2.3% - in the 1960s
""")
mygrid[0][3].write("""
Since 1975 the world has been growing by billion people every 12 years. 
It passed 7 billion in 2011 and, by the end of 2022, 
there will be 8 billion people in the world. 
But, the growth rate is below 1%, less than half its peak rate of growth - of 2.3% - in the 1960s
""")
mygrid[1][0].markdown("""
#### World population will peak at 10.4 billion in 2086
""")
mygrid[1][1].write("""
The world population has increased rapidly over the last century.
The UN projects that the global population will peak before the end of the century,
 in 2086, at just over 10.4 billion people.
""")
mygrid[1][2].write("""
The world population has increased rapidly over the last century.
The UN projects that the global population will peak before the end of the century,
 in 2086, at just over 10.4 billion people.
""")
mygrid[1][3].write("""
Since 1975 the world has been growing by billion people every 12 years. 
It passed 7 billion in 2011 and, by the end of 2022, 
there will be 8 billion people in the world. 
But, the growth rate is below 1%, less than half its peak rate of growth - of 2.3% - in the 1960s

Since 1975 the world has been growing by billion people every 12 years. 
It passed 7 billion in 2011 and, by the end of 2022, 
there will be 8 billion people in the world. 
But, the growth rate is below 1%, less than half its peak rate of growth - of 2.3% - in the 1960s
""")
mygrid[2][0].markdown("""
#### In 2023 India will overtake China as the world's most populous country
""")
mygrid[2][1].write("""
China is the world's most populous with more than 1.4 billion people. 
Now, its growth rate has fallen due to a rapid drop in fertility rate 
over the 1970s and 80s.

In India, the rate of decline has been slower, so it is expected to overtake China in 2023.
""")

mygrid[2][2].write("""
China is the world's most populous with more than 1.4 billion people. 
Now, its growth rate has fallen due to a rapid drop in fertility rate 
over the 1970s and 80s.

In India, the rate of decline has been slower, so it is expected to overtake China in 2023.
""")

# mygrid[3][0].markdown("""
# #### In 2023 India will overtake China as the world's most populous country
# """)


# video streaming
LIVE_URL = "https://www.youtube.com/watch?v=YLSELFy-iHQ"


# video frame
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# modzy config
MODEL_ID = "ccsvw2eofe"
MODEL_VERSION = "0.0.1"
PRED_OVERLAY = False

# app layout setup
# st.title("MiMi's Famous Crepes Customer Traffic Monitoring")
# st.title("St. George's Street Customer Traffic")
# col1, col2 = st.columns(2)
# frame_placeholder = page_grid[0][0].empty()


# # col2 - video filter configuration
# page_grid[0][1].markdown("### Video Configuration")
# confidence_threshold = page_grid[0][1].slider(
#     "Confidence threshold", 0.0, 1.0, 0.9, 0.05, key="slider_value"
# )
# confidence_threshold = page_grid[0][2].slider(
#     "Confidence threshold", 0.0, 1.0, 0.9, 0.05,
# )
# confidence_threshold = page_grid[0][0].slider(
#     "maybe?", 0.0, 1.0, 0.9, 0.05,
# )

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

