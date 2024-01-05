import pandas as pd
import streamlit as st
from utils import check_streaming_url, make_grid, NAMES
from modzy import EdgeClient

import av
import cv2
import json
import queue
import datetime
import numpy as np
import pandas as pd
from typing import Optional
import streamlit as st
from modzy.edge import InputSource
from utils import make_grid, update_analysis
from utils.util import NAMES, colors, xywh2xyxy
from streamlit_webrtc import WebRtcMode, RTCConfiguration, WebRtcStreamerContext, webrtc_streamer
from aiortc.contrib.media import MediaPlayer

import logging
st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.INFO)
aioice_logger = logging.getLogger("aioice")
aioice_logger.setLevel(logging.INFO)

# lock = threading.Lock()

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
if 'aggregate_df_list' not in st.session_state:
    st.session_state.aggregate_df_list = []
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
        },
        "EfficientNet Object Detection": {
            "id": "lejpphcg7y",
            "version": "0.0.1"
        }
    }  
# if 'model_meta_mapping' not in st.session_state:
#     st.session_state.model_meta_mapping = {
#         "YOLOv8 Object Detection": {
#             "id": "mk8halxkx6",
#             "version": "1.0.0"
#         },
#         "YOLOv8 Object Detection (OpenVINO Optmized)": {
#             "id": "vujnthlpfy",
#             "version": "1.0.0"
#         },
#         "EfficientNet Object Detection": {
#             "id": "pdg9argtdz",
#             "version": "0.0.1"
#         }
#     }              
if 'aggregate_df' not in st.session_state:
    st.session_state.aggregate_df =  pd.DataFrame(
        columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"]
    )   
if 'conf_data' not in st.session_state:
    st.session_state.conf_data = pd.DataFrame()
if 'counts_data' not in st.session_state:
    st.session_state.counts_data = pd.DataFrame()
if 'performance' not in st.session_state:
    st.session_state.performance = pd.DataFrame()        


# video streaming
LIVE_URL = "https://www.youtube.com/watch?v=nyjd_019-aw"
STREAM_URL = st.session_state.stream_url

# create app grid
st.title("AI-Powered Smart City Dashboard")
st.divider()
st.markdown("## Video")
page_grid = make_grid(2,1)

with page_grid[0][0]:
    video_stream_grid = make_grid(1,2, gap="medium")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP RIGHT QUADRANT: video filter configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
video_stream_grid[0][1].markdown("### Configuration")
selected_model = video_stream_grid[0][1].selectbox(
    "*Available Model Options*", 
    # options=["YOLOv8 Object Detection", "YOLOS Tiny Object Detection", "DETR ResNet-50 Object Detection", "DETR ResNet-50 Object Detection"],
    options=["YOLOv8 Object Detection", "YOLOv8 Object Detection (OpenVINO Optmized)", "EfficientNet Object Detection"], 
    index=0
)
# st.selectbox("", [], )
video_stream_grid[0][1].divider()
confidence_threshold = video_stream_grid[0][1].slider(
    "*Confidence threshold*", 0.0, 1.0, 0.15, 0.05, key="slider_value"
)
video_stream_grid[0][1].text(" ")
class_filters = video_stream_grid[0][1].multiselect(
    label = "*Select which class labels to view*",
    options = NAMES,
    default = ['person', 'bicycle', 'car', 'motorcycle', 'traffic light', 'stop sign', 'truck', 'dog', 'parking meter', 'fire hydrant', 'bench', 'bird'],
    key = "class_filters"
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP LEFT QUADRANT: video feed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
client = EdgeClient('localhost', 55000)
client.connect()
check_streaming_url(LIVE_URL, STREAM_URL, st.session_state)
q = queue.Queue()
video_stream_grid[0][0].markdown("### Feed")
with video_stream_grid[0][0]:
    predictions_overlay = st.toggle("Overlay Predictions")
    current_stream_url = st.session_state.stream_url
    model_mapping = st.session_state.model_meta_mapping
    frame_stats_all = []
    
    def create_player():
        return MediaPlayer(current_stream_url)
       
    def video_frame_callback(image):   
        while True:
            try:
                frame_read_start = datetime.datetime.now()
                frame = image.to_ndarray(format="bgr24")
                # cv2.imwrite("test.jpg", frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if predictions_overlay:
                    # Modzy inference & postprocssing       
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    input_object = InputSource(
                        key="image",
                        data=encoded_frame.tobytes()
                    )     
                    inference_time_start = datetime.datetime.now() 
                    # print(input_object)
                    inference = client.inferences.perform_inference(model_mapping[selected_model]["id"], model_mapping[selected_model]["version"], [input_object]) # direct mode
                    # print('inference complete!')
                    # print(inference)
                    postprocess_start = datetime.datetime.now()
                    if len(inference.result.outputs.keys()):
                        results = json.loads(inference.result.outputs['results.json'].data)
                        # Postprocessing
                        preds = [pred for pred in results['data']['result']['detections'] if (pred['class'] in class_filters and pred['score'] >= confidence_threshold)]                   
                        if len(preds):
                            for det in preds:
                                if selected_model == "EfficientNet Object Detection":
                                    box = [det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['x'] + det['bounding_box']['width'], det['bounding_box']['y'] + det['bounding_box']['height']]
                                else:
                                    box = xywh2xyxy(np.array([det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['width'], det['bounding_box']['height']]))       
                                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                label = det['class'] + " " + str(round(det['score'], 3))
                                color = colors(NAMES.index(det['class']), True)
                                # plot bboxes first
                                cv2.rectangle(frame, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
                                # then add text
                                w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
                                outside = p1[1] - h >= 3
                                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                                cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)
                                cv2.putText(frame,
                                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                            0,
                                            2 / 3,
                                            (255,255,255),
                                            thickness=1,
                                            lineType=cv2.LINE_AA)
                            video_frame = frame.copy()
                            postprocess_time_end = datetime.datetime.now()
                            frame_stats = [
                                [
                                    frame_read_start.strftime('%Y-%m-%d %H:%M:%S'), 
                                    det["class"],
                                    det["score"], 
                                    (inference_time_start-frame_read_start).total_seconds(),
                                    (postprocess_start-inference_time_start).total_seconds(),
                                    (postprocess_time_end-postprocess_start).total_seconds()
                                ] for det in preds
                            ]
                            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)                      
                        else:
                            video_frame = frame.copy()
                            frame_read_end = datetime.datetime.now()                         
                            frame_stats = [
                                [
                                    frame_read_start.strftime('%Y-%m-%d %H:%M:%S'),
                                    "No overlay",
                                    "No overlay",
                                    (frame_read_end-frame_read_start).total_seconds(),
                                    0.0,
                                    0.0                            
                                ]
                            ]                                                
                            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)                      
                    else:
                        video_frame = frame.copy()
                        frame_read_end = datetime.datetime.now()                         
                        frame_stats = [
                            [
                                frame_read_start,
                                "No overlay",
                                "No overlay",
                                (frame_read_end-frame_read_start).total_seconds(),
                                0.0,
                                0.0                            
                            ]
                        ]                                           
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)                      
                else:
                    frame_read_end = datetime.datetime.now()                         
                    frame_stats = [
                        [
                            frame_read_start.strftime('%Y-%m-%d %H:%M:%S'),
                            "No overlay",
                            "No overlay",
                            (frame_read_end-frame_read_start).total_seconds(),
                            0.0,
                            0.0                            
                        ]
                    ]          
                    video_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                              
            except Exception as e:
                print(f"ERROR: {e}")
                raise e
            
            try:
                # with lock:
                frame_stats_all.extend(frame_stats)
                q.put(frame_stats_all)                
            except Exception as e:
                print(e)   
            return av.VideoFrame.from_ndarray(video_frame, format="bgr24") 
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="yt-video-stream",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        player_factory=create_player,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    ) 
    

page_grid[0][0].divider()        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP HALF: 1/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[1][0].markdown("## Analysis")
# refresh_button = page_grid[1][0].button(label="Refresh Analysis Results", use_container_width=True)

# TODO - add this section back in
# page_grid[1][0].markdown("### Vehicle Counts")
# with page_grid[1][0]:
#     metrics_grid = make_grid(1,4)
#     # TODO: make these dynamic
#     metrics_grid[0][1].metric(label="*Today*", value=1567, delta="37%")
#     metrics_grid[0][2].metric(label="*Last Hour*", value=35, delta="-78%")
page_grid[1][0].markdown("### Predictions Charts")
with page_grid[1][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Class Prediction Counts over Time*")  
counts_placeholder = page_grid[1][0].empty()
with page_grid[1][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Confidence Scores over Time*")    
confidence_placeholder = page_grid[1][0].empty()
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOTTOM HALF: 2/2 of Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
page_grid[1][0].markdown("### Latency & Performance Statistics")
with page_grid[1][0]:
    title_subgrid = make_grid(1, 3)
    title_subgrid[0][0].markdown("*Mean Latencies (ms) & FPS*")    
performance_placeholder = page_grid[1][0].empty()


while True and ctx.state.playing:
    latest_frame_stats = q.get()
    st.session_state.aggregate_df_list = latest_frame_stats   
    try:
        conf_data, counts_data, performance = update_analysis(st.session_state)
        # print(performance)
        st.session_state.conf_data = conf_data.fillna(0)
        st.session_state.counts_data = counts_data.fillna(0)
        st.session_state.performance = performance
        counts_placeholder.area_chart(st.session_state.counts_data) #, width=300, height=300
        confidence_placeholder.line_chart(st.session_state.conf_data, width=300, height=300)
        performance_placeholder.dataframe(st.session_state.performance, hide_index=True, use_container_width=True)             
    except Exception as e:
        print("ERROR:", e.with_traceback(e.__traceback__))
        continue
    
counts_placeholder.area_chart(st.session_state.counts_data) #, width=300, height=300
confidence_placeholder.line_chart(st.session_state.conf_data, width=300, height=300)
performance_placeholder.dataframe(st.session_state.performance, hide_index=True, use_container_width=True)