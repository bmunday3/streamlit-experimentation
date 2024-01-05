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
page_grid = make_grid(1,2,gap="medium")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOP RIGHT QUADRANT: video filter configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
page_grid[0][1].markdown("### Video Configuration")
selected_model = page_grid[0][1].selectbox(
    "*Available Model Options*", 
    # options=["YOLOv8 Object Detection", "YOLOS Tiny Object Detection", "DETR ResNet-50 Object Detection", "DETR ResNet-50 Object Detection"],
    options=["YOLOv8 Object Detection", "YOLOv8 Object Detection (OpenVINO Optmized)", "EfficientNet Object Detection"], 
    index=0
)
# st.selectbox("", [], )
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
# TOP LEFT QUADRANT: video feed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
client = EdgeClient('localhost', 55000)
client.connect()
check_streaming_url(LIVE_URL, STREAM_URL, st.session_state)
q = queue.Queue()
page_grid[0][0].markdown("### Video Feed")
with page_grid[0][0]:
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if predictions_overlay:
                    # Modzy inference & postprocssing       
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    input_object = InputSource(
                        key="image",
                        data=encoded_frame.tobytes()
                    )     
                    inference_time_start = datetime.datetime.now() 
                    inference = client.inferences.perform_inference(model_mapping[selected_model]["id"], model_mapping[selected_model]["version"], [input_object]) # direct mode
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
                                    frame_read_start, 
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
                            frame_read_start,
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
                frame_stats_all.append(frame_stats)
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
    
    while True:
        latest_frame_stats = q.get()
        st.session_state.aggregate_df_list = latest_frame_stats
        
