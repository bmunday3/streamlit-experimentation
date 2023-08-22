import cv2
import json
import yt_dlp
import datetime
import threading
import traceback
import pandas as pd
import numpy as np
import streamlit as st
from .util import xywh2xyxy
from modzy.edge import InputSource
from modzy import EdgeClient


# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# dataframe
global aggregate_frame_df
aggregate_frame_df = pd.DataFrame(
    columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"]
)

# video frame
global video_frame
video_frame = None

# postprocessing config
TXT_COLOR=(255, 255, 255)
KEEP_FORMAT_IDS = ['94','95','96']

def get_streaming_url(link):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        format_ids = {format['format_id']:format['url'] for format in info['formats']}
        if not any(id in list(format_ids.keys()) for id in KEEP_FORMAT_IDS):
            raise ValueError("Format IDs 94, 95, 96 not available in YouTube link")
        elif '94' in list(format_ids.keys()):
            stream_url = format_ids['94']
        elif '95' in list(format_ids.keys()):
            stream_url = format_ids['95']        
        elif '96' in list(format_ids.keys()):
            stream_url = format_ids['96']
    return stream_url     

def check_streaming_url(youtube_link, url):
    if url is not None:
        try:
            _ = cv2.VideoCapture(url)
            stream_url = url
            return stream_url
        except Exception as e:
            stream_url = get_streaming_url(youtube_link)
            return stream_url
    else:
        stream_url = get_streaming_url(youtube_link)
    return stream_url

# @st.cache_resource
def process_frames(live_url, host, port, frame_placeholder, stop, overlay_on, overlay_off, session_state, model_id, model_version):
    global aggregate_frame_df
    video_capture = cv2.VideoCapture(live_url)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # count = 0
    client = EdgeClient(host, port)
    client.connect()
    while video_capture.isOpened() and not stop:
        try:
            frame_read_start = datetime.datetime.now()
            return_key, frame = video_capture.read()
            if not return_key:
                print("Video stream could not be read")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with thread_lock:
                if overlay_on:
                    # Modzy inference & postprocssing       
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    input_object = InputSource(
                        key="image",
                        data=encoded_frame.tobytes()
                    )     
                    inference_time_start = datetime.datetime.now() 
                    # inference = client.inferences.run(model_id, model_version, [input_object]) # server mode
                    inference = client.inferences.perform_inference(model_id, model_version, [input_object]) # direct mode
                    postprocess_start = datetime.datetime.now()
                    # print(inference)
                    if len(inference.result.outputs.keys()):
                        results = json.loads(inference.result.outputs['results.json'].data)
                        # Postprocessing
                        print(session_state)
                        preds = [pred for pred in results['data']['result']['detections'] if (pred['class'] in session_state['class_filters'] and pred['score'] >= session_state['slider_value'])]                   
                        
                        if len(preds):
                            for det in preds:
                                box = xywh2xyxy(np.array([det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['width'], det['bounding_box']['height']]))#.tolist()                     
                                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                label = det['class'] + " " + str(round(det['score'], 3))
                                color = (255,0,0)
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
                                            TXT_COLOR,
                                            thickness=1,
                                            lineType=cv2.LINE_AA)
                            video_frame = frame.copy()
                            postprocess_time_end = datetime.datetime.now()
                            frame_placeholder.image(image=video_frame, caption="St. George Street, St. Augustine, FL")
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
                    else:
                        continue          
                else:
                    frame_read_end = datetime.datetime.now()                         
                    frame_placeholder.image(image=frame, caption="St. George Street, St. Augustine, FL") 
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
            if stop:
                break    
            
            # aggregate data frame
            frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
            # print(frame_df.shape)
            aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
            # print(aggregate_frame_df.shape)
            session_state.aggregate_df = aggregate_frame_df
        except Exception as e:
            print(traceback.print_exc())
            # print(f"ERROR:\n{e.with_traceback()}")
            break
    # aggregate_frame_df.to_csv("data_agg.csv", index=False)
    video_capture.release() 