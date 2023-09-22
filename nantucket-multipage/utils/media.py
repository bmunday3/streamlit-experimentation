import av
import cv2
import json
import datetime
import pandas as pd
import streamlit as st
from modzy.edge import InputSource
from .util import NAMES, colors
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, RTCConfiguration, webrtc_streamer
from aiortc.contrib.media import MediaPlayer

import logging
st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.INFO)
aioice_logger = logging.getLogger("aioice")
aioice_logger.setLevel(logging.INFO)

TXT_COLOR=(255, 255, 255)

'''
class Media:
    def __init__(self, stream_url):
        # print(st.session_state.stream_url)
        self.stream_url = stream_url
    
    def create_player(self):
        return MediaPlayer(self.stream_url)

    def stream(self, client, overlay, session_state, selected_model):
        # global aggregate_frame_df
        
        def video_frame_callback(image):
            print("\n\n")
            print("process!!")
            print("\n\n")
            try:
                frame_read_start = datetime.datetime.now()
                frame = image.to_ndarray(format="bgr24")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                print(overlay)
                if overlay:
                    # Modzy inference & postprocssing       
                    _, encoded_frame = cv2.imencode('.jpg', frame)
                    input_object = InputSource(
                        key="image",
                        data=encoded_frame.tobytes()
                    )     
                    inference_time_start = datetime.datetime.now() 
                    # inference = client.inferences.run(session_state["model_meta_mapping"][selected_model]["id"], session_state["model_meta_mapping"][selected_model]["version"], [input_object]) # server mode
                    inference = client.inferences.perform_inference(session_state["model_meta_mapping"][selected_model]["id"], session_state["model_meta_mapping"][selected_model]["version"], [input_object]) # direct mode
                    postprocess_start = datetime.datetime.now()
                    if len(inference.result.outputs.keys()):
                        results = json.loads(inference.result.outputs['results.json'].data)
                        # Postprocessing
                        preds = [pred for pred in results['data']['result']['detections'] if (pred['class'] in session_state['class_filters'] and pred['score'] >= session_state['slider_value'])]                   
                        if len(preds):
                            for det in preds:
                                box = [det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['width'], det['bounding_box']['height']]               
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
                                            TXT_COLOR,
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
                            # aggregate data frame
                            frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                            aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                            session_state.aggregate_df = aggregate_frame_df                        
                            return av.VideoFrame.from_ndarray(video_frame, format="bgr24")
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
                            # aggregate data frame
                            frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                            aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                            session_state.aggregate_df = aggregate_frame_df                              
                            return av.VideoFrame.from_ndarray(frame, format="bgr24")                           
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
                        # aggregate data frame
                        frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                        aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                        session_state.aggregate_df = aggregate_frame_df                                            
                        return av.VideoFrame.from_ndarray(frame, format="bgr24")          
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
                    
                    # aggregate data frame
                    frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                    aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                    session_state.aggregate_df = aggregate_frame_df
                    return av.VideoFrame.from_ndarray(frame, format="bgr24")  
            except Exception as e:
                raise e

            # webrtc component
        
        
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            player_factory=self.create_player,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        ) 
        
'''

def stream(client, overlay, session_state, selected_model):
    def create_player():
        return MediaPlayer(session_state.stream_url)
    
    print(session_state.stream_url)    
    def video_frame_callback(image):
        print("\n\n")
        print("process!!")
        print("\n\n")
        try:
            frame_read_start = datetime.datetime.now()
            frame = image.to_ndarray(format="bgr24")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            print(overlay)
            if overlay:
                # Modzy inference & postprocssing       
                _, encoded_frame = cv2.imencode('.jpg', frame)
                input_object = InputSource(
                    key="image",
                    data=encoded_frame.tobytes()
                )     
                inference_time_start = datetime.datetime.now() 
                # inference = client.inferences.run(session_state["model_meta_mapping"][selected_model]["id"], session_state["model_meta_mapping"][selected_model]["version"], [input_object]) # server mode
                inference = client.inferences.perform_inference(session_state["model_meta_mapping"][selected_model]["id"], session_state["model_meta_mapping"][selected_model]["version"], [input_object]) # direct mode
                postprocess_start = datetime.datetime.now()
                if len(inference.result.outputs.keys()):
                    results = json.loads(inference.result.outputs['results.json'].data)
                    # Postprocessing
                    preds = [pred for pred in results['data']['result']['detections'] if (pred['class'] in session_state['class_filters'] and pred['score'] >= session_state['slider_value'])]                   
                    if len(preds):
                        for det in preds:
                            box = [det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['width'], det['bounding_box']['height']]               
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
                                        TXT_COLOR,
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
                        # aggregate data frame
                        frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                        aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                        session_state.aggregate_df = aggregate_frame_df                        
                        return av.VideoFrame.from_ndarray(video_frame, format="bgr24")
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
                        # aggregate data frame
                        frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                        aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                        session_state.aggregate_df = aggregate_frame_df                              
                        return av.VideoFrame.from_ndarray(frame, format="bgr24")                           
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
                    # aggregate data frame
                    frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                    aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                    session_state.aggregate_df = aggregate_frame_df                                            
                    return av.VideoFrame.from_ndarray(frame, format="bgr24")          
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
                
                # aggregate data frame
                frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Object", "Confidence Score", "Frame Read Time", "Inference Time", "Postprocess Time"])
                aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)        
                session_state.aggregate_df = aggregate_frame_df
                return av.VideoFrame.from_ndarray(frame, format="bgr24")  
        except Exception as e:
            raise e

        # webrtc component
    
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        player_factory=create_player,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    ) 
                    
                
        