
import cv2
import json
import time
import base64
import numpy as np
import av
import mediapipe as mp
from utils import visualize
from mediapipe.framework.formats import landmark_pb2, detection_pb2
from mediapipe.tasks.python.components.containers.detections import DetectionResult
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from modzy.edge import InputSource

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_connections = mp.solutions.hands.HAND_CONNECTIONS
pose_connections = mp.solutions.pose.POSE_CONNECTIONS

def stream(model_selection, client, MODEL_MAPPING):
    
    def video_frame_callback(image):
        image = image.to_ndarray(format="bgr24")
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_frame = cv2.imencode('.jpg', image)
        input_object = InputSource(
            key="image",
            data=encoded_frame.tobytes()
        )
        start = time.time()
        count = 0
        if model_selection["faces"]:
            count += 1
            try:
                inference_faces = client.inferences.perform_inference(MODEL_MAPPING["Face Detection"]["id"], MODEL_MAPPING["Face Detection"]["version"], [input_object]) # direct mode    
            except Exception as e:
                print(e)
                raise e
            # Draw face bboxes on image
            image.flags.writeable = True
            predictions_faces = json.loads(inference_faces.result.outputs["results.json"].data)[0]
            faces_proto = detection_pb2.DetectionList()
            faces_proto.ParseFromString(base64.b64decode(predictions_faces))
            detection_results = DetectionResult
            detection_results = detection_results.create_from_pb2(faces_proto)
            image = visualize(image, detection_results)            
        
        if model_selection["hands"]:
            count += 1            
            try:
                inference_hands = client.inferences.perform_inference(MODEL_MAPPING["Hand Landmark Detection"]["id"], MODEL_MAPPING["Hand Landmark Detection"]["version"], [input_object]) # direct mode    
            except Exception as e:
                print(e)
                raise e     
            # Draw the hand annotations on the image.            
            image.flags.writeable = True
            predictions_hands = json.loads(inference_hands.result.outputs["results.json"].data)
            if predictions_hands:
                for landmarks in predictions_hands:
                    results_hands = landmark_pb2.NormalizedLandmarkList()
                    results_hands.ParseFromString(base64.b64decode(landmarks))
                    # results_hands.append(proto_object)       
                    # for hand_landmarks in results_hands2.landmark:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=results_hands,
                        connections=hands_connections,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
            
        if  model_selection["poses"]:
            count += 1
            try:
                inference_poses = client.inferences.perform_inference(MODEL_MAPPING["Pose Landmark Detection"]["id"], MODEL_MAPPING["Pose Landmark Detection"]["version"], [input_object]) # direct mode    
            except Exception as e:
                print(e)
                raise e             
            # Draw the pose annotations on the image.                 
            image.flags.writeable = True
            predictions_poses = json.loads(inference_poses.result.outputs["results.json"].data)           
            if predictions_poses:
                for landmarks in predictions_poses:
                    results_poses = landmark_pb2.NormalizedLandmarkList()
                    results_poses.ParseFromString(base64.b64decode(landmarks))
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=results_poses,
                        connections=pose_connections,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )                                   
     
        end = time.time()
        print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms with {count} models run")
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                      
        image = cv2.flip(image, 1)                           
        return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    # webrtc component
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )                              