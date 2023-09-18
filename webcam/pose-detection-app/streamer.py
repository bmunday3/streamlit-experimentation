
import cv2
import json
import time
import base64
import numpy as np
import av
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from modzy.edge import InputSource

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_connections = mp.solutions.hands.HAND_CONNECTIONS
pose_connections = mp.solutions.pose.POSE_CONNECTIONS


def stream_both(client, MODEL_MAPPING):
    def process_both(image):
        image.flags.writeable = False
        # cv2.imwrite("multi-people.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_frame = cv2.imencode('.jpg', image)
        input_object = InputSource(
            key="image",
            data=encoded_frame.tobytes()
        )
        start = time.time()
        inference_hands = client.inferences.perform_inference(MODEL_MAPPING["Hand Landmark Detection"]["id"], MODEL_MAPPING["Hand Landmark Detection"]["version"], [input_object]) # direct mode    
        inference_poses = client.inferences.perform_inference(MODEL_MAPPING["Pose Landmark Detection"]["id"], MODEL_MAPPING["Pose Landmark Detection"]["version"], [input_object]) # direct mode    
        end = time.time()
        print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.
        predictions_hands = [] if isinstance(inference_hands, list) else json.loads(inference_hands.result.outputs["results.json"].data)
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
            
        # Draw the pose annotations on the image.     
        predictions_poses = [] if isinstance(inference_poses, list) else json.loads(inference_poses.result.outputs["results.json"].data)           
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
                                            
        return cv2.flip(image, 1)


    # webrtc component
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_both(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    
def stream_poses(client, MODEL_MAPPING):
    def process_poses(image):
        image.flags.writeable = False
        # cv2.imwrite("multi-people.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_frame = cv2.imencode('.jpg', image)
        input_object = InputSource(
            key="image",
            data=encoded_frame.tobytes()
        )
        start = time.time()
        inference_poses = client.inferences.perform_inference(MODEL_MAPPING["Pose Landmark Detection"]["id"], MODEL_MAPPING["Pose Landmark Detection"]["version"], [input_object]) # direct mode    
        end = time.time()
        print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")      
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw the pose annotations on the image.     
        predictions_poses = [] if isinstance(inference_poses, list) else json.loads(inference_poses.result.outputs["results.json"].data)           
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
                                            
        return cv2.flip(image, 1)


    # webrtc component
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_poses(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    
def stream_hands(client, MODEL_MAPPING):
    def process_hands(image):
        image.flags.writeable = False
        # cv2.imwrite("multi-people.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_frame = cv2.imencode('.jpg', image)
        input_object = InputSource(
            key="image",
            data=encoded_frame.tobytes()
        )

        start = time.time()
        inference_hands = client.inferences.perform_inference(MODEL_MAPPING["Hand Landmark Detection"]["id"], MODEL_MAPPING["Hand Landmark Detection"]["version"], [input_object]) # direct mode            
        inference_poses = []
        end = time.time()
        print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms") 
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.
        predictions_hands = [] if isinstance(inference_hands, list) else json.loads(inference_hands.result.outputs["results.json"].data)
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
                                            
        return cv2.flip(image, 1)


    # webrtc component
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process_hands(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )


def stream_og(client, MODEL_MAPPING):

    def process(image):
        image.flags.writeable = False
        # cv2.imwrite("multi-people.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_frame = cv2.imencode('.jpg', image)
        input_object = InputSource(
            key="image",
            data=encoded_frame.tobytes()
        )
        print(model_selection)
        if model_selection == "Both":
            start = time.time()
            inference_hands = client.inferences.perform_inference(MODEL_MAPPING["Hand Landmark Detection"]["id"], MODEL_MAPPING["Hand Landmark Detection"]["version"], [input_object]) # direct mode    
            inference_poses = client.inferences.perform_inference(MODEL_MAPPING["Pose Landmark Detection"]["id"], MODEL_MAPPING["Pose Landmark Detection"]["version"], [input_object]) # direct mode    
            end = time.time()
            print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")
        elif model_selection == "Hand Landmark Detection":
            start = time.time()
            inference_hands = client.inferences.perform_inference(MODEL_MAPPING["Hand Landmark Detection"]["id"], MODEL_MAPPING["Hand Landmark Detection"]["version"], [input_object]) # direct mode            
            inference_poses = []
            end = time.time()
            print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")
        else:
            start = time.time()
            inference_poses = client.inferences.perform_inference(MODEL_MAPPING["Pose Landmark Detection"]["id"], MODEL_MAPPING["Pose Landmark Detection"]["version"], [input_object]) # direct mode    
            inference_hands = []
            end = time.time()
            print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")      
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.
        predictions_hands = [] if isinstance(inference_hands, list) else json.loads(inference_hands.result.outputs["results.json"].data)
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
            
        # Draw the pose annotations on the image.     
        predictions_poses = [] if isinstance(inference_poses, list) else json.loads(inference_poses.result.outputs["results.json"].data)           
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
                                            
        return cv2.flip(image, 1)


    # webrtc component
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
