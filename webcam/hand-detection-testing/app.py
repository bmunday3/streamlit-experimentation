import cv2
import time
import base64
import numpy as np
import av
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from modzy.edge import InputSource
from modzy import EdgeClient

client = EdgeClient('localhost', 55000) # change to IP address
MODEL_ID = "esutxpoagu"
MODEL_VERSION = "0.0.2"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# hand landmark detection
mp_hands = mp.solutions.hands
hands_connections = mp_hands.HAND_CONNECTIONS
# mp_faces = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

client.connect()
def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, encoded_frame = cv2.imencode('.jpg', image)
    input_object = InputSource(
        key="image",
        data=encoded_frame.tobytes()
    )
    start = time.time()
    # inference = client.inferences.run(MODEL_ID, MODEL_VERSION, [input_object]) # server mode
    inference = client.inferences.perform_inference(MODEL_ID, MODEL_VERSION, [input_object]) # direct mode      
    end = time.time()
    print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if "results.json" in list(inference.result.outputs.keys()):
        data = base64.b64decode(inference.result.outputs["results.json"].data)
        results_hands = landmark_pb2.NormalizedLandmarkList()
        results_hands.ParseFromString(data)    
        # print(len(results_hands))
        # for hand_landmarks in results_hands2.landmark:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results_hands,
            connections=hands_connections,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )     
                                          
    return cv2.flip(image, 1)


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
