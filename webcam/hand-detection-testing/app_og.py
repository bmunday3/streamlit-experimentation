import cv2
import time
import numpy as np
import av
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# faces detection
mp_faces = mp.solutions.face_mesh
faces = mp_faces.FaceMesh()

# pose detection
mp_pose = mp.solutions.pose
poses = mp_pose.Pose

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    # cv2.imwrite("brad-face-hand.jpg", image)
    start = time.time()
    results_hands = hands.process(image) # pure python
    results_faces = faces.process(image)
    results_poses = poses.process(image)    
    
    end = time.time()
    print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")
    # print(results())

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
    # if results_faces.multi_face_landmarks:
    #     # print("facess")
    #     for face_landmarks in results_faces.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             image=image,
    #             landmark_list=face_landmarks,
    #             connections=mp_faces.FACEMESH_TESSELATION,
    #             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()            
    #         )                                       
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
