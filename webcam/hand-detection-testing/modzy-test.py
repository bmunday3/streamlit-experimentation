import cv2
import base64
import numpy as np
from modzy.edge import InputSource
from modzy import EdgeClient
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# hands connections
mp_hands = mp.solutions.hands
hands_connections = mp_hands.HAND_CONNECTIONS

# initialize modzy stuff and model
client = EdgeClient('localhost', 55000)
client.connect()
MODEL_ID = "esutxpoagu"
MODEL_VERSION = "0.0.2"

# drawing object
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# sample image for testing
image = cv2.imread("brad-face-hand-input.jpg")
# image = cv2.imread("brad-face-hand.jpg")
image.flags.writeable = False
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
_, encoded_frame = cv2.imencode('.jpg', image)
input_object = InputSource(
    key="image",
    data=encoded_frame.tobytes()
)
# inference = client.inferences.run(MODEL_ID, MODEL_VERSION, [input_object]) # server mode
inference = client.inferences.perform_inference(MODEL_ID, MODEL_VERSION, [input_object]) # direct mode    
print("results.json" in list(inference.result.outputs.keys()))
test = base64.b64decode(inference.result.outputs["results.json"].data)
print(test)
results_hands = landmark_pb2.NormalizedLandmarkList()
results_hands.ParseFromString(test)
# print(results_hands)

out_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if "results.json" in list(inference.result.outputs.keys()):
    data = base64.b64decode(inference.result.outputs["results.json"].data)
    results_hands = landmark_pb2.NormalizedLandmarkList()
    results_hands.ParseFromString(data)    
    # for hand_landmarks in results_hands2.landmark:
    mp_drawing.draw_landmarks(
        image=out_img,
        landmark_list=results_hands,
        connections=hands_connections,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )     
cv2.imwrite("test-out-3.jpg", out_img) 

