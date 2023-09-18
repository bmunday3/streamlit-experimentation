
import streamlit as st
from streamer import stream_both, stream_hands, stream_poses
from modzy import EdgeClient

st.set_page_config(
    page_title="Pose Detection",
    page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title('Model Selection')
model_selection = st.sidebar.radio(label="Select model for inference", options=["Hand Landmark Detection", "Pose Landmark Detection", "Both"], index=2)
st.sidebar.caption("*Note: Selecting 'Both' may reduce output video stream speed.*")
    
client = EdgeClient('localhost', 55000) # change to IP address
client.connect()

MODEL_MAPPING = {
    "Hand Landmark Detection": {
        "id": "esutxpoagu",
        "version": "0.0.6"
    },
    "Pose Landmark Detection": {
        "id": "9odlhi0ylq",
        "version": "0.0.1"        
    }
}

print(model_selection)
if model_selection == "Both":
    stream_both(client, MODEL_MAPPING)
elif model_selection == "Hand Landmark Detection":
    stream_hands(client, MODEL_MAPPING)
else:
    stream_poses(client, MODEL_MAPPING)


