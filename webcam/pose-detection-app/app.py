
import streamlit as st
from streamer import stream
from modzy import EdgeClient

st.set_page_config(
    page_title="Pose Detection",
    page_icon="imgs/modzy_badge_v4.png",
    # layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title('Model Selection')
model_selection = st.sidebar.radio(label="Select model for inference", options=["Hand Landmark Detection", "Pose Landmark Detection", "Both"], index=2)
st.sidebar.caption("*Note: Selecting 'Both' may reduce output video stream speed.*")
    
client = EdgeClient('localhost', 55000)
# client = EdgeClient('10.0.0.240', 55000) # change to IP address
# client = EdgeClient('host.docker.internal', 55000) # change to IP address
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

stream(model_selection, client, MODEL_MAPPING)


