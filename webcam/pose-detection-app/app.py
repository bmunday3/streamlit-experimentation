
import streamlit as st
from streamer import stream
from modzy import EdgeClient

st.set_page_config(
    page_title="Pose Detection",
    page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title('Model Selection')
st.sidebar.markdown("Select model(s) for inference")
faces = st.sidebar.toggle(label="Face Detection")
hands = st.sidebar.toggle(label="Hand Landmark Detection")
poses = st.sidebar.toggle(label="Pose Landmark Detection")
st.sidebar.caption("*Note: Selecting more than one model at a time may reduce output video stream speed.*")
    
client = EdgeClient('localhost', 55000)
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
    },
    "Face Detection": {
        "id": "r9zyziw6uk",
        "version": "0.0.2"
    }
}

stream({"faces": faces, "hands": hands, "poses": poses}, client, MODEL_MAPPING)


