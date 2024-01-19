import threading

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer

# test = st.slider(label="pick a number", min_value=0, max_value=10)

lock = threading.Lock()
# main_thread = threading.main_thread()
img_container = {"img": None, "count": 0}

if "count" not in st.session_state:
    st.session_state.count = 0


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        print("in call back", img_container["count"])
        # print(test)
        img_container["img"] = img
        img_container["count"] += 1
        

    return frame


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

fig_place = st.empty()
fig, ax = plt.subplots(1, 1)

while ctx.state.playing:
    with lock:
        img = img_container["img"]
        st.session_state.count = img_container["count"]
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)
    # print("\n")
    # print(st.session_state.count)
    # print("\n")
    
    st.text(f"in session state {st.session_state.count}")    
    