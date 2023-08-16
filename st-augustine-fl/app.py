import cv2
import time
import json
import traceback
import threading
import numpy as np
import streamlit as st
from utils import xywh2xyxy
from modzy import EdgeClient
from modzy.edge import InputSource

# """
# CONFIGURATION & APP INSTANCE VARIABLES
# """
# app configuration
st.set_page_config(
    page_title="MiMi's Famous Crepes",
    # page_icon="imgs/modzy_badge_v4.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# video streaming config

# video frame
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# modzy config
MODEL_ID = "ccsvw2eofe"
MODEL_VERSION = "0.0.1"
PRED_OVERLAY = False

# postprocessing config
TXT_COLOR=(255, 255, 255)


st.title("MiMi's Famous Crepes Customer Traffic Monitoring")
col1, col2 = st.columns(2)
frame_placeholder = col1.empty()
container1 = col1.container()
stop_button_pressed = container1.button("Stop Feed")
overlay_button_pressed = container1.button("Overlay Predictions")
# st.divider()
# col1.video("https://www.youtube.com/watch?v=YLSELFy-iHQ")

# capture video from youtube with opencv and stream to streamlit app
# STREAM_URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692136175/ei/j57bZOjIBJmN_9EPm4yUkA4/ip/2601:43:200:3750:5053:ae4c:2344:5b79/id/YLSELFy-iHQ.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9f99Acofn0zybTzEVfYHH65IRhYo/vprv/1/playlist_type/DVR/initcwndbps/1573750/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pcm2cms/yes/pl/34/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000023/mt/1692114072/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRQIhANSwxOGLgou-REgbh5YGHHjIR-IKtGGNI6AjR6Ta86APAiBGwoauJXa0PTTKhcoSUqyHzA9OLvh6VtYgeRlfFdkH1A%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pcm2cms,pl/lsig/AG3C_xAwRQIgPQAsiesWI_jfP-pSvzz-KZhgQAmbZJ89R8dJO5T56CkCIQDw-qmrBmn6-YMTpMYhn9o8szMkEqhUYxcddDT44sEs6w%3D%3D/playlist/index.m3u8"
# high res (8/15)
# video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692136175/ei/j57bZOjIBJmN_9EPm4yUkA4/ip/2601:43:200:3750:5053:ae4c:2344:5b79/id/YLSELFy-iHQ.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9f99Acofn0zybTzEVfYHH65IRhYo/vprv/1/playlist_type/DVR/initcwndbps/1573750/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pcm2cms/yes/pl/34/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000023/mt/1692114072/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRQIhANSwxOGLgou-REgbh5YGHHjIR-IKtGGNI6AjR6Ta86APAiBGwoauJXa0PTTKhcoSUqyHzA9OLvh6VtYgeRlfFdkH1A%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pcm2cms,pl/lsig/AG3C_xAwRQIgPQAsiesWI_jfP-pSvzz-KZhgQAmbZJ89R8dJO5T56CkCIQDw-qmrBmn6-YMTpMYhn9o8szMkEqhUYxcddDT44sEs6w%3D%3D/playlist/index.m3u8")
# medium res (8/15)
# video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692152235/ei/S93bZMKnOr-N_9EPuLyL2A0/ip/2601:43:200:3750:5053:ae4c:2344:5b79/id/YLSELFy-iHQ.1/itag/231/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/3600/manifest_duration/3600/vprv/1/playlist_type/DVR/initcwndbps/1646250/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pcm2cms/yes/pl/34/dover/13/pacing/0/short_key/1/keepalive/yes/fexp/24007246,51000024/beids/24350017/mt/1692130126/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRAIgGVEfQgyDF5AHF-lz8zII8wqNv8IgSHoMGCSQYdz01NMCIDwh8D43FezqslN1gepR-rC-zsebprE0gNQt08ky-Isy/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pcm2cms,pl/lsig/AG3C_xAwRQIgda63_VYa1N8HkTeOoaF3Z2x9nlNa0er-76h2VMkd-2ACIQC2GJNJ-OIUJywVBR5WxlcSVZQEf_uTxnx34jTaopJMxw%3D%3D/playlist/index.m3u8")
# medium res (8/16)
# video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692203395/ei/I6XcZIrqC_-W_9EP4NyNgAo/ip/2601:43:200:3750:586d:7795:39dc:2ed6/id/YLSELFy-iHQ.1/itag/95/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D136/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9f4oc1h2FhcW31mE13W1px3yUDw4/vprv/1/playlist_type/DVR/initcwndbps/1610000/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pl/34/dover/11/pacing/0/keepalive/yes/fexp/24007246/beids/24350018/mt/1692181483/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRQIhAKgc6Qv7M9JLmrEAwMHv1oW6gOXKx2jDcwfnbYn3DBnvAiBDnqPpsvJDhyOynlQzUVIaiFXwBUfp2iDRtd7gD_Mxfg%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRAIgQof4WX0AnnuVhXXzbPVIUD878qr6ve8VtJdb7yhWzZICIBWbjd9P8uTFJSY2OoMq1QqxHKmDED-q34KKodV152RH/playlist/index.m3u8")
# low res (8/16)
# video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692205776/ei/cK7cZLGRKo6N_9EPj62w4AI/ip/2601:43:200:3750:586d:7795:39dc:2ed6/id/YLSELFy-iHQ.1/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9fzn5X4zyjhNlJl9XjDZ_rHyLkE8/vprv/1/playlist_type/DVR/initcwndbps/1581250/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pl/34/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000023/mt/1692183885/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRQIhALlq2mrRJGV4ypuKSekO7bbqZ1Dt_fGVvA82snYytn-tAiBaYmDawuhwCOT97TSYMmQhuWX0n7fi9Gw1AUIfrdj1Cg%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRgIhANaV7x-32g06_fGUGYhRcbCn23zh6SmF9rG9lRCfm9VMAiEAnoHhw0UcDONf6OGy4C_7yN7WvfmSytCbajJpud7D_40%3D/playlist/index.m3u8")
video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692229287/ei/RwrdZNXWG8jl8wSIvJGADw/ip/68.80.59.169/id/YLSELFy-iHQ.1/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9f8IGX6WdGHMrwP2dOvugGiThjto/vprv/1/playlist_type/DVR/initcwndbps/1553750/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pl/21/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000022/mt/1692207403/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRAIgHbYImmv4ZocfxDxvVjw_86bSfhOtEwwzHxOMFT8u3BECIA0unkKE5n77du474rdEWhyFIyRDANhCv7Ahobil1YiA/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIgW3j8GBNOK8N8oyggrNqhA9EdRy6uMGeJwKCfdg8hJY8CIQCTxqGLX_BkOTbXd1PfRwvbMz1EeEy2_yD8znnWK6M0gA%3D%3D/playlist/index.m3u8")
# video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692229287/ei/RwrdZNXWG8jl8wSIvJGADw/ip/68.80.59.169/id/YLSELFy-iHQ.1/itag/95/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D136/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9f8IGX6WdGHMrwP2dOvugGiThjto/vprv/1/playlist_type/DVR/initcwndbps/1553750/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pl/21/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000022/mt/1692207403/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRgIhAMY1Ql3GiRup2qz1wuelvctyGsD2_h0hoFaAxdbQ8AjGAiEA7Xrr-E2yYda5Uxf-QxTaCrOlzKWc4vEFLXZd8xlsKQ8%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIgBZxmsQrfHX3d55J6W7UDsrVqjVUz-yWT6f0riuo2V4ACIQDF0wsBNIsKnoZmfWIXxdAIIpo7Sa-peVlKMiql02bx7Q%3D%3D/playlist/index.m3u8")
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
client = EdgeClient("localhost", 55000)
client.connect()
while video_capture.isOpened() and not stop_button_pressed:
    try:
        return_key, frame = video_capture.read()
        if not return_key:
            print("Video stream could not be read")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        if overlay_button_pressed: #PRED_OVERLAY:
            # Modzy inference & postprocssing       
            _, encoded_frame = cv2.imencode('.jpg', frame)
            input_object = InputSource(
                key="input",
                data=encoded_frame.tobytes()
            )          
            inference = client.inferences.run(MODEL_ID, MODEL_VERSION, [input_object])
            if len(inference.result.outputs.keys()):
                results = json.loads(inference.result.outputs['results.json'].data)
                # frame_stats = [[timestamp, det['class'], det['score']] for det in results['data']['result']['detections']]
                # frame_df = pd.DataFrame(frame_stats, columns=["Timestamp", "Defect", "Confidence Score"])
                # aggregate_frame_df = pd.concat([frame_df, aggregate_frame_df], ignore_index=True, axis=0)
                # Postprocessing
                preds = results['detections']      
            
                if len(preds):
                    for det in preds:
                        box = xywh2xyxy(np.array([det['bounding_box']['x'], det['bounding_box']['y'], det['bounding_box']['width'], det['bounding_box']['height']])).tolist()                     
                        p1, p2 = (box[0], box[1]), (box[2], box[3])
                        label = det['class']
                        color = (255,0,0)
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
                # frame_placeholder.image(image=video_frame, caption="this is so cool!")
                frame_placeholder.video(data=video_frame)
            else:
                continue          
        else:
            # frame_placeholder.image(image=frame, caption="this is so cool!")
            frame_placeholder.video(data=frame)            
        if stop_button_pressed:
            break            
        
    except Exception as e:
        print(traceback.print_exc())
        # print(f"ERROR:\n{e.with_traceback()}")
        break
video_capture.release() 


