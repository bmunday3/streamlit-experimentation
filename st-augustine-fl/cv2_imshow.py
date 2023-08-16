import cv2
import json
import traceback
from utils import xywh2xyxy
from modzy import EdgeClient
from modzy.edge import InputSource

STOP = False
OVERLAY = False

# modzy config
MODEL_ID = "ccsvw2eofe"
MODEL_VERSION = "0.0.1"


video_capture = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1692205776/ei/cK7cZLGRKo6N_9EPj62w4AI/ip/2601:43:200:3750:586d:7795:39dc:2ed6/id/YLSELFy-iHQ.1/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/rr2---sn-jvhj5nu-2iae.googlevideo.com/playlist_duration/30/manifest_duration/30/spc/UWF9fzn5X4zyjhNlJl9XjDZ_rHyLkE8/vprv/1/playlist_type/DVR/initcwndbps/1581250/mh/BT/mm/44/mn/sn-jvhj5nu-2iae/ms/lva/mv/m/mvi/2/pl/34/dover/11/pacing/0/keepalive/yes/fexp/24007246,51000023/mt/1692183885/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,spc,vprv,playlist_type/sig/AOq0QJ8wRQIhALlq2mrRJGV4ypuKSekO7bbqZ1Dt_fGVvA82snYytn-tAiBaYmDawuhwCOT97TSYMmQhuWX0n7fi9Gw1AUIfrdj1Cg%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRgIhANaV7x-32g06_fGUGYhRcbCn23zh6SmF9rG9lRCfm9VMAiEAnoHhw0UcDONf6OGy4C_7yN7WvfmSytCbajJpud7D_40%3D/playlist/index.m3u8")
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
client = EdgeClient("localhost", 55000)
client.connect()
while video_capture.isOpened() and not STOP:
    try:
        return_key, frame = video_capture.read()
        if not return_key:
            print("Video stream could not be read")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        if OVERLAY: #PRED_OVERLAY:
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
                cv2.imshow("frame", video_frame)
            else:
                continue          
        else:
            cv2.imshow("frame", frame)
        if STOP:
            break            
        
    except Exception as e:
        print(traceback.print_exc())
        # print(f"ERROR:\n{e.with_traceback()}")
        break
video_capture.release() 