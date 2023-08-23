import os
import json
import pandas as pd
import numpy as np
import streamlit as st

# read COCO labels used for YOLOv8 model
ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "names.json"), "r") as file:
    labels = json.load(file)
NAMES = [v for _,v in labels.items()]

# performance metrics
def update_performance_table(inf, non_inf):
    inference_mean = inf[["Frame Read Time", "Inference Time", "Postprocess Time", "Total"]].mean(axis=0)
    print(inference_mean)
    inference_mean = round(inference_mean*1000, 2)
    no_inference_mean = non_inf[["Frame Read Time", "Inference Time", "Postprocess Time", "Total"]].mean(axis=0)
    no_inference_mean = round(no_inference_mean*1000, 2)
    summary = pd.DataFrame(pd.concat([inference_mean, no_inference_mean], axis=1, ignore_index=True))
    summ_t = summary.transpose()
    summ_t["Type"] = ["Inference", "No Inference"]
    summ_t = summ_t[["Type", "Frame Read Time", "Inference Time", "Postprocess Time", "Total"]]
    summ_t["FPS"] = 1000 / summ_t["Total"]
    return summ_t

# chart data
def format_table(df):
    # confidence df
    filtered_df = df[["Timestamp", "Object", "Confidence Score"]]
    confidence_df = filtered_df.pivot(columns="Object", values="Confidence Score")   
    # counts df 
    counts = filtered_df.groupby(["Timestamp", "Object"])["Confidence Score"].count()
    counts_df = counts.reset_index()
    counts_df = counts_df.rename(columns={"Confidence Score": "Counts"})
    counts_df = counts_df.pivot(columns="Object", values="Counts")
    return confidence_df, counts_df

# update analysis statistics
def update_analysis(df):
    # create total column
    df["Total"] = df.iloc[:, 3:6].sum(axis=1)
    # split into inference & non-inference
    no_inference = df[df["Object"] == "No overlay"]
    inference = df[df["Object"] != "No overlay"]   
    # transform inference data into digestible format
    conf_data, counts_data = format_table(inference) 
    # summarize inference and non-inference tables into summary statistics
    performance = update_performance_table(inference, no_inference)
    return conf_data, counts_data, performance

# convert bounding boxes from xywh to xyxy format
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# create grid
def make_grid(rows,cols,gap="small"):
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols, gap=gap)
    return grid

# colors for postprocessing
class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
                '753156', "5A28AB", 'A2FF76', "CCD840", "B496B4", "98DD41", "3FAEBE", "C13D54", "124D93", "C2E99D",
                "5232F7", '0DE276', "43CE20", "106EA9", "5588CD", "0CA65C", 'F99769', '0071C7', '866291', '147E51',
                '8E0144', 'E4A501', '816B33', '9D6A54', '341227', '390193', 'E001A7', '81850A', '374F50', '27DD49',
                '61E6A9', '9F2598', '11AAA0', '465C6A', '2AB1B4', '3BAF48', 'F1CF62', '0F466D', 'A8F891', '8F3FDF',
                '953DF9', "3E8101", '2E39F6', 'FA1806', '6F6AEB', '0A1099', '8B8953', '6347F2', 'B7A588', '060C8E',
                "F45CDF", '5C8B79', '5043F6', '39AEC0', '3982B0', 'AE7136', '442E58', '577167', "4CDFD4", '839508')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

