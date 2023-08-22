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
    confidence_df = df[["Timestamp", "Object", "Confidence Score"]]
    confidence_df = df.pivot(columns="Object", values="Confidence Score")    
    return confidence_df, None

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