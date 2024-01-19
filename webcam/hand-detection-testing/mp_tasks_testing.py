import cv2
import json
import time
import base64
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    mp.solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      mp.solutions.pose.POSE_CONNECTIONS,
      mp.solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image
        
        
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=50
  )
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file("multi-people.jpg")
# image = mp.Image.create_from_file("yoga.jpg")
image = cv2.cvtColor(cv2.imread("yoga.jpg"), cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

# STEP 4: Detect pose landmarks from the input image.
start = time.time()
detection_result = detector.detect(mp_image)
end = time.time()
print(detection_result)
print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")

# STEP 4B: serialize outputs
test = detection_result.pose_landmarks[0][0]
output_preds = []
for landmarks in detection_result.pose_landmarks:
  pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  pose_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
  ])
  pose_landmarks_proto.SerializeToString()
  output_preds.append(pose_landmarks_proto)
  
# print(output_preds, len(output_preds))
  # print(landmarks)
# out = [[base64.b64encode(landmark.SerializeToString()).decode() for landmark in landmarks] for landmarks in results.pose_landmarks] if results.pose_landmarks else []

import sys
sys.exit()

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
cv2.imwrite("test-out-tasks-yoga-cv2.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))        