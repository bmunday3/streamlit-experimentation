import cv2
import json
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
poses = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

image = cv2.imread("multi-people.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start = time.time()
pose_landmarks = poses.process(image)
end = time.time()
print(f"Processed frame in {round(((end-start)/60)*1000, 6)} ms")

# print(dir(pose_landmarks))
# print(pose_landmarks.pose_landmarks, type(pose_landmarks.pose_landmarks))
print(dir(pose_landmarks))

if pose_landmarks.pose_landmarks:
    mp_drawing.draw_landmarks(
      image,
      pose_landmarks.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
cv2.imwrite("poses-out-multi.jpg", image)    
        
        
        