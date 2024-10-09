from collections import defaultdict
import cv2
import numpy as np
#from ultralytics import YOLO
from tensorflow.keras.models import load_model

#def track_video(video_path):
model = load_model("model.h5")