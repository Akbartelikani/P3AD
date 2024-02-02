#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np

def extract_and_process_periods(video_path, period_length):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    total_frames = 0

    while True:
        period_frames_list = []
        for _ in range(period_length):
            ret, frame = cap.read()
            if not ret:
                break
            period_frames_list.append(frame)
            total_frames += 1

        if not period_frames_list:
            break

        frames.append(period_frames_list)

    return frames


# In[ ]:


import os
from PIL import Image
import numpy as np
import Perturbation_Generation as PG
import Pixel_set_selection as PSS
import Frame_Selection as FS
import P3AD     

if __name__ == "__main__":
    video_path = "session_1.mp4"
    target_fps = 29
    period_length = 30
    
    extracted_frames = extract_and_process_periods(video_path, period_length)
    P3AD(extracted_frames)
    

