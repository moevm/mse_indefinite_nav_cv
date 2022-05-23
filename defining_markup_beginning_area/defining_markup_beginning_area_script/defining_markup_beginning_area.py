import cv2
import numpy as np

# Process each video frame, note how "history" is passed around
#path = "20160429223859_neptunus.video.mp4"
path = "20160406233412_tesla.video.mp4"
frames_list = []
video = cv2.VideoCapture(path)
color_1 = (255, 0, 0)
color_2 = (0, 255, 0)
color_3 = (0, 0, 255)
thickness = 1
area_1 = [(35, 130), (150, 250)]
area_2 = [(200, 150), (375, 200)]
area_3 = [(475, 175), (638, 330)]

while True:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frame = cv2.rectangle(frame, area_1[0], area_1[1], color_1, thickness)
    frame = cv2.rectangle(frame, area_2[0], area_2[1], color_2, thickness)
    frame = cv2.rectangle(frame, area_3[0], area_3[1], color_3, thickness)
    cv2.imshow('image', frame)

    crop_area_1 = frame[130:251, 35:151]
    crop_area_2 = frame[150:201, 200:376]
    crop_area_3 = frame[175:331, 475:639]
    cv2.imshow('crop_area_1', crop_area_1)
    cv2.imshow('crop_area_2', crop_area_2)
    cv2.imshow('crop_area_3', crop_area_3)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
