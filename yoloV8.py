# pylint: disable-all

import cv2
from ultralytics import YOLO
import numpy as np
import torch

classArray = []
with open("classes.txt") as file:
    classArray = [line.rstrip() for line in file]

#print(classArray)

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("training_videos/NYC_walking.mp4")
# Starting from frame 600
cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1440, 810))

    # Use GPU instead of CPU -> improve performance ~x10
    results = model(frame, device='1')
    bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.putText(frame, classArray[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
