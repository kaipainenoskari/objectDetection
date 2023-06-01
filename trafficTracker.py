# pylint: disable-all

import cv2
from tracker import EuclideanDistTracker

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("training_videos/highway3.mp4")
#cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    #print(frame.shape) 

    roi = frame[250:720, 100:1180]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 300 and area < 3000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, idOfBox = box_id
        cv2.putText(roi, str(idOfBox), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(20)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
