# pylint: disable-all
import cv2
import mediapipe as mp
from faceLandmarks import FaceLandmarks
import numpy as np

# Load helpers
fl = FaceLandmarks()
cap = cv2.VideoCapture(0)

while True:
    # Get frame
    ret, frame = cap.read()
    frame_copy = frame.copy()

    height, width, _ = frame.shape

    # 1. Face landmarks detection
    landmarks = fl.get_facial_landmarks(frame)
    # If no face is visible, show original frame
    if landmarks.size == 0:
        result = frame
    else:
        #print(landmarks)
        convexhull = cv2.convexHull(landmarks)

        # 2. Face blurrying
        mask = np.zeros((height, width), np.uint8)
        #cv2.polylines(mask, [convexhull], True, (0, 255, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        # Extract face
        frame_copy = cv2.blur(frame_copy, (27, 27))
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
        #blurred_face = cv2.GaussianBlur(face_extracted, (27, 27), 0)

        # Extract background
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Final blurred result
        result = cv2.add(background, face_extracted)

    cv2.imshow("Blurred", result)
    cv2.imshow("Original", frame)

    # Press Esc to end run and close all windows
    key = cv2.waitKey(5)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
