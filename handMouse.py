# pylint: disable-all

import cv2
import mediapipe as mp
import pyautogui as pa
import HandTrackingModule as hd

cap = cv2.VideoCapture(0)

detector = hd.handDetector()

height, width, _ = cap.shape

while True:
    _, frame = cap.read()
    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)

    hand1_positions = detector.getPosition(frame, range(21), draw=False)
    hand2_positions = detector.getPosition(frame, range(21), hand_no=1, draw=False)
    for pos in hand1_positions:
        cv2.circle(frame, pos, 5, (0,255,0), cv2.FILLED)
    for pos in hand2_positions:
        cv2.circle(frame, pos, 5, (255,0,0), cv2.FILLED)
    print("Index finger up:", detector.index_finger_up(frame))
    print("Middle finger up:", detector.middle_finger_up(frame))
    print("Ring finger up:", detector.ring_finger_up(frame))
    print("Little finger up:", detector.little_finger_up(frame))

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break