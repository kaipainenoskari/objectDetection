# pylint: disable-all
import cv2
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = frame.shape

    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            location_data = detection.location_data
            bb = location_data.relative_bounding_box

            x = int(bb.xmin * width)
            y = int(bb.ymin * height)
            w = int(bb.width * width)
            h = int(bb.height * height)

            bb_box = [
                x, y,
                w, h
            ]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print(f"FACEBOX: {bb_box}")

    #x, y, w, h = results.detections[0].location_data.relative_bounding_box

    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
