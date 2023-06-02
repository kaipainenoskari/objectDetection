# pylint: disable-all
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if ret is not True:
        break

    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_image)

    for facial_landmark in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmark.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)

    key = cv2.waitKey(5)

    cv2.imshow("Frame", image)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
