# pylint: disable-all

import cv2
import mediapipe as mp
import numpy

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]

cap = cv2.VideoCapture(0)

face = mp.solutions.face_mesh
Face = face.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

while True:
	_, frame = cap.read()

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	results = Face.process(rgb)

	if results.multi_face_landmarks:
		mesh_coords = land

	cv2.imshow("window", frame)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break