# pylint: disable-all
import mediapipe as mp
import cv2
import numpy as np

# Helper class for getting face landmarks

class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        # If no face is visible, return empty array
        if result.multi_face_landmarks == None:
            return np.array([], np.int32)

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)
