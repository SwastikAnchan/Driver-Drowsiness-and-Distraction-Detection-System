import mediapipe as mp

class FaceMeshDetector:

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame_rgb):

        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]

        return None