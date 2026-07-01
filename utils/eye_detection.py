import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# ----------------------------
# MediaPipe Face Mesh
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

# ----------------------------
# Eye Landmark Indices
# ----------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------------------
# EAR Calculation Function
# ----------------------------

def calculate_EAR(eye_points, landmarks):

    # Horizontal points
    left_point = landmarks[eye_points[0]]
    right_point = landmarks[eye_points[3]]

    # Vertical points
    top_point1 = landmarks[eye_points[1]]
    bottom_point1 = landmarks[eye_points[5]]

    top_point2 = landmarks[eye_points[2]]
    bottom_point2 = landmarks[eye_points[4]]

    # Distances
    horizontal_length = hypot(
        left_point[0] - right_point[0],
        left_point[1] - right_point[1]
    )

    vertical_length1 = hypot(
        top_point1[0] - bottom_point1[0],
        top_point1[1] - bottom_point1[1]
    )

    vertical_length2 = hypot(
        top_point2[0] - bottom_point2[0],
        top_point2[1] - bottom_point2[1]
    )

    # EAR Formula
    ear = (vertical_length1 + vertical_length2) / (2.0 * horizontal_length)

    return ear

# ----------------------------
# Start Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    frame_height, frame_width = frame.shape[:2]

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # Store all landmarks
            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)

                landmarks.append((x, y))

            # Calculate EAR
            left_ear = calculate_EAR(LEFT_EYE, landmarks)
            right_ear = calculate_EAR(RIGHT_EYE, landmarks)

            avg_ear = (left_ear + right_ear) / 2

            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Detect Eye Closure
            if avg_ear < 0.22:

                cv2.putText(
                    frame,
                    "Eyes Closed",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Eyes Open",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    # Show frame
    cv2.imshow("Eye Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()