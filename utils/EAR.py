import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Eye Landmark Indices
# -----------------------------

# Left eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -----------------------------
# EAR Function
# -----------------------------
def calculate_ear(eye_points):

    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])

    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    # EAR formula
    ear = (v1 + v2) / (2.0 * h)

    return ear

# -----------------------------
# Drowsiness Parameters
# -----------------------------
EAR_THRESHOLD = 0.20
CLOSED_EYES_FRAMES = 20

frame_counter = 0

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # -----------------------------
            # Extract Eye Coordinates
            # -----------------------------
            left_eye_points = []
            right_eye_points = []

            # LEFT EYE
            for idx in LEFT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_eye_points.append([x, y])

                # Draw eye landmarks
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # RIGHT EYE
            for idx in RIGHT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_eye_points.append([x, y])

                # Draw eye landmarks
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Convert to NumPy arrays
            left_eye_points = np.array(left_eye_points)
            right_eye_points = np.array(right_eye_points)

            # -----------------------------
            # Calculate EAR
            # -----------------------------
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            # Display EAR value
            cv2.putText(frame,
                        f'EAR: {ear:.2f}',
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        2)

            # -----------------------------
            # Drowsiness Detection
            # -----------------------------
            if ear < EAR_THRESHOLD:

                frame_counter += 1

                # If eyes closed for long duration
                if frame_counter >= CLOSED_EYES_FRAMES:

                    cv2.putText(frame,
                                'DROWSINESS ALERT!',
                                (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                3)

            else:
                frame_counter = 0

    # Show output
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()