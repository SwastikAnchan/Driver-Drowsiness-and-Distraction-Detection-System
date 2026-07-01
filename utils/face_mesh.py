import cv2
import mediapipe as mp
import math
import winsound

# =========================
# Initialize MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Drawing Style
# =========================
drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=1,
    circle_radius=1
)

# =========================
# Eye Landmarks
# =========================

# Left Eye Points
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye Points
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =========================
# Blink Variables
# =========================
blink_count = 0
blink_started = False

# =========================
# Drowsiness Variables
# =========================
closed_eye_frames = 0

# Professional Thresholds
EAR_THRESHOLD = 0.22

# Eye must remain closed
# for 15 continuous frames
DROWSY_FRAMES = 15

# =========================
# Distance Function
# =========================
def euclidean_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

# =========================
# EAR Function
# =========================
def calculate_EAR(eye_points):

    # Vertical distances
    d1 = euclidean_distance(
        eye_points[1],
        eye_points[5]
    )

    d2 = euclidean_distance(
        eye_points[2],
        eye_points[4]
    )

    # Horizontal distance
    d3 = euclidean_distance(
        eye_points[0],
        eye_points[3]
    )

    EAR = (d1 + d2) / (2 * d3)

    return EAR

# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror Effect
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process Frame
    results = face_mesh.process(rgb_frame)

    # =========================
    # Face Detected
    # =========================
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # =========================
            # Draw Face Mesh
            # =========================
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # =========================
            # LEFT EYE
            # =========================
            left_eye_points = []

            for idx in LEFT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x * w
                )

                y = int(
                    face_landmarks.landmark[idx].y * h
                )

                left_eye_points.append((x, y))

                # Draw Points
                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 0, 255),
                    -1
                )

            # =========================
            # RIGHT EYE
            # =========================
            right_eye_points = []

            for idx in RIGHT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x * w
                )

                y = int(
                    face_landmarks.landmark[idx].y * h
                )

                right_eye_points.append((x, y))

                # Draw Points
                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 0, 255),
                    -1
                )

            # =========================
            # Calculate EAR
            # =========================
            left_EAR = calculate_EAR(left_eye_points)

            right_EAR = calculate_EAR(right_eye_points)

            avg_EAR = (
                left_EAR + right_EAR
            ) / 2

            # =========================
            # Blink Logic
            # =========================

            # Eyes Closed
            if avg_EAR < EAR_THRESHOLD:

                closed_eye_frames += 1

                # Blink Start
                if not blink_started:
                    blink_started = True

            # Eyes Open
            else:

                # Count Blink
                if blink_started:
                    blink_count += 1
                    blink_started = False

                # Reset Counter
                closed_eye_frames = 0

            # =========================
            # Drowsiness Detection
            # =========================
            if closed_eye_frames >= DROWSY_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Alarm Sound
                winsound.Beep(1000, 500)

            # =========================
            # Display Blink Count
            # =========================
            cv2.putText(
                frame,
                f"Blink Count: {blink_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # =========================
            # Display EAR
            # =========================
            cv2.putText(
                frame,
                f"EAR: {avg_EAR:.2f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # =========================
            # Display Closed Frames
            # =========================
            cv2.putText(
                frame,
                f"Closed Frames: {closed_eye_frames}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # =========================
    # Show Window
    # =========================
    cv2.imshow(
        "EAR Drowsiness Detection",
        frame
    )

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Release Resources
# =========================
cap.release()
cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Drawing utilities
# mp_drawing = mp.solutions.drawing_utils

# drawing_spec = mp_drawing.DrawingSpec(
#     thickness=1,
#     circle_radius=1
# )

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:

#     # Read frame
#     success, frame = cap.read()

#     if not success:
#         break

#     # Flip frame for mirror view
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame
#     results = face_mesh.process(rgb_frame)

#     # Draw landmarks if face detected
#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec
#             )

#     # Display frame
#     cv2.imshow("Face Mesh", frame)

#     # Press q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # GREEN dots and lines
# drawing_spec = mp_drawing.DrawingSpec(
#     color=(0, 255, 0),   # Green
#     thickness=1,
#     circle_radius=1
# )

# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()

#     if not success:
#         break

#     frame = cv2.flip(frame, 1)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:

#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec
#             )

#     cv2.imshow("Face Mesh", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Drawing utility
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(
#     thickness=1,
#     circle_radius=1
# )

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()

#     if not success:
#         breakq
#     # Flip image for mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame
#     results = face_mesh.process(rgb_frame)

#     # If face detected
#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             # Draw landmarks
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec
#             )

#     # Show frame
#     cv2.imshow("Face Mesh", frame)

#     # Press q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import mediapipe as mp

# # Base options
# base_options = python.BaseOptions(
#     model_asset_path=None
# )

# # Face Landmarker options
# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     output_face_blendshapes=False,
#     output_facial_transformation_matrixes=False,
#     num_faces=1
# )

# # Create landmarker
# detector = vision.FaceLandmarker.create_from_options(options)

# # Webcam
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()

#     if not success:
#         break

#     frame = cv2.flip(frame, 1)

#     cv2.imshow("Webcam", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()