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
# Mouth Landmarks
# =========================

# Upper Lip
UPPER_LIP = 13

# Lower Lip
LOWER_LIP = 14

# =========================
# Yawn Variables
# =========================
yawn_count = 0
yawn_started = False

open_mouth_frames = 0

# Mouth Opening Threshold
MOUTH_THRESHOLD = 25

# Continuous Frames
YAWN_FRAMES = 15

# =========================
# Distance Function
# =========================
def euclidean_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return distance

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
    # Face Detection
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
            # Mouth Points
            # =========================
            upper_lip_point = face_landmarks.landmark[UPPER_LIP]
            lower_lip_point = face_landmarks.landmark[LOWER_LIP]

            upper_lip = (
                int(upper_lip_point.x * w),
                int(upper_lip_point.y * h)
            )

            lower_lip = (
                int(lower_lip_point.x * w),
                int(lower_lip_point.y * h)
            )

            # Draw Mouth Points
            cv2.circle(
                frame,
                upper_lip,
                3,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                lower_lip,
                3,
                (0, 0, 255),
                -1
            )

            # =========================
            # Mouth Opening Distance
            # =========================
            mouth_distance = euclidean_distance(
                upper_lip,
                lower_lip
            )

            # =========================
            # Yawn Logic
            # =========================

            # Mouth Open
            if mouth_distance > MOUTH_THRESHOLD:

                open_mouth_frames += 1

                if not yawn_started:
                    yawn_started = True

            # Mouth Closed
            else:

                if yawn_started:

                    # Count Yawn
                    if open_mouth_frames >= YAWN_FRAMES:
                        yawn_count += 1

                    yawn_started = False

                # Reset Counter
                open_mouth_frames = 0

            # =========================
            # Yawning Detection
            # =========================
            if open_mouth_frames >= YAWN_FRAMES:

                cv2.putText(
                    frame,
                    "YAWNING DETECTED!",
                    (80, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Alarm Sound
                winsound.Beep(1200, 500)

            # =========================
            # Display Yawn Count
            # =========================
            cv2.putText(
                frame,
                f"Yawn Count: {yawn_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # =========================
            # Display Mouth Distance
            # =========================
            cv2.putText(
                frame,
                f"Mouth Dist: {int(mouth_distance)}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # =========================
            # Display Open Frames
            # =========================
            cv2.putText(
                frame,
                f"Open Frames: {open_mouth_frames}",
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
        "Yawn Detection System",
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