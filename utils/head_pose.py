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
# Important Landmarks
# =========================

# Nose Tip
NOSE_TIP = 1

# Left Face
LEFT_FACE = 234

# Right Face
RIGHT_FACE = 454

# Forehead
FOREHEAD = 10

# Chin
CHIN = 152

# =========================
# Detection Counters
# =========================
down_frames = 0
side_frames = 0

# =========================
# Thresholds
# =========================

# Side face turning sensitivity
SIDE_THRESHOLD = 60

# Looking down sensitivity
DOWN_THRESHOLD = 20

# Continuous frames required

# Side distraction frames
SIDE_ALERT_FRAMES = 40

# Looking down frames
DOWN_ALERT_FRAMES = 25

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
            # Landmark Coordinates
            # =========================

            nose = face_landmarks.landmark[NOSE_TIP]
            left_face = face_landmarks.landmark[LEFT_FACE]
            right_face = face_landmarks.landmark[RIGHT_FACE]
            forehead = face_landmarks.landmark[FOREHEAD]
            chin = face_landmarks.landmark[CHIN]

            # Convert to Pixel Coordinates
            nose_point = (
                int(nose.x * w),
                int(nose.y * h)
            )

            left_point = (
                int(left_face.x * w),
                int(left_face.y * h)
            )

            right_point = (
                int(right_face.x * w),
                int(right_face.y * h)
            )

            forehead_point = (
                int(forehead.x * w),
                int(forehead.y * h)
            )

            chin_point = (
                int(chin.x * w),
                int(chin.y * h)
            )

            # =========================
            # Draw Key Points
            # =========================
            cv2.circle(frame, nose_point, 4, (0, 0, 255), -1)
            cv2.circle(frame, left_point, 4, (255, 0, 0), -1)
            cv2.circle(frame, right_point, 4, (255, 0, 0), -1)
            cv2.circle(frame, forehead_point, 4, (0, 255, 255), -1)
            cv2.circle(frame, chin_point, 4, (0, 255, 255), -1)

            # =========================
            # Side Face Detection
            # =========================

            left_distance = euclidean_distance(
                nose_point,
                left_point
            )

            right_distance = euclidean_distance(
                nose_point,
                right_point
            )

            # Difference between left and right
            side_difference = abs(
                left_distance - right_distance
            )

            # Display Side Difference
            cv2.putText(
                frame,
                f"Side Diff: {int(side_difference)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Side Detection Logic
            if side_difference > SIDE_THRESHOLD:

                side_frames += 1

            else:

                side_frames = 0

            # =========================
            # Looking Down Detection
            # =========================

            # Face Center
            face_center_y = (
                forehead_point[1] + chin_point[1]
            ) // 2

            # Nose shifted downward
            down_difference = (
                nose_point[1] - face_center_y
            )

            # Display Down Difference
            cv2.putText(
                frame,
                f"Down Diff: {int(down_difference)}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Looking Down Logic
            if down_difference > DOWN_THRESHOLD:

                down_frames += 1

            else:

                down_frames = 0

            # =========================
            # Sideways Alert
            # =========================
            if side_frames >= SIDE_ALERT_FRAMES:

                cv2.putText(
                    frame,
                    "DISTRACTION DETECTED!",
                    (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                winsound.Beep(1200, 500)

            # =========================
            # Looking Down Alert
            # =========================
            if down_frames >= DOWN_ALERT_FRAMES:

                cv2.putText(
                    frame,
                    "LOOKING DOWN!",
                    (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                winsound.Beep(1000, 500)

            # =========================
            # Display Frame Counters
            # =========================
            cv2.putText(
                frame,
                f"Side Frames: {side_frames}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Down Frames: {down_frames}",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =========================
    # Show Window
    # =========================
    cv2.imshow(
        "Attention Monitoring System",
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