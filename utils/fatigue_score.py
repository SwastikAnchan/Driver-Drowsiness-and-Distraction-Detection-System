# fatigue_score.py

import cv2
import time
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14

NOSE = 1
CHIN = 152

LEFT_FACE = 234
RIGHT_FACE = 454


# =========================================================
# DRIVER STATE CLASSIFIER
# =========================================================

class DriverStateClassifier:

    def __init__(self):

        self.state = "Awake"

    def classify(
        self,
        fatigue_score,
        eye_closure_duration,
        blink_rate,
        yawn_frequency,
        head_down,
        distracted
    ):

        # DISTRACTED

        if distracted:

            self.state = "Distracted"

        # VERY DROWSY

        elif (
            fatigue_score > 60 or
            eye_closure_duration > 1.5 or
            yawn_frequency >= 3
        ):

            self.state = "Very Drowsy"

        # SLEEPY

        elif (
            fatigue_score > 35 or
            blink_rate > 12
        ):

            self.state = "Sleepy"

        # HEAD DOWN

        elif head_down:

            self.state = "Distracted"

        # AWAKE

        else:

            self.state = "Awake"

        return self.state


# =========================================================
# FATIGUE SCORER
# =========================================================

class FatigueScorer:

    def __init__(self):

        # Real-time events

        self.blink_events = deque()

        self.yawn_events = deque()

        # Display counters

        self.blink_count = 0

        self.yawn_count = 0

        # Previous states

        self.prev_eye_closed = False

        # Eye closure timing

        self.eye_closed_start = None

        # Fatigue smoothing

        self.fatigue_score = 0

        # Yawn cooldown

        self.last_yawn_time = 0

        self.yawn_cooldown = 3

    # =====================================================
    # EYE ASPECT RATIO
    # =====================================================

    def eye_aspect_ratio(self, eye):

        A = dist.euclidean(eye[1], eye[5])

        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    # =====================================================
    # NORMALIZE
    # =====================================================

    def normalize(self, value, min_val, max_val):

        value = max(min_val, min(value, max_val))

        return (value - min_val) / (max_val - min_val)

    # =====================================================
    # FATIGUE SCORE
    # =====================================================

    def calculate_score(
        self,
        eye_closure_duration,
        blink_rate,
        mouth_ratio,
        yawn_frequency,
        vertical_ratio,
        eye_slope,
        side_ratio
    ):

        E = self.normalize(
            eye_closure_duration,
            0,
            3
        ) * 100

        B = self.normalize(
            blink_rate,
            0,
            40
        ) * 100

        M = self.normalize(
            mouth_ratio,
            0,
            1
        ) * 100

        Y = self.normalize(
            yawn_frequency,
            0,
            10
        ) * 100

        V = self.normalize(
            vertical_ratio,
            0,
            1
        ) * 100

        S = self.normalize(
            abs(eye_slope),
            0,
            15
        ) * 100

        H = self.normalize(
            side_ratio,
            0,
            1
        ) * 100

        current_score = (
            0.35 * E +
            0.20 * B +
            0.20 * M +
            0.10 * Y +
            0.10 * V +
            0.03 * S +
            0.02 * H
        )

        current_score = max(
            0,
            min(100, current_score)
        )

        # Faster smoothing

        self.fatigue_score = (
            0.65 * self.fatigue_score +
            0.35 * current_score
        )

        return round(self.fatigue_score, 2)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    fatigue_system = FatigueScorer()

    driver_classifier = DriverStateClassifier()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)

                y = int(lm.y * h)

                landmarks.append((x, y))

            # =====================================================
            # EYES
            # =====================================================

            left_eye = [
                landmarks[i]
                for i in LEFT_EYE
            ]

            right_eye = [
                landmarks[i]
                for i in RIGHT_EYE
            ]

            left_ear = (
                fatigue_system.eye_aspect_ratio(
                    left_eye
                )
            )

            right_ear = (
                fatigue_system.eye_aspect_ratio(
                    right_eye
                )
            )

            ear = (
                left_ear + right_ear
            ) / 2.0

            # Draw eyes

            for point in left_eye + right_eye:

                cv2.circle(
                    frame,
                    point,
                    2,
                    (0, 255, 0),
                    -1
                )

            # =====================================================
            # BLINK DETECTION
            # =====================================================

            eyes_closed = ear < 0.16

            current_time = time.time()

            # Blink event

            if (
                eyes_closed and
                not fatigue_system.prev_eye_closed
            ):

                fatigue_system.blink_events.append(
                    current_time
                )

            fatigue_system.prev_eye_closed = eyes_closed

            # =====================================================
            # EYE CLOSURE DURATION
            # =====================================================

            if eyes_closed:

                if (
                    fatigue_system.eye_closed_start
                    is None
                ):

                    fatigue_system.eye_closed_start = (
                        current_time
                    )

                eye_closure_duration = (
                    current_time -
                    fatigue_system.eye_closed_start
                )

            else:

                fatigue_system.eye_closed_start = None

                eye_closure_duration = 0

            # =====================================================
            # YAWN DETECTION
            # =====================================================

            upper_lip = landmarks[UPPER_LIP]

            lower_lip = landmarks[LOWER_LIP]

            mouth_distance = dist.euclidean(
                upper_lip,
                lower_lip
            )

            mouth_ratio = mouth_distance / 50

            # Draw lips

            cv2.circle(
                frame,
                upper_lip,
                3,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                lower_lip,
                3,
                (255, 0, 0),
                -1
            )

            # REAL YAWN DETECTION

            if (
                mouth_ratio > 0.80 and
                current_time -
                fatigue_system.last_yawn_time >
                fatigue_system.yawn_cooldown
            ):

                fatigue_system.yawn_events.append(
                    current_time
                )

                fatigue_system.last_yawn_time = (
                    current_time
                )

            # =====================================================
            # REMOVE OLD EVENTS (60 sec)
            # =====================================================

            while (
                fatigue_system.blink_events and
                current_time -
                fatigue_system.blink_events[0] > 60
            ):

                fatigue_system.blink_events.popleft()

            while (
                fatigue_system.yawn_events and
                current_time -
                fatigue_system.yawn_events[0] > 60
            ):

                fatigue_system.yawn_events.popleft()

            # =====================================================
            # REAL-TIME RATES
            # =====================================================

            blink_rate = len(
                fatigue_system.blink_events
            )

            yawn_frequency = len(
                fatigue_system.yawn_events
            )

            fatigue_system.blink_count = blink_rate

            fatigue_system.yawn_count = yawn_frequency

            # =====================================================
            # HEAD DOWN
            # =====================================================

            nose = landmarks[NOSE]

            chin = landmarks[CHIN]

            vertical_ratio = (
                abs(chin[1] - nose[1]) / h
            )

            head_down = vertical_ratio > 0.30

            # =====================================================
            # SIDE DISTRACTION
            # =====================================================

            left_face = landmarks[LEFT_FACE]

            right_face = landmarks[RIGHT_FACE]

            side_ratio = abs(
                (nose[0] - left_face[0]) -
                (right_face[0] - nose[0])
            ) / w

            distracted = side_ratio > 0.15

            # =====================================================
            # EYE SLOPE
            # =====================================================

            eye_slope = abs(
                left_eye[0][1] -
                right_eye[0][1]
            )

            # =====================================================
            # FATIGUE SCORE
            # =====================================================

            fatigue = (
                fatigue_system.calculate_score(
                    eye_closure_duration,
                    blink_rate,
                    mouth_ratio,
                    yawn_frequency,
                    vertical_ratio,
                    eye_slope,
                    side_ratio
                )
            )

            # =====================================================
            # DRIVER STATE
            # =====================================================

            driver_state = (
                driver_classifier.classify(
                    fatigue,
                    eye_closure_duration,
                    blink_rate,
                    yawn_frequency,
                    head_down,
                    distracted
                )
            )

            # =====================================================
            # COLORS
            # =====================================================

            state_color = (0, 255, 0)

            if driver_state == "Sleepy":

                state_color = (0, 255, 255)

            elif driver_state == "Very Drowsy":

                state_color = (0, 0, 255)

            elif driver_state == "Distracted":

                state_color = (255, 0, 255)

            # =====================================================
            # UI
            # =====================================================

            cv2.putText(
                frame,
                f"Fatigue: {fatigue:.1f}%",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"State: {driver_state}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                state_color,
                2
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Blinks(1m): {blink_rate}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yawns(1m): {yawn_frequency}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Mouth Ratio: {mouth_ratio:.2f}",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Eye Closure: {eye_closure_duration:.1f}s",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # =====================================================
            # ALERT
            # =====================================================

            if driver_state == "Very Drowsy":

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        cv2.imshow(
            "Fatigue Detection",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()