from utils.EAR import calculate_ear

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

EAR_THRESHOLD = 0.22

class EyeDetector:

    def __init__(self):
        self.closed_frames = 0
        self.blink_count = 0
        self.eye_closed = False

    def process(self, landmarks, w, h):

        points = []

        for lm in landmarks.landmark:
            points.append(
                (int(lm.x*w), int(lm.y*h))
            )

        left_eye = [points[i] for i in LEFT_EYE]
        right_eye = [points[i] for i in RIGHT_EYE]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:

            self.closed_frames += 1

            if not self.eye_closed:
                self.eye_closed = True

        else:

            if self.eye_closed:
                self.blink_count += 1

            self.eye_closed = False
            self.closed_frames = 0

        return ear, self.blink_count, self.closed_frames