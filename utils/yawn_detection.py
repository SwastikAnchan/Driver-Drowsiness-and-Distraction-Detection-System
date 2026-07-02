import math

UPPER_LIP = 13
LOWER_LIP = 14

class YawnDetector:

    def __init__(self):

        self.yawn_count = 0
        self.open_frames = 0

    def distance(self,p1,p2):

        return math.dist(p1,p2)

    def process(self, landmarks, w, h):

        upper = landmarks.landmark[UPPER_LIP]
        lower = landmarks.landmark[LOWER_LIP]

        p1 = (int(upper.x*w), int(upper.y*h))
        p2 = (int(lower.x*w), int(lower.y*h))

        mouth_distance = self.distance(p1,p2)

        if mouth_distance > 25:

            self.open_frames += 1

        else:

            if self.open_frames > 15:
                self.yawn_count += 1

            self.open_frames = 0

        return mouth_distance, self.yawn_count