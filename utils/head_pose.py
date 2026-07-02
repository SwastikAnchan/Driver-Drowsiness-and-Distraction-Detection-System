LEFT_FACE = 234
RIGHT_FACE = 454
NOSE = 1
CHIN = 152

class HeadPoseDetector:

    def process(self, landmarks, w, h):

        nose = landmarks.landmark[NOSE]
        left = landmarks.landmark[LEFT_FACE]
        right = landmarks.landmark[RIGHT_FACE]
        chin = landmarks.landmark[CHIN]

        nose_x = int(nose.x*w)
        left_x = int(left.x*w)
        right_x = int(right.x*w)

        side_diff = abs(
            (nose_x-left_x) -
            (right_x-nose_x)
        )

        vertical_ratio = abs(
            chin.y - nose.y
        )

        distracted = side_diff > 60

        return distracted, side_diff, vertical_ratio