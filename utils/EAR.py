import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    v1 = euclidean(eye_points[1], eye_points[5])
    v2 = euclidean(eye_points[2], eye_points[4])
    h = euclidean(eye_points[0], eye_points[3])

    if h == 0:
        return 0

    return (v1 + v2) / (2.0 * h)