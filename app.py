import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import os
import platform
import threading
import time

# =====================================================================
# CONFIGURATION, THRESHOLDS & CSV SETUP
# =====================================================================
DROWSY_FRAME_LIMIT = 20
SLEEP_FRAME_LIMIT = 45
YAWN_FRAME_LIMIT = 20
DISTRACTION_FRAME_LIMIT = 30
LOG_FILE = "event_history_log.csv"

# ---- Performance knobs ------------------------------------------------
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 20
YOLO_INFER_SCALE = 0.5
YOLO_SKIP_FRAMES = 1
FACE_DETECT_SCALE = 0.75
MAX_BAD_FRAME_RETRIES = 10
# -----------------------------------------------------------------------

# Non-blocking Audio Alert System
last_beep_time = 0.0
BEEP_COOLDOWN_SEC = 1.5

if platform.system() == "Windows":
    import winsound
    def _sync_beep(freq, dur):
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass
    def alert_beep(frequency, duration):
        global last_beep_time
        now = time.time()
        if now - last_beep_time > max(BEEP_COOLDOWN_SEC, duration / 1000.0):
            last_beep_time = now
            threading.Thread(target=_sync_beep, args=(frequency, duration), daemon=True).start()
else:
    def alert_beep(frequency, duration):
        global last_beep_time
        now = time.time()
        if now - last_beep_time > BEEP_COOLDOWN_SEC:
            last_beep_time = now
            print('\a', end='', flush=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Cabin State", "Identity", "Event Triggered"])

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty() or eye_cascade.empty():
    print("[ERROR] Failed to load Haar cascade classifiers.")
    exit()

print("[INFO] Loading Unified YOLOv8 Engine...")
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO: {e}")
    exit()

# Camera setup
if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera initialization failed.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# State variables
closed_eye_counter = 0
yawn_counter = 0
distraction_counter = 0
consecutive_face_loss = 0
last_logged_event = ""
last_valid_head_pose = "Center Focus"

cached_person_boxes = []
cached_phone_boxes = []
frame_counter = 0
bad_frame_streak = 0

print(f"[INFO] Core System Armed. Press 'q' to quit.")

# =====================================================================
# FULLSCREEN SETUP — Create window ONCE before the loop
# =====================================================================
WINDOW_NAME = "Module 8: Unified Dashboard System"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get actual screen resolution for HUD scaling
# Note: We can't query screen size directly via OpenCV, but we can infer
# from the first frame or use a reasonable default. We'll scale HUD dynamically.
screen_w = 1920  # Default assumption; will adjust on first frame
screen_h = 1080


def log_event_to_csv(cabin, identity, event):
    global last_logged_event
    if event != last_logged_event and event != "System Nominal":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cabin, identity, event])
        last_logged_event = event


def detect_yawn(roi_gray, fw, fh):
    """Detects yawning by analyzing the lower face region."""
    mouth_y_start = int(fh * 0.60)
    mouth_roi_gray = roi_gray[mouth_y_start:, :]
    
    if mouth_roi_gray.size == 0 or fw < 20 or fh < 20:
        return False, 0.0

    blurred = cv2.GaussianBlur(mouth_roi_gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    lower_face_area = fw * (fh * 0.40)
    mouth_ratio = area / lower_face_area if lower_face_area > 0 else 0
    is_yawning = mouth_ratio > 0.30
    
    return is_yawning, mouth_ratio


def run_yolo_scaled(frame, scale):
    """Runs YOLO on downscaled frame for speed."""
    if scale >= 0.999:
        small_frame = frame
        inv_scale = 1.0
    else:
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0 / scale

    try:
        result = yolo_model(small_frame, verbose=False)[0]
    except Exception:
        return [], []

    person_boxes = []
    phone_boxes = []
    if result.boxes is not None:
        for item in result.boxes:
            label = int(item.cls)
            coords = item.xyxy.flatten().tolist()[:4]
            coords = [c * inv_scale for c in coords]
            if label == 0:
                person_boxes.append(coords)
            elif label == 67:
                phone_boxes.append(coords)

    return person_boxes, phone_boxes


def get_face_eye_state(roi_gray, fw, fh):
    """Robust eye state detection."""
    eye_roi = roi_gray[0:int(fh * 0.55), :]
    eyes = eye_cascade.detectMultiScale(eye_roi, 1.1, 4, minSize=(15, 15), maxSize=(fw//2, fh//3))
    
    if len(eyes) >= 1:
        return True, "Open"
    
    if fw < 80 or fh < 80:
        return False, "Uncertain"
    
    left_eye_region = roi_gray[int(fh*0.20):int(fh*0.45), int(fw*0.15):int(fw*0.45)]
    right_eye_region = roi_gray[int(fh*0.20):int(fh*0.45), int(fw*0.55):int(fw*0.85)]
    
    for region in [left_eye_region, right_eye_region]:
        if region.size > 0:
            std = np.std(region)
            if std > 25:
                return True, "Open"
    
    return False, "Closed"


# =====================================================================
# MAIN LOOP
# =====================================================================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            bad_frame_streak += 1
            if bad_frame_streak > MAX_BAD_FRAME_RETRIES:
                print("[ERROR] Camera feed lost. Exiting.")
                break
            continue
        bad_frame_streak = 0

        frame = cv2.flip(frame, 1)
        h_max, w_max, _ = frame.shape
        
        # =====================================================================
        # FULLSCREEN SCALING: Resize camera frame to fill entire screen
        # =====================================================================
        # Get screen dimensions from the window (first frame only)
        if frame_counter == 0:
            # Try to get actual screen size from window properties
            try:
                # On some systems this works; on others it returns (-1, -1)
                rect = cv2.getWindowImageRect(WINDOW_NAME)
                if rect[2] > 0 and rect[3] > 0:
                    screen_w, screen_h = rect[2], rect[3]
            except:
                pass
            print(f"[INFO] Screen resolution detected: {screen_w}x{screen_h}")

        # Scale frame to fill screen while maintaining aspect ratio (letterbox if needed)
        scale_x = screen_w / w_max
        scale_y = screen_h / h_max
        scale = min(scale_x, scale_y)  # Fit within screen, maintain aspect ratio
        
        # For true fullscreen stretch (fills entire screen, may distort):
        # frame = cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
        
        # For aspect-ratio-preserved with black bars:
        new_w = int(w_max * scale)
        new_h = int(h_max * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create black canvas of screen size and center the frame
        display_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        y_offset = (screen_h - new_h) // 2
        x_offset = (screen_w - new_w) // 2
        display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        # Update working dimensions for HUD calculations
        h_max, w_max = screen_h, screen_w
        frame = display_frame  # Work with the fullscreen canvas
        
        # Also need to rescale coordinates for drawing on fullscreen canvas
        # We'll track the offset and scale for coordinate transformation
        coord_scale = scale
        coord_x_offset = x_offset
        coord_y_offset = y_offset
        
        gray_canvas = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Default states
        cabin_occupancy = "Empty Seat"
        driver_classification = "N/A"
        eye_status = "Scanning"
        head_pose = "Center Focus"
        phone_status = "Clean"
        yawn_status = "Normal"
        active_alert = "System Nominal"
        alert_color = (0, 255, 0)
        person_present = False
        phone_present = False

        # ---------------------------------------------------------
        # PIPELINE 1: YOLO Object Detection
        # ---------------------------------------------------------
        if frame_counter % (YOLO_SKIP_FRAMES + 1) == 0:
            # Run YOLO on original-sized frame for accuracy, then scale boxes to fullscreen
            cached_person_boxes, cached_phone_boxes = run_yolo_scaled(frame, YOLO_INFER_SCALE)
        frame_counter += 1

        for (px1, py1, px2, py2) in cached_person_boxes:
            px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
            person_present = True
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

            midpoint = (px1 + px2) / 2
            if midpoint < w_max * 0.55:
                driver_classification = "Driver (Adult)"
                cabin_occupancy = "Driver Active"
            else:
                if driver_classification in ["N/A", "Driver (Adult)"]:
                    driver_classification = "Passenger"
                    cabin_occupancy = "Passenger Present"

        for (cx1, cy1, cx2, cy2) in cached_phone_boxes:
            phone_present = True
            cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

        if phone_present:
            phone_status = "VIOLATION"

        # ---------------------------------------------------------
        # PIPELINE 2: Face Detection & Facial Telemetry
        # ---------------------------------------------------------
        # Run face detection on a region of interest or full frame
        # For fullscreen, we run on the full display frame
        small_gray = cv2.resize(
            gray_canvas, None,
            fx=FACE_DETECT_SCALE, fy=FACE_DETECT_SCALE,
            interpolation=cv2.INTER_LINEAR
        )
        min_size_scaled = max(1, int(60 * FACE_DETECT_SCALE))
        faces_tracked = face_cascade.detectMultiScale(
            small_gray, 1.15, 5, minSize=(min_size_scaled, min_size_scaled)
        )

        inv_face_scale = 1.0 / FACE_DETECT_SCALE
        faces_tracked = [
            (int(x * inv_face_scale), int(y * inv_face_scale),
             int(w * inv_face_scale), int(h * inv_face_scale))
            for (x, y, w, h) in faces_tracked
        ]

        face_detected = len(faces_tracked) > 0

        if face_detected:
            consecutive_face_loss = 0
            faces_tracked = sorted(faces_tracked, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces_tracked[0]
            
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

            face_center = fx + (fw / 2)
            left_thresh = w_max * 0.35
            right_thresh = w_max * 0.65
            
            if face_center < left_thresh:
                head_pose = "Looking Right"
                last_valid_head_pose = "Looking Right"
                distraction_counter += 1
            elif face_center > right_thresh:
                head_pose = "Looking Left"
                last_valid_head_pose = "Looking Left"
                distraction_counter += 1
            else:
                head_pose = "Center Focus"
                last_valid_head_pose = "Center Focus"
                distraction_counter = max(0, distraction_counter - 2)

            roi_gray = gray_canvas[fy:fy + fh, fx:fx + fw]
            eyes_open, eye_status = get_face_eye_state(roi_gray, fw, fh)

            if not eyes_open:
                closed_eye_counter += 1
            else:
                closed_eye_counter = max(0, closed_eye_counter - 3)

            is_yawning, y_ratio = detect_yawn(roi_gray, fw, fh)
            
            yawn_text_y = max(20, fy - 10)
            cv2.putText(frame, f"Y:{y_ratio:.2f}", (fx + fw + 5, yawn_text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if is_yawning:
                yawn_counter += 1
                yawn_status = f"Yawning ({y_ratio:.2f})"
                cv2.putText(frame, "YAWN", (fx, max(25, fy - 25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawn_counter = max(0, yawn_counter - 2)
                yawn_status = f"Normal ({y_ratio:.2f})"

        else:
            consecutive_face_loss += 1
            
            if consecutive_face_loss <= 5:
                head_pose = last_valid_head_pose
                eye_status = "Searching..."
            elif last_valid_head_pose in ["Looking Left", "Looking Right"]:
                head_pose = last_valid_head_pose
                distraction_counter += 1
                closed_eye_counter = max(0, closed_eye_counter - 1)
                eye_status = "Face Turned Away"
            else:
                head_pose = "Face Lost"
                eye_status = "Unknown"
                closed_eye_counter = max(0, closed_eye_counter - 1)

        # ---------------------------------------------------------
        # PIPELINE 3: Alert Arbitration
        # ---------------------------------------------------------
        violations = []

        if closed_eye_counter >= SLEEP_FRAME_LIMIT:
            violations.append(("CRITICAL: DRIVER SLEEP DETECTED", (0, 0, 255), 2200, 400))
        elif phone_present:
            violations.append(("MOBILE PHONE USAGE DETECTED", (0, 0, 255), 1400, 150))
        elif closed_eye_counter >= DROWSY_FRAME_LIMIT:
            violations.append(("WARNING: DROWSINESS DETECTED", (0, 165, 255), 900, 200))
        elif yawn_counter >= YAWN_FRAME_LIMIT:
            violations.append(("WARNING: YAWNING DETECTED", (0, 165, 255), 700, 250))
        elif distraction_counter >= DISTRACTION_FRAME_LIMIT:
            violations.append(("PLEASE FOCUS ON THE ROAD", (0, 165, 255), 550, 200))

        if violations:
            active_alert, alert_color, beep_freq, beep_dur = violations[0]
            alert_beep(beep_freq, beep_dur)
        else:
            active_alert = "System Nominal"
            alert_color = (0, 255, 0)
            last_logged_event = ""

        log_event_to_csv(cabin_occupancy, driver_classification, active_alert)

        # ---------------------------------------------------------
        # PIPELINE 4: HUD Display (Scaled for Fullscreen)
        # ---------------------------------------------------------
        # HUD scales with screen size
        hud_w = min(480, w_max // 3)
        hud_h = min(320, h_max // 3)
        hud_x = 20
        hud_y = 20
        
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (35, 35, 35), cv2.FILLED)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (120, 120, 120), 2)

        metrics = [
            f"Cabin State : {cabin_occupancy}",
            f"Identity    : {driver_classification}",
            f"Eye Status  : {eye_status} [{closed_eye_counter}]",
            f"Head Pose   : {head_pose} [{distraction_counter}]",
            f"Phone Link  : {phone_status}",
            f"Yawn Status : {yawn_status}",
            f"Face Loss   : {consecutive_face_loss}",
        ]

        y_pos = hud_y + 30
        font_scale = 0.55
        line_height = 28
        for metric in metrics:
            cv2.putText(frame, metric, (hud_x + 15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            y_pos += line_height

        # Alert banner at bottom
        banner_h = 70
        banner_y1 = h_max - banner_h
        banner_y2 = h_max - 10
        cv2.rectangle(frame, (10, banner_y1), (w_max - 10, banner_y2), alert_color, cv2.FILLED)

        text_size = cv2.getTextSize(active_alert, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)[0]
        text_x = max(20, (w_max - text_size[0]) // 2)
        text_y = banner_y1 + (banner_y2 - banner_y1 + text_size[1]) // 2
        cv2.putText(frame, active_alert, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)

        # Exit instruction (small, bottom-left corner)
        exit_text = "Press 'q' to exit fullscreen"
        cv2.putText(frame, exit_text, (20, h_max - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Environment offline. Log saved to '{LOG_FILE}'.")
    