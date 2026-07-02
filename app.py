import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# CONFIGURATION & THRESHOLDS
# =====================================================================
# Frame threshold: Alert if eyes stay closed for ~2 seconds (approx 30 frames)
CONSECUTIVE_FRAMES_TRIGGER = 30  

# Load OpenCV's built-in Face and Eye detection files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# =====================================================================
# MAIN INITIALIZATION
# =====================================================================
print("[INFO] Loading Pre-trained YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt") 

print("[INFO] Accessing Laptop Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Check if another app is using it.")
    exit()

# State tracking variables
closed_eye_frame_counter = 0
drowsy_alert = False

# =====================================================================
# APPLICATION LOOP
# =====================================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    # Flip horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------------
    # PART 1: OCCUPANT CLASSIFICATION (YOLOv8)
    # -----------------------------------------------------------------
    # Extract the first result cleanly from the list returned by the model
    yolo_results = yolo_model(frame, verbose=False)[0]

    occupant_status = "Empty"
    person_detected = False

    for box in yolo_results.boxes:
        class_id = int(box.cls)
        if class_id == 0:  # Class ID 0 is 'person'
            person_detected = True
            # Extract coordinates safely into Python numbers
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  

    if person_detected:
        occupant_status = "Occupied (Person)"

    # -----------------------------------------------------------------
    # PART 2: DROWSINESS DETECTION (Haar Cascades)
    # -----------------------------------------------------------------
    if person_detected:
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            # Face tracking lost temporarily
            closed_eye_frame_counter = 0
            drowsy_alert = False
        
        for (x, y, w, h) in faces:
            # Isolate the face region to check eyes
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            
            # If a face is found but 0 eyes are detected, the driver's eyes are likely closed/blinking
            if len(eyes) == 0:
                closed_eye_frame_counter += 1
                if closed_eye_frame_counter >= CONSECUTIVE_FRAMES_TRIGGER:
                    drowsy_alert = True
            else:
                closed_eye_frame_counter = 0
                drowsy_alert = False
                
            # Draw visual indicator for face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
    else:
        # Reset tracker if no person sits in front of the camera
        closed_eye_frame_counter = 0
        drowsy_alert = False

    # -----------------------------------------------------------------
    # PART 3: UI OVERLAY & DISPLAY
    # -----------------------------------------------------------------
    # Display Occupant Status
    status_color = (0, 255, 0) if person_detected else (0, 0, 255)
    cv2.putText(frame, f"Occupant: {occupant_status}", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Display Drowsiness Warning Banner if triggered
    if drowsy_alert:
        cv2.rectangle(frame, (20, height - 80), (width - 20, height - 20), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (int(width/4), height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # Render window
    cv2.imshow("Driver Monitor System", frame)

    # Press 'q' to close down safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] System closed safely.")
