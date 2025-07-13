import cv2
from ultralytics import YOLO
import os
from utils import log_violation

# Path to YOLOv8 weights
MODEL_PATH = os.path.join('yolov8', 'yolov8n.pt')

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()

print('Press "q" to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to capture frame.')
        break

    # Save latest frame for dashboard live feed
    cv2.imwrite(os.path.join('logs', 'latest.jpg'), frame)

    # Run YOLOv8 inference on the frame
    results = model(frame)
    boxes = results[0].boxes
    names = results[0].names
    classes = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else []

    # Find persons and helmets
    person_indices = [i for i, c in enumerate(classes) if names[c] == 'person']
    helmet_indices = [i for i, c in enumerate(classes) if names[c] == 'helmet']

    # For each person, check if a helmet overlaps
    for i in person_indices:
        px1, py1, px2, py2 = xyxy[i]
        has_helmet = False
        for j in helmet_indices:
            hx1, hy1, hx2, hy2 = xyxy[j]
            # Check if helmet box overlaps with person box
            if not (hx2 < px1 or hx1 > px2 or hy2 < py1 or hy1 > py2):
                has_helmet = True
                break
        if not has_helmet:
            log_violation(frame, 'No Helmet')
            # Draw red box around person without helmet
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0,0,255), 2)
            cv2.putText(frame, 'No Helmet', (int(px1), int(py1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 