import os
import cv2
import pandas as pd
from datetime import datetime

LOG_DIR = 'logs'
CSV_PATH = os.path.join(LOG_DIR, 'violations.csv')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize CSV if not exists
def init_log():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=['timestamp', 'violation_type', 'image_path'])
        df.to_csv(CSV_PATH, index=False)

# Log a violation: save image and append to CSV
def log_violation(frame, violation_type):
    init_log()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'violation_{timestamp}.jpg'
    image_path = os.path.join(LOG_DIR, image_name)
    cv2.imwrite(image_path, frame)
    df = pd.DataFrame([[timestamp, violation_type, image_path]], columns=['timestamp', 'violation_type', 'image_path'])
    df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    print(f'Violation logged: {violation_type} at {timestamp}') 