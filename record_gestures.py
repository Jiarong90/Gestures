# Import libraries
import cv2
import mediapipe as mp
import os, time
import csv
import numpy as np
import pandas as pd

# Session tagging - label where and when data is captured to formulate the
# session_id, for better splitting 
SESSION_LOCATIONS = {
    "w": "br_desk",
    "e": "br_bed",
    "r": "lr_sofa",
    "t": "lr_desk",
    "y": "others",
    "u": "for_testing",
}
SESSION_TIMES = {
    "d": "day",
    "n": "night",
}

# Define gestures
# Map key to gesture
GESTURES = {"1": "fist", "2": "point_left", "3": "point_right", "4": "point_up", "5": "palm"}
# Define frames captured per capture. 
# I used 2 as images can be blurry during capture due to hand motion
CAPTURE_FRAMES = 2
# Save raw image if True, settings is present because initially  
# only MediaPipe landmarks were used. Afterwards CNN was introduced
SAVE_EVERY_FRAME_IMAGE = True
IMG_EXT = "jpg"
JPG_QUALITY = 85

OUT_DIR = "dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
CSV_PATH = os.path.join(OUT_DIR, "landmarks_raw.csv")
CSV_NORM = os.path.join(OUT_DIR, "landmarks_norm.csv")

# Create directories for each gesture
def init_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    for gesture in set(GESTURES.values()):
        os.makedirs(os.path.join(IMG_DIR, gesture), exist_ok=True)

# Create CSV with headers
def init_csv():
    header = ["label", "image_file", "session_id", "burst_id"]
    # MediaPipe uses 21 landmarks, save x, y, z position of each landmark
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    for path in [CSV_PATH, CSV_NORM]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

def extract_vec(hand_landmarks):
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array(coords, dtype=np.float32)

# Normalize hand position so gesture can be detected regardless of hand position
# during recording
def normalize_vec(vec): 
    # Reshape the flattened vector back into 21 hand landmarks
    points = vec.reshape(21, 3).copy()
    # Use wrist as origin point
    wrist = points[0]
    points -= wrist
    # Scale based on palm size
    scale = np.linalg.norm(points[9])
    if scale > 1e-6:
        points /= scale
    return points.reshape(-1)

def save_row(csv_path, label, image_file, session_id, burst_id, vec):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, image_file, session_id, burst_id] + vec.tolist())

# To show text during capture, for visuals only so it is easier to record gestures
def put_text(frame, lines):
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y+=28

def main():

    init_dirs()
    init_csv()

    current_loc = "lr_desk"
    current_time = "night"

    # Taken from MediaPipe docs
    # To show the MediaPipe Landmarks during recording
    # Then save those Landmarks into a CSV file
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open video capture for the gesture recording
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found.")
    
    current_label = "fist"
    saved_counts = {g: 0 for g in set(GESTURES.values())}
    msg = "Press C for capture. Press Q to quit."
    jpg_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Check if hand detected
            # Display the Landmarks on the hand for visuals
            hand_detected = False
            hand_landmarks = None

            if res.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = res.multi_hand_landmarks[0]
  
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Wait for key press and record the key that was pressed
            key = cv2.waitKey(1) & 0xFF
            kch = chr(key) if 0 <= key <= 255 else ""

            # Quit
            if key == ord("q"):
                break

            # Switch gestures
            if kch in GESTURES:
                current_label = GESTURES[kch]
                msg = f"Selected: {current_label}"

            # Switch session location
            if kch in SESSION_LOCATIONS:
                current_loc = SESSION_LOCATIONS[kch]
                msg = f"Session location: {current_loc}"

            # Switch session time
            if kch in SESSION_TIMES:
                current_time = SESSION_TIMES[kch]
                msg = f"Session time: {current_time}"

            # Create the session id
            session_id = f"{current_loc}_{current_time}"
            
            lines = [
                f"Label: {current_label}",
                f"Session: {session_id}",
                f"Hand: {'OK' if hand_detected else 'NO HAND DETECTED'}",
                f"Saved: " + " | ".join([f"{k}: {saved_counts[k]}" for k in saved_counts]),
                msg 
            ]
            put_text(frame, lines)
            cv2.imshow("MediaPipe Hands (press q)", frame)

            # Capture gesture
            if key == ord("c"):
                burst_id = f"{session_id}_{current_label}_{int(time.time()*1000)}"
                if not hand_detected:
                    msg = "No hand detected."
                    continue
                cv2.waitKey(1)
                time.sleep(0.25)
                for _ in range(CAPTURE_FRAMES):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    frame2 = cv2.flip(frame2, 1)
                    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    res2 = hands.process(rgb2)

                    if not res2.multi_hand_landmarks:
                        continue

                    # Take from 1 hand only. Due to time constraints we limit to 1 hand
                    # (right-hand) for now
                    hand_landmarks2 = res2.multi_hand_landmarks[0]
                    vec = extract_vec(hand_landmarks2)
                    vec_norm = normalize_vec(vec)
                    image_file = ""

                    # Save file as raw image for future processing
                    if SAVE_EVERY_FRAME_IMAGE:
                        timestamp = int(time.time() * 1000)
                        image_file = f"{session_id}_{current_label}_{timestamp}.{IMG_EXT}"
                        out_path = os.path.join(IMG_DIR, current_label, image_file)
                        write_status = cv2.imwrite(out_path, frame2, jpg_param)
                        if not write_status:
                            print("Failed to save image")

                    save_row(CSV_PATH, current_label, image_file, session_id, burst_id, vec)
                    save_row(CSV_NORM, current_label, image_file, session_id, burst_id, vec_norm)
                    saved_counts[current_label] += 1
                    time.sleep(0.08)

                msg = f"Done. Total {current_label}: {saved_counts[current_label]}"

    cap.release()
    cv2.destroyAllWindows()
    print("Saved")

if __name__ == "__main__":
    main()
