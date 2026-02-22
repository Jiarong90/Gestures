import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
from pynput.keyboard import Controller, Key
import socket

# Taken from my previous Cybersecurity module. Just a simple UDP setup
# to send current predicted gesture to game window so label displays correctly
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ("127.0.0.1", 5055)

MODEL_PATH = "cnn_model.keras"
CLASSES_PATH = "cnn_classes.json"

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)
print("Loaded class_names:", class_names)

# Preprocess config, use same configurations as from preprocess.py
IMG_SIZE = 128
PADDING = 0.35        
MIN_BOX = 80           
CONF_THRESHOLD = 0.70  
COOLDOWN = 0.15

keyboard = Controller()

# Set key mapping
KEY_MAP_HOLD = {
    "point_left": "a",
    "point_right": "d",
}

KEY_MAP_TAP = {
    "point_up": "w",
    "fist": "space",
    "palm": "p"
}

# Preprocess image same as in preprocess.py 
def pad_to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=img.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img
    return canvas

def bbox_from_landmarks(lms, w, h, padding=0.35, min_box=80):
    # Notice we changed padding to 0.35 to match your preprocess.py
    xs = [lm.x * w for lm in lms.landmark]
    ys = [lm.y * h for lm in lms.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(bw, min_box)
    bh = max(bh, min_box)

    x1 = cx - bw / 2
    x2 = cx + bw / 2
    y1 = cy - bh / 2
    y2 = cy + bh / 2

    x1 -= bw * padding
    x2 += bw * padding
    y1 -= bh * padding
    y2 += bh * padding

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w, x2))
    y2 = int(min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def preprocess_crop(frame_bgr, box):
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    
    # --- THE COLOR FIX: Convert BGR to RGB before resizing ---
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # ---------------------------------------------------------
    
    crop = pad_to_square(crop)
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    return crop

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)


    current_label = "none"
    last_switch_time = 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found.")


    last_time = time.time()
    fps = 0.0

    # Tap and hold keys, hold is for movement, tap is for actions like attack
    held_key = None
    tap_active = None  
    tap_until = 0.0
    TAP_HOLD = 0.12   

    # Hold previous prediction to check if gesture has changed
    prev_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        label = "no_hand"
        conf = 0.0
        box = None

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            box = bbox_from_landmarks(lms, w, h)
            if box is not None:
                crop = preprocess_crop(frame, box)
                x = np.expand_dims(crop, axis=0)  # (1,128,128,3)
                probs = model.predict(x, verbose=0)[0]
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])
                raw_label = class_names[pred_id]

                now = time.time()

                if conf >= CONF_THRESHOLD:
                    if (now - last_switch_time) > COOLDOWN:
                        current_label = raw_label
                        last_switch_time = now

                label = current_label

                # Send label to game window
                sock.sendto(label.encode("utf-8"), DEST)

                # Release tapped key when time is up
                if tap_active is not None and now >= tap_until:
                    if tap_active == "space":
                        keyboard.release(Key.space)
                    else:
                        keyboard.release(tap_active)
                    tap_active = None

                # New tap when the label changes
                if label != prev_label and tap_active is None:
                    tap_key = KEY_MAP_TAP.get(label)
                    if tap_key is not None:
                        if tap_key == "space":
                            keyboard.press(Key.space)
                        else:
                            keyboard.press(tap_key)
                        tap_active = tap_key
                        tap_until = now + TAP_HOLD

                prev_label = label

                # KEY MAPPING now
                now = time.time()

                # Hold keys
                desired = KEY_MAP_HOLD.get(label)  # "a" / "d" / None

                # If movement changed 
                if desired != held_key:
                    # release old
                    if held_key is not None:
                        keyboard.release(held_key)
                    held_key = desired
                    # Press new
                    if held_key is not None:
                        keyboard.press(held_key)
                

        # Reset if no hand detected
        if not res.multi_hand_landmarks:
            label = "no_hand"
            
            sock.sendto(label.encode("utf-8"), DEST)

            if held_key is not None:
                keyboard.release(held_key)
                held_key = None
            prev_label = "no_hand"
            if tap_active is not None:
                if tap_active == "space":
                    keyboard.release(Key.space)
                else:
                    keyboard.release(tap_active)
                tap_active = None
        
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # Draw bbox and text
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Pred: {label}  conf:{conf:.2f}  fps:{fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, "Press q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow("CNN Live Demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    if held_key is not None:
        keyboard.release(held_key)
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
