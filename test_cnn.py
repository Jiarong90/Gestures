import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_RAW  = "dataset/landmarks_raw_fixed.csv"
IMG_ROOT = "dataset/images_cropped"
MODEL_PATH = "final/cnn_model.keras"
CLASSES_PATH = "final/cnn_classes.json"
IMG_SIZE = 128
BATCH_SIZE = 32

def load_image(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, y


def main():
    df = pd.read_csv(CSV_RAW)
    df = df.drop_duplicates(subset=["label", "image_file"]).copy()
    df = df[df["image_file"].notna()].copy()
    df["label"] = df["label"].astype(str)
    df["image_file"] = df["image_file"].astype(str)

    # ONLY for_testing
    df = df[df["session_id"].astype(str).str.contains("for_testing")]
    if len(df) == 0:
        print("No for_testing rows found in CSV.")
        return

    # keep only rows where cropped image exists
    def exists_row(r):
        p = os.path.join(IMG_ROOT, r["label"], r["image_file"])
        return os.path.exists(p)

    df = df[df.apply(exists_row, axis=1)].copy()
    if len(df) == 0:
        print("No cropped for_testing images found in images_cropped.")
        return

    # load training class order
    if not os.path.exists(CLASSES_PATH):
        print("Missing", CLASSES_PATH, "- save le.classes_ from training first.")
        return

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
    label_to_id = {c:i for i,c in enumerate(class_names)}
    paths = [os.path.join(IMG_ROOT, r["label"], r["image_file"]) for _, r in df.iterrows()]
    y_true = np.array([label_to_id[x] for x in df["label"].to_numpy()], dtype=np.int32)

    # dataset
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    y_ds = tf.data.Dataset.from_tensor_slices(y_true)
    ds = tf.data.Dataset.zip((path_ds, y_ds)).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("Loading CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    preds = model.predict(ds, verbose=0)
    y_pred = preds.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== CNN for_testing results ===")
    print("samples:", len(y_true))
    print("accuracy:", acc)
    print("class_names:", class_names)
    print("confusion matrix:\n", cm)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
