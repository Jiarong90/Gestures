import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CSV_RAW = "dataset/landmarks_raw_fixed.csv"
IMG_ROOT = "dataset/images_cropped"

# Parameters to try to tune
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
SEED = 8
PATIENCE = 5
LEARNING_RATE = 4e-3

# Use session_id for group split. Because during training, location and time of day
# were encoded as session_id. For images this is useful because the lighting
# and environment may have played a part in the training, so it may potentially
# cause data leakage without segmenting by session_id
GROUP_COL = "session_id"


def build_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["label", "image_file"]).copy()
    df = df[df["image_file"].notna()].copy()
    df["label"] = df["label"].astype(str)
    df["image_file"] = df["image_file"].astype(str)

    # Keep only rows where the cropped image exists
    def exists_row(r):
        p = os.path.join(IMG_ROOT, r["label"], r["image_file"])
        return os.path.exists(p)

    df = df[df.apply(exists_row, axis=1)].copy()
    # Exclude for_testing data in images, I created that location to specifically
    # collect images outside of backgrounds used during training to verify
    # the accuracy of the model in case it was overfitting or training based on
    # background
    df = df[~df["session_id"].str.contains("for_testing")]
    return df


def split_df(df: pd.DataFrame, group_col: str, test_size=0.2, seed=SEED):
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    groups = df[group_col].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(df)), y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    return df_train, df_test, le

def group_train_val_split(df_train, group_col, val_size=0.2, seed=SEED):
    le_tmp = LabelEncoder()
    y = le_tmp.fit_transform(df_train["label"].to_numpy())
    groups = df_train[group_col].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(np.zeros(len(df_train)), y, groups=groups))
    return df_train.iloc[tr_idx].copy(), df_train.iloc[va_idx].copy()

def load_image(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3) 
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # Normalize pixel values
        img = tf.cast(img, tf.float32) / 255.0
        return img, y

def make_dataset(df: pd.DataFrame, label_encoder: LabelEncoder, training: bool):
    paths = [
        os.path.join(IMG_ROOT, row["label"], row["image_file"])
        for _, row in df.iterrows()
    ]
    labels = label_encoder.transform(df["label"].to_numpy())

    # Load images from disk
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Apply some randomness, to smooth inconsistencies. This helped increase accuracy
        # as finger angle may differ in the images
        rdm = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05), # Tested 0.01, 0.08, 0.2. 0.05 provided best improvement
            tf.keras.layers.RandomTranslation(0.06, 0.06), # Tested 0.12, 0.06, not much differences
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.15),
        ])
        # If training, apply randomness to x and return x and label
        ds = ds.map(lambda x, y: (rdm(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)

    # For faster training
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def main():
    df = build_df(CSV_RAW)
    if len(df) == 0:
        raise RuntimeError("No images found.")

    if GROUP_COL not in df.columns:
        raise ValueError(f"GROUP_COL '{GROUP_COL}' not found..")

    df_train, df_test, le = split_df(df, GROUP_COL, test_size=0.2, seed=SEED)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print("Train sessions:", sorted(df_train["session_id"].unique())[:10], "...")
    print("Test sessions:", sorted(df_test["session_id"].unique())[:10], "...")
    print("Num train sessions:", df_train["session_id"].nunique())
    print("Num test sessions:", df_test["session_id"].nunique())

    print("Classes:", class_names)
    print("Total:", len(df), "Train:", len(df_train), "Test:", len(df_test))
    print("Train counts:", df_train["label"].value_counts().to_dict())
    print("Test counts:", df_test["label"].value_counts().to_dict())

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
    ]

    df_train2, df_val = group_train_val_split(df_train, GROUP_COL, val_size=0.2, seed=SEED)

    test_ds = make_dataset(df_test, le, training=False)

    train_ds2 = make_dataset(df_train2, le, training=True)
    val_ds2   = make_dataset(df_val, le, training=False)

    print("Val sessions:", sorted(df_val["session_id"].unique())[:10], "...")
    print("Num val sessions:", df_val["session_id"].nunique())

    model.fit(train_ds2, validation_data=val_ds2, epochs=EPOCHS, callbacks=callbacks)
    

    # Evaluate on test
    y_true = le.transform(df_test["label"].to_numpy())
    preds = model.predict(test_ds, verbose=0)

    y_pred = preds.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    print("\nCNN TEST RESULTS")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=class_names))
    model.save("final/cnn_model.keras")

if __name__ == "__main__":
    main()
