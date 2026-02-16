import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_NORM = "dataset/landmarks_norm.csv"  
MODEL_PATH = "final/svm_landmarks_norm.joblib"
ENC_PATH   = "final/label_encoder.joblib"

def main():
    df = pd.read_csv(CSV_NORM)

    # ONLY for_testing
    df = df[df["session_id"].astype(str).str.contains("for_testing")]
    if len(df) == 0:
        print("No for_testing rows found in CSV. Did you record with location=for_testing?")
        return

    feature_cols = []
    for i in range(21):
        feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

    X = df[feature_cols].astype(np.float32).to_numpy()
    y_true = df["label"].astype(str).to_numpy()

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENC_PATH)

    y_pred_id = model.predict(X)
    y_pred = le.inverse_transform(y_pred_id)

    acc = accuracy_score(y_true, y_pred)
    print("\n=== SVM for_testing results ===")
    print("samples:", len(y_true))
    print("accuracy:", acc)
    labels = sorted(set(y_true) | set(y_pred))
    print("confusion matrix labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

if __name__ == "__main__":
    main()
