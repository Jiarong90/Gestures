import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import joblib

MODEL_PATH = "final/svm_landmarks_norm.joblib"
ENC_PATH   = "final/label_encoder.joblib"

MODEL_PATH2 = "final/mlp_landmarks_norm.joblib"
ENC_PATH2   = "final/mlp_label_encoder.joblib"

CSV_RAW = "dataset/landmarks_raw_fixed.csv"
CSV_NORM = "dataset/landmarks_norm.csv"

def load_data(path: str):
    df = pd.read_csv(path)
    # Exclude any row with for_testing. for_testing location was created specifically to
    # record gestures in new places not used for training data, it is used to double
    # check that no overfitting occurred 
    df = df[~df["session_id"].str.contains("for_testing")]
    feature_cols = []
    for i in range(21):
        feature_cols += [f"x{i}", f"y{i}", f"z{i}"]
    if len(feature_cols) != 63:
        raise ValueError("Expected 63 features")
    
    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["label"].astype(str).to_numpy()
    # Group by burst_id. Because burst_id groups frames captured together
    # so the Landmarks may be identical. So if data with the same burst_id is
    # separated into training/testing it may be considered data leakage
    groups = df["burst_id"].astype(str).to_numpy()
    
    return X, y, groups

def group_split(X, y, groups, test_size=0.2, seed=8):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx

def eval_model(name, model, X_train, y_train, X_test, y_test, class_names):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=np.arange(len(class_names)))

    print(name)
    print(f"Accuracy: {accuracy}")
    print(cm)
    print(classification_report(y_test, pred, target_names=class_names))
    return accuracy

def run_experiment(title, X, y_raw, groups):
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y_raw)
    class_names = list(labelEncoder.classes_)

    train_idx, test_idx = group_split(X, y, groups, test_size=0.2, seed=8)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n\n{title}")
    print("Total samples:", len(y))
    print("Train samples:", len(train_idx), "Test samples:", len(test_idx))
    print("Train class counts:", {class_names[i]: int(np.sum(y_train == i)) for i in range(len(class_names))})
    print("Test class counts:", {class_names[i]: int(np.sum(y_test == i)) for i in range(len(class_names))})

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=5.0, gamma="scale"))
    ])

    svm_acc = eval_model("SVM", svm, X_train, y_train, X_test, y_test, class_names)

    # MLP Architecture Testing
    architectures = [
    (32,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (64, 64),
    (128, 64, 32),
    ]

    best_acc = 0
    best_model = None
    best_arch = None

    for arch in architectures:
        mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=arch,
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=8
            ))
        ])

        acc = eval_model(f"MLP {arch}", mlp, X_train, y_train, X_test, y_test, class_names)

        if acc > best_acc:
            best_acc = acc
            best_model = mlp
            best_arch = arch

    print("\nBest MLP architecture:", best_arch, "Accuracy:", best_acc)
    
    if "Normalized" in title:
        joblib.dump(svm, MODEL_PATH)
        joblib.dump(best_model, MODEL_PATH2)
        joblib.dump(labelEncoder, ENC_PATH)
        print("Saved models to:")
        print(" -", MODEL_PATH)
        print(" -", MODEL_PATH2)
        print(" -", ENC_PATH)

def main():
    X_raw, y_raw, groups_raw = load_data(CSV_RAW)
    X_norm, y_norm, groups_norm = load_data(CSV_NORM)

    run_experiment("Raw landmarks", X_raw, y_raw, groups_raw)
    run_experiment("Normalized landmarks", X_norm, y_norm, groups_norm)

if __name__ == "__main__":
    main()

    