from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Config
# --------------------------
HANDLE = "florina-lucaciu"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_ITER_LOGREG = 2000

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Nu am găsit fișierul: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"[INFO] Loading data from {path}...")
    
    df = pd.read_csv(path, index_col=0)
    
    # Transpunem matricea: Rows devin Sample-uri, Columns devin Gene
    df = df.T
    print(f"[INFO] Transposed Shape: {df.shape} (Samples x Genes)")

    # Extragem etichetele din index (Sample IDs)
    labels = [sample_id.split('_')[0] for sample_id in df.index]
    y = pd.Series(labels)
    
    X = df
    
    print(f"[INFO] Classes found: {y.unique()}")
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, LogisticRegression, StandardScaler]:
    print("[INFO] Scaling features for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Random Forest (pe date raw)
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)

    # 2. Logistic Regression (pe date scalate)
    print("[INFO] Training Logistic Regression...")
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs", 
        max_iter=MAX_ITER_LOGREG,
        n_jobs=-1,
        class_weight='balanced'
    )
    logreg.fit(X_train_scaled, y_train)

    return rf, logreg, scaler


def compare_models(
    rf: RandomForestClassifier,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_txt: Path,
) -> None:
    print("[INFO] Evaluating models...")
    target_names = [str(c) for c in label_encoder.classes_]

    # --- Evaluare Random Forest ---
    y_pred_rf = rf.predict(X_test)
    report_rf = classification_report(y_test, y_pred_rf, target_names=target_names)

    # --- Evaluare Logistic Regression ---
    X_test_scaled = scaler.transform(X_test)
    y_pred_logreg = logreg.predict(X_test_scaled)
    report_logreg = classification_report(y_test, y_pred_logreg, target_names=target_names)

    # --- Afișare și Salvare ---
    header = f"COMPARISON REPORT: {HANDLE}\n=============================\n"
    
    section_rf = (
        "\n1. RANDOM FOREST CLASSIFIER\n"
        "---------------------------\n"
        f"{report_rf}\n"
    )
    
    section_lr = (
        "\n2. LOGISTIC REGRESSION (Scaled)\n"
        "-----------------------------\n"
        f"{report_logreg}\n"
    )

    full_report = header + section_rf + section_lr
    
    print(full_report)
    out_txt.write_text(full_report)
    print(f"[INFO] Report saved to {out_txt}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DATA_CSV)

    X, y = load_dataset(DATA_CSV)

    y_enc, le = encode_labels(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    rf_model, lr_model, scaler_obj = train_models(X_train, y_train)

    compare_models(rf_model, lr_model, scaler_obj, X_test, y_test, le, OUT_REPORT_TXT)

    print("\n[SUCCESS] Pipeline 8b completed.")