"""
ids.model -- Random Forest classifier for intrusion detection
"""

import os
import logging
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

ATTACK_TYPES = {
    "normal":  "Normal Traffic",
    "dos":     "DoS / DDoS Attack",
    "probe":   "Port Scan / Probe",
    "r2l":     "Remote-to-Local Attack",
    "u2r":     "Privilege Escalation",
    "unknown": "Unknown Threat",
}

FEATURE_COLS = [
    "packet_size", "is_tcp", "is_udp", "is_icmp",
    "src_port", "dst_port",
    "tcp_flags_syn", "tcp_flags_rst", "tcp_flags_fin", "tcp_flags_ack",
    "payload_size", "ttl", "ip_frag", "header_length",
]


class IDSModel:
    """
    Wraps a scikit-learn RandomForestClassifier with persistence and
    a simple predict interface for the IDS pipeline.

    Parameters
    ----------
    model_path   : path to save/load the pickled model
    n_estimators : number of trees in the forest
    """

    def __init__(self, model_path: str = "models/ids_model.pkl",
                 n_estimators: int = 100) -> None:
        self.model_path = model_path
        self.n_estimators = n_estimators
        self._clf: RandomForestClassifier | None = None
        self._label_enc = LabelEncoder()
        self._attack_type_clf: RandomForestClassifier | None = None
        self._load_if_exists()

    # -- Persistence ---------------------------------------------------------

    def _load_if_exists(self) -> None:
        if os.path.isfile(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self._clf             = data["clf"]
                self._attack_type_clf = data.get("attack_type_clf")
                self._label_enc       = data["label_enc"]
                logger.info("Model loaded from %s", self.model_path)
            except Exception as exc:
                logger.warning("Could not load model: %s", exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "clf":             self._clf,
                "attack_type_clf": self._attack_type_clf,
                "label_enc":       self._label_enc,
            }, f)
        logger.info("Model saved to %s", self.model_path)

    def is_trained(self) -> bool:
        return self._clf is not None

    # -- Training ------------------------------------------------------------

    def train(self, dataset_path: str) -> None:
        """
        Train the classifier on a CSV dataset.

        Expected columns: all FEATURE_COLS + 'label' (normal/attack)
        Optional column : 'attack_type' (dos/probe/r2l/u2r)
        """
        logger.info("Loading dataset: %s", dataset_path)
        df = pd.read_csv(dataset_path)

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")
        if "label" not in df.columns:
            raise ValueError("Dataset must have a 'label' column (normal/attack).")

        X = df[FEATURE_COLS].fillna(0).values
        y_raw = df["label"].str.lower().values
        y = self._label_enc.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Training Random Forest (%d estimators)...", self.n_estimators)
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        self._clf.fit(X_train, y_train)

        y_pred = self._clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Accuracy: %.2f%%", acc * 100)
        print("\n" + classification_report(
            y_test, y_pred,
            target_names=self._label_enc.classes_
        ))

        if "attack_type" in df.columns:
            self._train_attack_type(df)

        self._save()

    def train_default(self) -> None:
        """Train on synthetically generated sample data (no dataset needed)."""
        logger.info("Generating synthetic training data...")
        X, y_binary, y_type = _generate_synthetic_data(n_samples=5000)

        self._label_enc.fit(["normal", "attack"])
        self._clf = RandomForestClassifier(
            n_estimators=50, max_depth=8, n_jobs=-1, random_state=42
        )
        self._clf.fit(X, y_binary)

        self._attack_type_clf = RandomForestClassifier(
            n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
        )
        self._attack_type_clf.fit(X, y_type)

        self._save()
        logger.info("Default model trained on synthetic data.")

    def _train_attack_type(self, df: pd.DataFrame) -> None:
        X = df[FEATURE_COLS].fillna(0).values
        y = df["attack_type"].str.lower().fillna("unknown").values
        self._attack_type_clf = RandomForestClassifier(
            n_estimators=self.n_estimators // 2,
            max_depth=8, n_jobs=-1, random_state=42,
        )
        self._attack_type_clf.fit(X, y)
        logger.info("Attack-type sub-classifier trained.")

    # -- Inference -----------------------------------------------------------

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict whether a feature vector is an attack.

        Returns
        -------
        (label, confidence) : ("normal"|"attack", 0.0-1.0)
        """
        if self._clf is None:
            return "normal", 0.0
        x     = features.reshape(1, -1)
        proba = self._clf.predict_proba(x)[0]
        idx   = int(np.argmax(proba))
        label = self._label_enc.inverse_transform([idx])[0]
        return label, float(proba[idx])

    def predict_type(self, features: np.ndarray) -> str:
        """Predict the specific attack type."""
        if self._attack_type_clf is None:
            return "unknown"
        x = features.reshape(1, -1)
        return str(self._attack_type_clf.predict(x)[0])


# -- Synthetic data generator ------------------------------------------------

def _generate_synthetic_data(n_samples: int = 5000):
    """Generate synthetic labeled packet features for default training."""
    rng  = np.random.default_rng(42)
    half = n_samples // 2

    normal = np.column_stack([
        rng.integers(40, 1500, half),
        rng.integers(0, 2, half),
        rng.integers(0, 2, half),
        rng.integers(0, 2, half),
        rng.integers(1024, 65535, half),
        rng.integers(80, 443, half),
        np.zeros(half), np.zeros(half), np.zeros(half), np.ones(half),
        rng.integers(0, 1400, half),
        rng.integers(48, 128, half),
        np.zeros(half),
        rng.integers(20, 60, half),
    ]).astype(np.float32)

    attack = np.column_stack([
        rng.integers(40, 80, half),
        np.ones(half), np.zeros(half), np.zeros(half),
        rng.integers(1, 1024, half),
        rng.integers(1, 65535, half),
        np.ones(half),
        rng.integers(0, 2, half),
        np.zeros(half), np.zeros(half),
        np.zeros(half),
        rng.integers(1, 32, half),
        rng.integers(0, 2, half),
        rng.integers(20, 40, half),
    ]).astype(np.float32)

    X        = np.vstack([normal, attack])
    y_binary = np.array([0] * half + [1] * half)
    y_type   = np.array(
        ["normal"] * half +
        ["dos"]    * (half // 2) +
        ["probe"]  * (half - half // 2)
    )

    idx = rng.permutation(n_samples)
    return X[idx], y_binary[idx], y_type[idx]
