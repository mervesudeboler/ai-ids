"""
AI-Powered Intrusion Detection System (AI-IDS)
===============================================
Real-time network threat detection using a Random Forest classifier
trained on KDD-style synthetic traffic data.

Features
--------
- Synthetic dataset generation (KDD Cup-style features)
- Random Forest with 97%+ accuracy on test split
- Live packet simulation with configurable attack injection
- Per-packet real-time alerting with confidence scores
- Attack type classification: DoS/DDoS, Port Scan, Brute-Force
- Timestamped CSV alert log
- Model persistence via joblib

Author  : Merve Sude Böler
GitHub  : https://github.com/mervesudeboler
License : MIT
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ids package integration (optional — used when running as part of the package)
try:
    from ids import PacketCapture, FeatureExtractor, IDSModel, AlertManager
    _IDS_PACKAGE_AVAILABLE = True
except ImportError:
    _IDS_PACKAGE_AVAILABLE = False

# ------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
ATTACK_LABELS = {
    0: "Normal",
    1: "DoS/DDoS",
    2: "Port Scan",
    3: "Brute-Force/R2L",
}

FEATURE_NAMES = [
    "packet_size",
    "duration",
    "src_port",
    "dst_port",
    "protocol",
    "flag_syn",
    "flag_ack",
    "flag_rst",
    "flag_fin",
    "bytes_per_sec",
    "packets_per_sec",
]


# ------------------------------------------------------------------
#  Data Classes
# ------------------------------------------------------------------
@dataclass
class PacketRecord:
    timestamp: str
    src_ip: str
    dst_ip: str
    features: list
    label: Optional[int] = None
    confidence: float = 0.0
    attack_type: str = "-"


@dataclass
class IDSConfig:
    model_path: str = "outputs/ids_model.pkl"
    scaler_path: str = "outputs/ids_scaler.pkl"
    alert_log: str = "outputs/alerts.csv"
    output_dir: str = "outputs"
    n_samples: int = 5000
    n_packets: int = 100
    interval: float = 0.03
    threshold: float = 0.70
    attack_ratio: float = 0.15
    quiet: bool = False


# ------------------------------------------------------------------
#  Dataset Generator
# ------------------------------------------------------------------
class DatasetGenerator:
    """Generates a synthetic KDD Cup-style network flow dataset."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def _normal_flow(self) -> list:
        rng = self.rng
        return [
            int(rng.integers(64, 1500)),
            float(rng.uniform(0.001, 2.0)),
            int(rng.integers(1024, 65535)),
            int(rng.choice([80, 443, 22, 8080])),
            int(rng.choice([0, 1])),
            int(rng.choice([0, 1], p=[0.3, 0.7])),
            int(rng.choice([0, 1], p=[0.2, 0.8])),
            0,
            int(rng.choice([0, 1], p=[0.9, 0.1])),
            float(rng.uniform(100, 5000)),
            float(rng.uniform(1, 50)),
        ]

    def _dos_flow(self) -> list:
        rng = self.rng
        return [
            int(rng.integers(40, 100)),
            float(rng.uniform(0.0001, 0.01)),
            int(rng.integers(1024, 65535)),
            int(rng.choice([80, 443])),
            2,
            1, 0, 0, 0,
            float(rng.uniform(50000, 200000)),
            float(rng.uniform(500, 5000)),
        ]

    def _portscan_flow(self) -> list:
        rng = self.rng
        return [
            int(rng.integers(40, 60)),
            float(rng.uniform(0.0001, 0.005)),
            int(rng.integers(1024, 65535)),
            int(rng.integers(1, 1024)),
            0,
            1, 0, 0, 0,
            float(rng.uniform(10, 200)),
            float(rng.uniform(100, 1000)),
        ]

    def _bruteforce_flow(self) -> list:
        rng = self.rng
        return [
            int(rng.integers(100, 300)),
            float(rng.uniform(0.01, 0.5)),
            int(rng.integers(1024, 65535)),
            int(rng.choice([22, 21, 23, 3389])),
            0,
            1, 1, 0, 0,
            float(rng.uniform(500, 3000)),
            float(rng.uniform(10, 100)),
        ]

    def generate(self, n_samples: int):
        n_normal   = int(n_samples * 0.60)
        n_each_atk = (n_samples - n_normal) // 3

        rows, labels = [], []
        for _ in range(n_normal):
            rows.append(self._normal_flow());    labels.append(0)
        for _ in range(n_each_atk):
            rows.append(self._dos_flow());       labels.append(1)
        for _ in range(n_each_atk):
            rows.append(self._portscan_flow());  labels.append(2)
        for _ in range(n_samples - n_normal - 2 * n_each_atk):
            rows.append(self._bruteforce_flow()); labels.append(3)

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int64)
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]


# ------------------------------------------------------------------
#  IDS Engine
# ------------------------------------------------------------------
class IDSEngine:
    """Core AI-IDS engine: training, persistence, and inference.

    When the ``ids`` package is available, training and inference are
    delegated to :class:`ids.IDSModel` so that the package and the CLI
    share a single model implementation.  When the package is not on the
    path (e.g. running main.py in isolation) the engine falls back to its
    own inline Random Forest + StandardScaler pipeline.
    """

    def __init__(self, config: IDSConfig) -> None:
        self.config  = config
        self.model   = None
        self.scaler  = None
        # ids.IDSModel delegate (used when ids package is importable)
        self._ids_model: Optional["IDSModel"] = (
            IDSModel(model_path=config.model_path) if _IDS_PACKAGE_AVAILABLE else None
        )
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self) -> None:
        logger.info("Generating %d synthetic flows ...", self.config.n_samples)
        gen = DatasetGenerator()
        X, y = gen.generate(self.config.n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_test_s    = self.scaler.transform(X_test)

        logger.info("Training Random Forest (100 estimators) ...")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        acc    = (y_pred == y_test).mean()
        logger.info("Test accuracy: %.2f%%", acc * 100)

        if not self.config.quiet:
            print("\n" + classification_report(
                y_test, y_pred, target_names=list(ATTACK_LABELS.values())
            ))

        self._save()

    def _save(self) -> None:
        joblib.dump(self.model,  self.config.model_path)
        joblib.dump(self.scaler, self.config.scaler_path)
        logger.info("Model saved  -> %s", self.config.model_path)
        logger.info("Scaler saved -> %s", self.config.scaler_path)

    def load(self) -> None:
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.config.model_path}'. "
                "Run with --train first."
            )
        if not os.path.isfile(self.config.scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at '{self.config.scaler_path}'. "
                "Run with --train first."
            )
        self.model  = joblib.load(self.config.model_path)
        self.scaler = joblib.load(self.config.scaler_path)
        logger.info("Model loaded from %s", self.config.model_path)

    def predict(self, features: list):
        # Delegate to ids.IDSModel when available
        if _IDS_PACKAGE_AVAILABLE and self._ids_model is not None and self._ids_model.is_trained():
            label_str, confidence = self._ids_model.predict(np.array(features, dtype=np.float32))
            label_idx = 0 if label_str == "normal" else 1
            return label_idx, confidence
        # Fallback: inline pipeline
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        x          = self.scaler.transform([features])
        proba      = self.model.predict_proba(x)[0]
        label_idx  = int(np.argmax(proba))
        confidence = float(proba[label_idx])
        return label_idx, confidence


# ------------------------------------------------------------------
#  Packet Simulator
# ------------------------------------------------------------------
class PacketSimulator:
    """Simulates a realistic stream of network packets with injected attacks."""

    _NORMAL_IPS = [f"192.168.1.{i}" for i in range(2, 50)]
    _ATTACK_IPS = [f"172.16.0.{i}"  for i in range(100, 130)]
    _DST_IPS    = [f"10.0.0.{i}"    for i in range(1, 10)]

    def __init__(self, attack_ratio: float = 0.15) -> None:
        self.attack_ratio = attack_ratio
        self._gen = DatasetGenerator(seed=int(time.time()))

    def next_packet(self) -> PacketRecord:
        is_attack = random.random() < self.attack_ratio
        if is_attack:
            attack_type = random.randint(1, 3)
            generators  = {
                1: self._gen._dos_flow,
                2: self._gen._portscan_flow,
                3: self._gen._bruteforce_flow,
            }
            features = generators[attack_type]()
            src_ip   = random.choice(self._ATTACK_IPS)
        else:
            attack_type = 0
            features    = self._gen._normal_flow()
            src_ip      = random.choice(self._NORMAL_IPS)

        return PacketRecord(
            timestamp  = datetime.now().strftime("%H:%M:%S.%f")[:-4],
            src_ip     = src_ip,
            dst_ip     = random.choice(self._DST_IPS),
            features   = features,
            label      = attack_type,
        )


# ------------------------------------------------------------------
#  Alert Logger
# ------------------------------------------------------------------
class AlertLogger:
    """Writes attack alerts to a timestamped CSV file."""

    HEADER = ["timestamp", "src_ip", "dst_ip", "attack_type", "confidence"]

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(self.HEADER)

    def log(self, pkt: PacketRecord) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                pkt.timestamp, pkt.src_ip, pkt.dst_ip,
                pkt.attack_type, f"{pkt.confidence:.4f}",
            ])


# ------------------------------------------------------------------
#  Live Monitor
# ------------------------------------------------------------------
class LiveMonitor:
    """Runs the live packet monitoring loop."""

    W = 65

    def __init__(self, engine: IDSEngine, config: IDSConfig) -> None:
        self.engine  = engine
        self.config  = config
        self.sim     = PacketSimulator(attack_ratio=config.attack_ratio)
        self.al      = AlertLogger(config.alert_log)

    def _header(self) -> None:
        print("\n" + "=" * self.W)
        print("  AI-IDS -- Live Packet Monitor (simulated traffic)")
        print("=" * self.W)
        print(f"  {'TIME':<13} {'SRC IP':<17} {'DST IP':<14} {'LABEL':<10} {'CONF':>6}  TYPE")
        print("-" * self.W)

    def _row(self, pkt: PacketRecord) -> str:
        label_str = "[ATTACK]" if pkt.label != 0 else "[NORMAL]"
        color     = "\033[91m" if pkt.label != 0 else "\033[92m"
        reset     = "\033[0m"
        return (
            f"  {pkt.timestamp:<13} "
            f"{pkt.src_ip:<17} "
            f"{pkt.dst_ip:<14} "
            f"{color}{label_str:<10}{reset} "
            f"{pkt.confidence:>5.1%}  "
            f"{pkt.attack_type}"
        )

    def run(self) -> None:
        if not self.config.quiet:
            self._header()

        total_packets = 0
        total_attacks = 0

        try:
            for _ in range(self.config.n_packets):
                pkt = self.sim.next_packet()
                label_idx, conf = self.engine.predict(pkt.features)

                pkt.label       = label_idx
                pkt.confidence  = conf
                pkt.attack_type = ATTACK_LABELS[label_idx] if label_idx != 0 else "-"

                total_packets += 1
                is_flagged = label_idx != 0 and conf >= self.config.threshold

                if is_flagged:
                    total_attacks += 1
                    self.al.log(pkt)

                if not self.config.quiet:
                    print(self._row(pkt))

                time.sleep(self.config.interval)

        except KeyboardInterrupt:
            print("\n  [Interrupted by user]")

        print("-" * self.W)
        print(f"  Total packets  : {total_packets}")
        print(f"  Attacks flagged: {total_attacks} ({total_attacks/max(total_packets,1):.1%})")
        print(f"  Alert log      : {self.config.alert_log}")
        print("=" * self.W + "\n")


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-ids",
        description="AI-Powered Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --train\n"
            "  python main.py --train --simulate --packets 200\n"
            "  python main.py --simulate --model outputs/ids_model.pkl\n"
        ),
    )
    p.add_argument("--train",     action="store_true",
                   help="Train the RF model on synthetic data")
    p.add_argument("--simulate",  action="store_true",
                   help="Run the live packet monitor")
    p.add_argument("--packets",   type=int,   default=100,  metavar="N",
                   help="Number of packets to simulate (default: 100)")
    p.add_argument("--interval",  type=float, default=0.03, metavar="S",
                   help="Delay between packets in seconds (default: 0.03)")
    p.add_argument("--model",     type=str,   default="outputs/ids_model.pkl",
                   metavar="PATH", help="Model save/load path")
    p.add_argument("--threshold", type=float, default=0.70, metavar="F",
                   help="Min confidence to flag as attack (default: 0.70)")
    p.add_argument("--samples",   type=int,   default=5000, metavar="N",
                   help="Training dataset size (default: 5000)")
    p.add_argument("--quiet",     action="store_true",
                   help="Suppress per-packet output; show summary only")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args   = parser.parse_args()

    if not args.train and not args.simulate:
        parser.print_help()
        sys.exit(0)

    config = IDSConfig(
        model_path  = args.model,
        scaler_path = args.model.replace(".pkl", "_scaler.pkl"),
        n_samples   = args.samples,
        n_packets   = args.packets,
        interval    = args.interval,
        threshold   = args.threshold,
        quiet       = args.quiet,
    )

    engine = IDSEngine(config)

    try:
        if args.train:
            engine.train()

        if args.simulate:
            if not args.train:
                engine.load()
            monitor = LiveMonitor(engine, config)
            monitor.run()

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
