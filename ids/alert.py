"""
ids.alert — AlertManager: structured alert handling and CSV logging

Wraps the raw CSV writing from main.py's AlertLogger into a richer
AlertManager class that supports severity levels, in-memory history,
and optional console callbacks.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Severity thresholds mapped to confidence ranges
SEVERITY_LEVELS = {
    "LOW":      (0.70, 0.84),
    "MEDIUM":   (0.85, 0.94),
    "HIGH":     (0.95, 1.00),
}

CSV_HEADER = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "attack_type",
    "confidence",
    "severity",
]


@dataclass
class Alert:
    """
    Represents a single IDS alert event.

    Attributes
    ----------
    timestamp   : HH:MM:SS.ff string
    src_ip      : source IP address
    dst_ip      : destination IP address
    attack_type : human-readable attack label (e.g. 'DoS/DDoS')
    confidence  : model confidence score in [0, 1]
    severity    : 'LOW' | 'MEDIUM' | 'HIGH'
    """
    timestamp:   str
    src_ip:      str
    dst_ip:      str
    attack_type: str
    confidence:  float
    severity:    str = field(init=False)

    def __post_init__(self) -> None:
        self.severity = Alert._classify_severity(self.confidence)

    @staticmethod
    def _classify_severity(confidence: float) -> str:
        for level, (lo, hi) in SEVERITY_LEVELS.items():
            if lo <= confidence <= hi:
                return level
        return "HIGH"  # anything above the table range is critical

    def to_row(self) -> list:
        return [
            self.timestamp,
            self.src_ip,
            self.dst_ip,
            self.attack_type,
            f"{self.confidence:.4f}",
            self.severity,
        ]

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.timestamp}  "
            f"{self.src_ip} -> {self.dst_ip}  "
            f"{self.attack_type} ({self.confidence:.1%})"
        )


class AlertManager:
    """
    Manages IDS alerts: logging to CSV, in-memory history, and callbacks.

    Parameters
    ----------
    log_path    : path to the CSV alert log file
    threshold   : minimum confidence to generate an alert (default 0.70)
    on_alert    : optional callback invoked with each new Alert object
    max_history : max alerts kept in memory (0 = unlimited)

    Usage
    -----
    >>> am = AlertManager("outputs/alerts.csv")
    >>> am.process(
    ...     src_ip="172.16.0.1", dst_ip="10.0.0.2",
    ...     attack_type="Port Scan", confidence=0.93
    ... )
    """

    def __init__(
        self,
        log_path:    str = "outputs/alerts.csv",
        threshold:   float = 0.70,
        on_alert:    Optional[Callable[[Alert], None]] = None,
        max_history: int = 1000,
    ) -> None:
        self.log_path    = log_path
        self.threshold   = threshold
        self.on_alert    = on_alert
        self.max_history = max_history
        self._history:   List[Alert] = []

        self._init_log()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def process(
        self,
        src_ip:      str,
        dst_ip:      str,
        attack_type: str,
        confidence:  float,
        timestamp:   Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Evaluate a detection result and generate an Alert if it exceeds
        the confidence threshold.

        Parameters
        ----------
        src_ip      : source IP string
        dst_ip      : destination IP string
        attack_type : label string (e.g. 'DoS/DDoS')
        confidence  : model confidence score
        timestamp   : override timestamp; defaults to current time

        Returns
        -------
        Alert object if an alert was generated, else None
        """
        if confidence < self.threshold:
            return None

        ts = timestamp or datetime.now().strftime("%H:%M:%S.%f")[:-4]
        alert = Alert(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            attack_type=attack_type,
            confidence=confidence,
        )

        self._write_row(alert)
        self._store(alert)

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as exc:
                logger.warning("on_alert callback raised: %s", exc)

        logger.warning("ALERT  %s", alert)
        return alert

    # ------------------------------------------------------------------
    #  History helpers
    # ------------------------------------------------------------------
    @property
    def history(self) -> List[Alert]:
        """Return a copy of the in-memory alert history."""
        return list(self._history)

    def total_alerts(self) -> int:
        return len(self._history)

    def summary(self) -> dict:
        """Return counts grouped by severity and attack type."""
        by_severity   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        by_attack     = {}
        for a in self._history:
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
            by_attack[a.attack_type] = by_attack.get(a.attack_type, 0) + 1
        return {"by_severity": by_severity, "by_attack_type": by_attack}

    def clear_history(self) -> None:
        """Flush the in-memory alert history."""
        self._history.clear()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _init_log(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # Only write header if the file does not exist yet (append-safe across restarts)
        file_exists = os.path.isfile(self.log_path)
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            if not file_exists:
                csv.writer(fh).writerow(CSV_HEADER)
        logger.info("Alert log %s: %s", "opened" if file_exists else "created", self.log_path)

    def _write_row(self, alert: Alert) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(alert.to_row())

    def _store(self, alert: Alert) -> None:
        self._history.append(alert)
        if self.max_history and len(self._history) > self.max_history:
            self._history.pop(0)
