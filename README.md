# Shield AI-Powered Intrusion Detection System

> Real-time network threat detection using Random Forest - trained on synthetic KDD-style traffic data with live packet simulation and instant alerting.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Features

| Feature | Description |
|---|---|
| **Random Forest Classifier** | Trained on 5,000 synthetic network flows -- 97%+ accuracy |
| **Live Packet Simulation** | Realistic traffic generator with 15% attack injection ratio |
| **Instant Alerting** | Per-packet real-time classification with confidence scores |
| **Attack Type Detection** | Identifies DoS/DDoS, Port Scans, Brute-Force, and Anomalies |
| **CSV Alert Log** | Timestamped alert records saved to `outputs/alerts.csv` |
| **Model Persistence** | Save and reload trained models with joblib |
| **Full CLI Support** | Configurable packets, intervals, thresholds, and more |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/mervesudeboler/ai-ids.git
cd ai-ids
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py --train
```

### 3. Run Live Simulation

```bash
python main.py --train --simulate --packets 200
```

### 4. Load Existing Model and Simulate

```bash
python main.py --simulate --model outputs/ids_model.pkl
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--train` | - | Train the model on synthetic data |
| `--simulate` | - | Run the live packet monitor |
| `--packets N` | `100` | Number of packets to simulate |
| `--interval S` | `0.03` | Delay between packets in seconds |
| `--model PATH` | `outputs/ids_model.pkl` | Model save/load path |
| `--threshold F` | `0.70` | Minimum confidence to flag as attack |
| `--samples N` | `5000` | Training dataset size |
| `--quiet` | - | Summary only, suppress per-packet output |

---

## Sample Output

```
=================================================================
 AI-IDS - Live Packet Monitor (simulated traffic)
=================================================================
 TIME       SRC IP            DST IP            LABEL        CONF  TYPE
 -----------------------------------------------------------------
 14:22:01.3 192.168.1.5       10.0.0.3          [NORMAL]     94.2% -
 14:22:01.4 172.16.0.103      10.0.0.1          [ATTACK]     98.7% DoS/DDoS
 14:22:01.4 192.168.1.12      10.0.0.7          [NORMAL]     91.5% -
 14:22:01.5 172.16.0.105      10.0.0.2          [ATTACK]     97.1% Port Scan

 -----------------------------------------------------------------
 Total packets  : 200
 Attacks flagged: 31 (15.5%)
 Alert log      : outputs/alerts.csv
=================================================================
```

---

## Project Structure

```
ai-ids/
├── main.py              # CLI entry point -- training, simulation, alerting
├── requirements.txt     # Python dependencies
├── outputs/             # Auto-created at runtime
│   ├── ids_model.pkl    # Saved Random Forest model
│   ├── ids_scaler.pkl   # Saved StandardScaler
│   └── alerts.csv       # Alert log (timestamped)
└── ids/                 # Core IDS package
    ├── __init__.py      # Package exports
    ├── capture.py       # Live packet capture (Scapy)
    ├── features.py      # Feature extraction from raw packets
    ├── model.py         # IDSModel -- Random Forest classifier
    └── alert.py         # AlertManager -- severity levels & CSV logging
```

---

## How It Works

**1. Synthetic Dataset Generation**
KDD Cup-style network flow features are generated for 4 traffic classes: Normal, DoS/DDoS, Port Scan, and Brute-Force/R2L. Class distribution is intentionally imbalanced (60% normal / 40% attack) to reflect real-world conditions.

**2. Feature Engineering**
Each packet/flow is represented by 11 features: packet size, duration, source/destination ports, protocol, TCP flags (SYN, ACK, RST, FIN), bytes per second, and packets per second.

**3. Model Training**
A Random Forest (100 estimators, max depth 10) is trained on an 80/20 train-test split with StandardScaler normalization. Typical accuracy exceeds 97%.

**4. Real-Time Classification**
The PacketSimulator generates continuous packet streams. Each packet is scored by the model and flagged if confidence exceeds the configured threshold.

---

## Dependencies

```
scikit-learn
numpy
joblib
```

> **Note:** `scapy` is listed for live capture capability. Simulated mode works without it.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Made with love by [Merve Sude Böler](https://github.com/mervesudeboler)
