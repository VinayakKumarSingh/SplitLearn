# SplitLearn: Federated Split Learning for Industrial Vibration Fault Detection

SplitLearn is a privacy-preserving distributed learning framework comparing **U-Shape Split Learning** against **Normal Split Learning** for vibration-based fault detection across industrial machines (M01, M02, M03).

---

## 🛠️ Quick Start

### 1. Activate Environment
```bash
# Activate the pre-configured virtual environment:
source .venv/bin/activate
```

### 2. Run Everything with One Command
```bash
python run_pipeline.py full
```
This executes:
1. **U-Shape Split Training** across 3 clients + server
2. **Normal Split Training** across 3 clients + server
3. **Automated Comparison & Plot Generation** (`auto_compare.py` & `auto_plot.py`)
4. **Privacy Vulnerability & Leakage Analysis** (`privacy_analysis.py`)

---

## 🚀 Individual Execution Modes

### A. U-Shape Split Learning
```bash
# In terminal 1 (Server):
python server_split.py

# In terminal 2, 3, 4 (Clients):
python client_split.py M01
python client_split.py M02
python client_split.py M03
```

### B. Normal Split Learning
```bash
# In terminal 1 (Server):
python server_normal_split.py

# In terminal 2, 3, 4 (Clients):
python client_normal_split.py M01
python client_normal_split.py M02
python client_normal_split.py M03
```

### C. Multi-PC / Distributed Setup
```bash
# On Server Machine:
python server_diff_pc.py

# On Client Machines (specify server IP):
python client_diff_pc.py M01 <SERVER_IP>
python client_diff_pc.py M02 <SERVER_IP>
python client_diff_pc.py M03 <SERVER_IP>
```

### D. Analysis & Plots
```bash
# Generate comparison metrics table and CSVs:
python auto_compare.py

# Generate publication plots in plots/:
python auto_plot.py

# Run privacy vulnerability analysis:
python privacy_analysis.py
```

---

## 🌐 Web Applications & Dashboards

### 1. Analysis Dashboard (`app.py`)
Interactive metrics dashboard with comparison tables, Chart.js plots, and privacy audit:
```bash
python app.py
```
Open: **[http://127.0.0.1:5050](http://127.0.0.1:5050)**

### 2. Factory Portal (`portal_modified.py`)
Factory portal for secure login, H5 dataset upload, and live client training:
```bash
python portal_modified.py
```
Open: **[http://127.0.0.1:5051](http://127.0.0.1:5051)**
- Default credentials available in `factories.json` (e.g. `M01` / `Surya@9402`).

### 3. Real-Time Inference & Simulation (`flask_frontend_split.py`)
Live fault prediction and threshold-based continuous alert simulation:
```bash
python flask_frontend_split.py
```
Open: **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## 📁 Project Structure

```
├── client_split.py           # U-Shape split client (M01, M02, M03)
├── server_split.py           # U-Shape split server
├── client_normal_split.py    # Normal split client
├── server_normal_split.py    # Normal split server
├── client_diff_pc.py         # Multi-PC U-Shape client
├── server_diff_pc.py         # Multi-PC U-Shape server
├── client_hybrid.py          # Hybrid split client
├── server_hybrid.py          # Hybrid split server
├── auto_compare.py           # Aggregates logs/*.json -> comparison_summary.csv
├── auto_plot.py              # Generates charts in plots/
├── privacy_analysis.py       # Privacy attacks, MI, BC overlap & gradient leakage
├── app.py                    # Analysis Dashboard Web App (Port 5050)
├── portal_modified.py        # Factory Portal Web App (Port 5051)
├── flask_frontend_split.py   # Machine Inference Service (Port 8080)
├── run_pipeline.py           # Pipeline runner CLI
├── shared_utils.py           # Reproducible seeds and index management
├── new_train/                # Training data per machine (M01, M02, M03)
├── new_test/                 # Evaluation test dataset
├── logs/                     # JSON metrics and evaluation reports
├── plots/                    # Output visual charts
├── factories.json            # Factory credentials store
└── requirements.txt          # Python dependencies
```
