from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import requests

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATHS = {
    "front": Path(r"front_M01_ushape.pth"),
    "middle": Path(r"middle_model.pth"),
    "tail": Path(r"tail_M01_ushape.pth")
}
DATA_FOLDER = Path(r"simulation_10good_10bad")
SERVICE_GOOD_FOLDER = Path(r"150_good")
SERVICE_BAD_FOLDER = Path(r"service_bad")
ALERT_URL = "http://localhost:5000/ml-event"
SIMULATION_RESULT_URL = "http://localhost:3001/verify-service"
CONTINUOUS_BAD_THRESHOLD = 3
CONTINUOUS_GOOD_THRESHOLD = 100
MACHINE_ID = "final phase 3"

# Load models globally
front_model = None
middle_model = None
tail_model = None

# Global flag to track if simulation buttons should be shown
show_simulation_buttons = False

# ✅ ADD THIS: Global variable to store current service record
current_service_record = None

# ============================================================
# PYTORCH MODEL DEFINITIONS (U-shape split architecture)
# ============================================================
class FrontModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 512)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 256)
        return x


class MiddleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))        # (B, 128, 256)
        x = self.gap(x).squeeze(-1)        # (B, 128)
        x = self.relu2(self.fc(x))         # (B, 64)
        return x


class TailModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.sigmoid(self.fc(x))       # (B, 1)
        return x


# ============================================================
# MODEL LOADING
# ============================================================
def load_models():
    global front_model, middle_model, tail_model
    
    if front_model is None or middle_model is None or tail_model is None:
        # Initialize models
        front_model = FrontModel()
        middle_model = MiddleModel()
        tail_model = TailModel()
        
        # Load saved weights
        front_model.load_state_dict(torch.load(MODEL_PATHS["front"], weights_only=True, map_location='cpu'))
        middle_model.load_state_dict(torch.load(MODEL_PATHS["middle"], weights_only=True, map_location='cpu'))
        tail_model.load_state_dict(torch.load(MODEL_PATHS["tail"], weights_only=True, map_location='cpu'))
        
        # Set to evaluation mode
        front_model.eval()
        middle_model.eval()
        tail_model.eval()
        
        print(f"✅ Split models loaded:")
        print(f"   Front: {MODEL_PATHS['front']}")
        print(f"   Middle: {MODEL_PATHS['middle']}")
        print(f"   Tail: {MODEL_PATHS['tail']}")


def predict_split(vibration_data):
    """
    Run inference using the split U-shape model: front → middle → tail
    Input: vibration_data with shape (1024, 3)
    Output: prediction probability (0 to 1), where >0.5 = GOOD, <=0.5 = BAD
    """
    # Reshape for PyTorch Conv1d: (B, channels, sequence) = (1, 3, 1024)
    x = torch.tensor(vibration_data.transpose(1, 0), dtype=torch.float32).unsqueeze(0)  # (1, 3, 1024)
    
    with torch.no_grad():
        # Forward through split architecture
        act = front_model(x)           # (1, 64, 256)
        out = middle_model(act)        # (1, 64)
        pred = tail_model(out)         # (1, 1)
        
        # Return scalar probability
        return pred.item()


# ============================================================
# HTML TEMPLATE (UNCHANGED)
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction Service</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        body {
            background: #f8fafc;
            color: #1e293b;
            min-height: 100vh;
            padding: 2rem;
        }
        
        .container {
            max-width: 900px;
            margin: auto;
        }
        .container {
            max-width: 900px;
            margin: auto;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2rem;
            color: #0f172a;
        }
        
        .config {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .config h3 {
            color: #0f172a;
            font-size: 1.2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .config p {
            margin: 0.5rem 0;
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .config strong {
            color: #0f172a;
            font-weight: 600;
        }
        
        button {
            background: #22c55e;
            color: #022c22;
            padding: 0.6rem 1.5rem;
            text-align: center;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
            margin: 1rem 0;
            transition: all 0.2s ease;
        }
        
        button:hover {
            opacity: 0.9;
        }
        
        button:active {
            transform: scale(0.98);
        }
        
        button:disabled {
            background: #e2e8f0;
            color: #94a3b8;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        .simulation-buttons {
            display: none;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .simulation-buttons button {
            flex: 1;
        }
        
        .simulate-good {
            background: #22c55e;
            color: #022c22;
        }
        
        .simulate-good:hover {
            opacity: 0.9;
        }
        
        .simulate-bad {
            background: #ef4444;
            color: #fff;
        }
        
        .simulate-bad:hover {
            opacity: 0.9;
        }
        
        .results {
            margin-top: 2rem;
            padding: 0;
            border-radius: 12px;
            display: none;
        }
        
        .results.success {
            background: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .results.error {
            background: #fee2e2;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .results.error h3 {
            color: #991b1b;
            margin-bottom: 0.5rem;
        }
        
        .results.error p {
            color: #dc2626;
        }
        
        .summary {
            margin: 0;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 12px 12px 0 0;
        }
        
        .summary h3 {
            margin: 0 0 1rem 0;
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        .summary p {
            margin: 0.7rem 0;
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .summary strong {
            color: #0f172a;
            font-weight: 600;
        }
        
        .predictions-list {
            max-height: 450px;
            overflow-y: auto;
            margin-top: 0;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 0 0 12px 12px;
        }
        
        .predictions-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .predictions-list::-webkit-scrollbar-track {
            background: #e2e8f0;
            border-radius: 10px;
        }
        
        .predictions-list::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }
        
        .predictions-list::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        .predictions-list h3 {
            color: #0f172a;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        
        .prediction-item {
            padding: 0.6rem 0.8rem;
            margin: 0.5rem 0;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }
        
        .prediction-item:hover {
            transform: translateX(5px);
        }
        
        .prediction-good {
            background: #dcfce7;
            border-left-color: #22c55e;
            color: #166534;
        }
        
        .prediction-bad {
            background: #fee2e2;
            border-left-color: #ef4444;
            color: #991b1b;
        }
        
        .loader {
            border: 6px solid #e2e8f0;
            border-top: 6px solid #22c55e;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 0.8s linear infinite;
            margin: 2rem auto;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert-badge {
            background: #ef4444;
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .success-badge {
            background: #22c55e;
            color: #022c22;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }
            
            h1 {
                font-size: 1.5rem;
            }
            
            .simulation-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CNC Failure Detection Simulator</h1>
        
        <div class="config">
            <p><strong>Model:</strong> U-Shape Split (Front+Middle+Tail)</p>
            <p><strong>Machine ID:</strong> {{ machine_id }}</p>
            <p><strong>Continuous Bad Threshold:</strong> {{ threshold }}</p>
        </div>

        <button id="runBtn" onclick="runPredictions()">
            🚀 Run Predictions
        </button>

        <div class="simulation-buttons" id="simulationButtons">
            <button class="simulate-good" onclick="simulateGood()">
                ✅ Simulate Good
            </button>
            <button class="simulate-bad" onclick="simulateBad()">
                ⚠️ Simulate Bad
            </button>
        </div>

        <div class="loader" id="loader"></div>

        <div id="results" class="results"></div>
    </div>

    <script>
        // Poll for simulation button visibility
        function checkSimulationStatus() {
            fetch('/simulation-status')
                .then(response => response.json())
                .then(data => {
                    const buttons = document.getElementById('simulationButtons');
                    buttons.style.display = data.show_buttons ? 'flex' : 'none';
                })
                .catch(error => console.error('Error checking status:', error));
        }

        // Check immediately on page load and then every 2 seconds
        document.addEventListener('DOMContentLoaded', function() {
            checkSimulationStatus();
            setInterval(checkSimulationStatus, 2000);
        });

        async function runPredictions() {
            const btn = document.getElementById('runBtn');
            const loader = document.getElementById('loader');
            const results = document.getElementById('results');
            
            btn.disabled = true;
            btn.textContent = '⏳ Running predictions...';
            loader.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/run-predictions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    displayError(data.error || 'Unknown error occurred');
                }
            } catch (error) {
                displayError(error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Run Predictions';
                loader.style.display = 'none';
            }
        }

        async function simulateGood() {
            const loader = document.getElementById('loader');
            const results = document.getElementById('results');
            
            loader.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/simulate-good', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displaySimulationResults(data, 'good');
                } else {
                    displayError(data.error || 'Unknown error occurred');
                }
            } catch (error) {
                displayError(error.message);
            } finally {
                loader.style.display = 'none';
            }
        }

        async function simulateBad() {
            const loader = document.getElementById('loader');
            const results = document.getElementById('results');
            
            loader.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/simulate-bad', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displaySimulationResults(data, 'bad');
                } else {
                    displayError(data.error || 'Unknown error occurred');
                }
            } catch (error) {
                displayError(error.message);
            } finally {
                loader.style.display = 'none';
            }
        }

        function displayResults(data) {
            const results = document.getElementById('results');
            results.className = 'results success';
            results.style.display = 'block';
            
            let html = `
                <div class="summary">
                    <h3>📊 Prediction Summary</h3>
                    <p><strong>Total Files Processed:</strong> ${data.total_files}</p>
                    <p><strong>Total GOOD Predictions:</strong> ${data.total_good}</p>
                    <p><strong>Total BAD Predictions:</strong> ${data.total_bad}</p>
                    <p><strong>Alert Triggered:</strong> ${data.alert_triggered ? '<span class="alert-badge">YES - Alert Sent!</span>' : 'NO'}</p>
                </div>
                
                <h3>📋 Detailed Predictions</h3>
                <div class="predictions-list">
            `;
            
            data.predictions.forEach((pred, index) => {
                const className = pred.prediction === 'GOOD' ? 'prediction-good' : 'prediction-bad';
                const icon = pred.prediction === 'GOOD' ? '✅' : '⚠️';
                html += `
                    <div class="prediction-item ${className}">
                        [${index + 1}] ${icon} ${pred.prediction} - ${pred.file} 
                        ${pred.prediction === 'BAD' ? `(Continuous: ${pred.continuous_bad_count})` : ''}
                    </div>
                `;
            });
            
            html += '</div>';
            results.innerHTML = html;
        }

        function displaySimulationResults(data, type) {
            const results = document.getElementById('results');
            results.className = 'results success';
            results.style.display = 'block';
            
            const targetLabel = type === 'good' ? '100 Continuous GOOD' : '3 Continuous BAD';
            const targetReached = type === 'good' ? data.target_reached : data.threshold_reached;
            
            let html = `
                <div class="summary">
                    <h3>📊 Simulation ${type.toUpperCase()} Results</h3>
                    <p><strong>Total Files Processed:</strong> ${data.total_files}</p>
                    <p><strong>Total GOOD Predictions:</strong> ${data.total_good}</p>
                    <p><strong>Total BAD Predictions:</strong> ${data.total_bad}</p>
                    ${type === 'good' ? 
                        `<p><strong>Continuous Good Count:</strong> ${data.continuous_good_count}</p>` :
                        `<p><strong>Continuous Bad Count:</strong> ${data.continuous_bad_count}</p>`
                    }
                    <p><strong>${targetLabel} Target:</strong> ${targetReached ? '<span class="success-badge">ACHIEVED - Result Sent!</span>' : '<span class="alert-badge">NOT REACHED</span>'}</p>
                    <p><strong>Result Sent:</strong> ${data.result_sent ? 'true' : 'false'}</p>
                </div>
                
                <h3>📋 Detailed Predictions</h3>
                <div class="predictions-list">
            `;
            
            data.predictions.forEach((pred, index) => {
                const className = pred.prediction === 'GOOD' ? 'prediction-good' : 'prediction-bad';
                const icon = pred.prediction === 'GOOD' ? '✅' : '⚠️';
                html += `
                    <div class="prediction-item ${className}">
                        [${index + 1}] ${icon} ${pred.prediction} - ${pred.file}
                    </div>
                `;
            });
            
            html += '</div>';
            results.innerHTML = html;
        }

        function displayError(error) {
            const results = document.getElementById('results');
            results.className = 'results error';
            results.style.display = 'block';
            results.innerHTML = `
                <h3>❌ Error</h3>
                <p>${error}</p>
            `;
        }
    </script>
</body>
</html>
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def send_alert(machine_id, status):
    """Send POST request to alert endpoint"""
    try:
        payload = {
            "machine_id": machine_id,
            "status": status
        }
        response = requests.post(
            ALERT_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"🚨 Alert sent: {payload} - Response: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Error sending alert: {e}")
        return False

def send_simulation_result(result):
    """Send simulation result to port 3001"""
    global current_service_record, show_simulation_buttons
    try:
        if not current_service_record:
            print("⚠️ No service record available - cannot send result")
            return False
            
        payload = {
            "service_record": current_service_record,
            "verified": result
        }
        
        print(f"📤 Sending to {SIMULATION_RESULT_URL}")
        print(f"📦 Payload: {payload}")
        
        response = requests.post(
            SIMULATION_RESULT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✅ Response status: {response.status_code}")
        print(f"📦 Response body: {response.text}")
        
        if response.status_code == 200:
            print(f"✅ Simulation result sent successfully!")
            # Reset to initial state after successful send
            show_simulation_buttons = False
            current_service_record = None
            print(f"🔄 Reset to initial state - buttons hidden")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending simulation result: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    """Serve the frontend"""
    return render_template_string(
        HTML_TEMPLATE,
        machine_id=MACHINE_ID,
        threshold=CONTINUOUS_BAD_THRESHOLD,
        alert_url=ALERT_URL,
        show_buttons=show_simulation_buttons
    )

@app.route('/simulation-status', methods=['GET'])
def simulation_status():
    """Check if simulation buttons should be shown"""
    global show_simulation_buttons
    return jsonify({
        "show_buttons": show_simulation_buttons
    }), 200

@app.route('/trigger-simulation', methods=['POST'])
def trigger_simulation():
    """External endpoint to trigger simulation buttons display"""
    global show_simulation_buttons, current_service_record
    
    data = request.json or {}
    current_service_record = data.get('service_record')
    
    show_simulation_buttons = True
    print(f"🎮 Simulation mode activated - buttons are now visible")
    print(f"📝 Service record: {current_service_record}")
    
    return jsonify({
        "status": "success",
        "message": "Simulation buttons activated"
    }), 200

@app.route('/run-predictions', methods=['POST'])
def run_predictions():
    """Run predictions on all files in simulation_10good_10bad using split model"""
    try:
        load_models()  # Load PyTorch split models
        
        # Get all h5 files in order
        h5_files = sorted(list(DATA_FOLDER.glob('*.h5')))
        
        if not h5_files:
            return jsonify({
                "error": "No .h5 files found in simulation_10good_10bad"
            }), 404
        
        # Tracking variables
        continuous_bad_count = 0
        total_files = 0
        total_good = 0
        total_bad = 0
        alert_triggered = False
        predictions = []
        
        print(f"\n{'='*60}")
        print(f"🔍 Starting predictions on {len(h5_files)} files (U-Shape Split Model)...")
        print(f"{'='*60}\n")
        
        # Process each file
        for file_path in h5_files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    # Load vibration_data
                    data = hf["vibration_data"][:]  # (1024, 3)
                    
                    # Predict using split model
                    prob = predict_split(data)  # Returns probability 0-1
                    predicted_class = int(prob > 0.5)  # >0.5 = GOOD (1), <=0.5 = BAD (0)
                    prediction_label = "GOOD" if predicted_class == 1 else "BAD"
                    
                    total_files += 1
                    
                    if predicted_class == 0:  # BAD
                        continuous_bad_count += 1
                        total_bad += 1
                        print(f"[{total_files:3d}] ⚠️  BAD (continuous: {continuous_bad_count}): {file_path.name} (prob={prob:.4f})")
                        
                        # Check if threshold reached
                        if continuous_bad_count >= CONTINUOUS_BAD_THRESHOLD and not alert_triggered:
                            print(f"\n🚨 ALERT: {CONTINUOUS_BAD_THRESHOLD} continuous BAD predictions detected!")
                            send_alert(MACHINE_ID, "BAD")
                            alert_triggered = True
                            
                    else:  # GOOD
                        total_good += 1
                        print(f"[{total_files:3d}] ✅ GOOD: {file_path.name} (prob={prob:.4f})")
                        continuous_bad_count = 0  # Reset counter
                    
                    predictions.append({
                        "file": file_path.name,
                        "prediction": prediction_label,
                        "continuous_bad_count": continuous_bad_count
                    })
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                predictions.append({
                    "file": file_path.name,
                    "error": str(e)
                })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 PREDICTION SUMMARY:")
        print(f"  Total files processed: {total_files}")
        print(f"  Total GOOD predictions: {total_good}")
        print(f"  Total BAD predictions: {total_bad}")
        print(f"  Alert triggered: {'YES' if alert_triggered else 'NO'}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "total_files": total_files,
            "total_good": total_good,
            "total_bad": total_bad,
            "alert_triggered": alert_triggered,
            "continuous_bad_threshold": CONTINUOUS_BAD_THRESHOLD,
            "predictions": predictions
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/simulate-good', methods=['POST'])
def simulate_good():
    """Run predictions on service_good folder using split model"""
    try:
        load_models()
        
        # Get all h5 files from service_good
        h5_files = sorted(list(SERVICE_GOOD_FOLDER.glob('*.h5')))
        
        if not h5_files:
            return jsonify({
                "error": "No .h5 files found in service_good"
            }), 404
        
        # Tracking variables
        continuous_good_count = 0
        total_files = 0
        total_good = 0
        total_bad = 0
        target_reached = False
        result_sent = False
        predictions = []
        
        print(f"\n{'='*60}")
        print(f"✅ SIMULATE GOOD: Starting predictions on {len(h5_files)} files (U-Shape Split)...")
        print(f"{'='*60}\n")
        
        # Process each file
        for file_path in h5_files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    data = hf["vibration_data"][:]
                    prob = predict_split(data)
                    predicted_class = int(prob > 0.5)
                    prediction_label = "GOOD" if predicted_class == 1 else "BAD"
                    
                    total_files += 1
                    
                    if predicted_class == 1:  # GOOD
                        continuous_good_count += 1
                        total_good += 1
                        print(f"[{total_files:3d}] ✅ GOOD (continuous: {continuous_good_count}): {file_path.name} (prob={prob:.4f})")
                        
                        if continuous_good_count >= CONTINUOUS_GOOD_THRESHOLD and not target_reached:
                            print(f"\n🎯 SUCCESS: {CONTINUOUS_GOOD_THRESHOLD} continuous GOOD predictions!")
                            send_simulation_result(True)
                            target_reached = True
                            result_sent = True
                            
                    else:  # BAD
                        total_bad += 1
                        print(f"[{total_files:3d}] ⚠️  BAD: {file_path.name} (prob={prob:.4f})")
                        continuous_good_count = 0
                    
                    predictions.append({
                        "file": file_path.name,
                        "prediction": prediction_label
                    })
                    
                    if target_reached:
                        break
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                predictions.append({
                    "file": file_path.name,
                    "error": str(e)
                })
        
        print(f"\n{'='*60}")
        print(f"📊 SIMULATE GOOD SUMMARY:")
        print(f"  Total files processed: {total_files}")
        print(f"  Total GOOD: {total_good}, BAD: {total_bad}")
        print(f"  Continuous good: {continuous_good_count}, Target: {'YES' if target_reached else 'NO'}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "total_files": total_files,
            "total_good": total_good,
            "total_bad": total_bad,
            "continuous_good_count": continuous_good_count,
            "target_reached": target_reached,
            "result_sent": result_sent,
            "predictions": predictions
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/simulate-bad', methods=['POST'])
def simulate_bad():
    """Run predictions on service_bad folder using split model"""
    try:
        load_models()
        
        h5_files = sorted(list(SERVICE_BAD_FOLDER.glob('*.h5')))
        
        if not h5_files:
            return jsonify({"error": "No .h5 files found in service_bad"}), 404
        
        continuous_bad_count = 0
        total_files = 0
        total_good = 0
        total_bad = 0
        threshold_reached = False
        result_sent = False
        predictions = []
        
        print(f"\n{'='*60}")
        print(f"⚠️  SIMULATE BAD: Starting predictions on {len(h5_files)} files (U-Shape Split)...")
        print(f"{'='*60}\n")
        
        for file_path in h5_files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    data = hf["vibration_data"][:]
                    prob = predict_split(data)
                    predicted_class = int(prob > 0.5)
                    prediction_label = "GOOD" if predicted_class == 1 else "BAD"
                    
                    total_files += 1
                    
                    if predicted_class == 0:  # BAD
                        continuous_bad_count += 1
                        total_bad += 1
                        print(f"[{total_files:3d}] ⚠️  BAD (continuous: {continuous_bad_count}): {file_path.name} (prob={prob:.4f})")
                        
                        if continuous_bad_count >= CONTINUOUS_BAD_THRESHOLD and not threshold_reached:
                            print(f"\n🚨 THRESHOLD REACHED: {CONTINUOUS_BAD_THRESHOLD} continuous BAD!")
                            send_simulation_result(False)
                            threshold_reached = True
                            result_sent = True
                            
                    else:
                        total_good += 1
                        print(f"[{total_files:3d}] ✅ GOOD: {file_path.name} (prob={prob:.4f})")
                        continuous_bad_count = 0
                    
                    predictions.append({"file": file_path.name, "prediction": prediction_label})
                    
                    if threshold_reached:
                        break
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                predictions.append({"file": file_path.name, "error": str(e)})
        
        print(f"\n{'='*60}")
        print(f"📊 SIMULATE BAD SUMMARY:")
        print(f"  Total: {total_files}, GOOD: {total_good}, BAD: {total_bad}")
        print(f"  Continuous bad: {continuous_bad_count}, Threshold: {'YES' if threshold_reached else 'NO'}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "total_files": total_files,
            "total_good": total_good,
            "total_bad": total_bad,
            "continuous_bad_count": continuous_bad_count,
            "threshold_reached": threshold_reached,
            "result_sent": result_sent,
            "predictions": predictions
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": all([front_model, middle_model, tail_model])
    }), 200

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("🚀 Starting Flask Prediction Service on port 8080...")
    print(f"📁 Data folder: {DATA_FOLDER}")
    print(f"📁 Service good: {SERVICE_GOOD_FOLDER}")
    print(f"📁 Service bad: {SERVICE_BAD_FOLDER}")
    print(f"🤖 Split Models:")
    print(f"   Front: {MODEL_PATHS['front']}")
    print(f"   Middle: {MODEL_PATHS['middle']}")
    print(f"   Tail: {MODEL_PATHS['tail']}")
    print(f"🎯 Thresholds: Bad={CONTINUOUS_BAD_THRESHOLD}, Good={CONTINUOUS_GOOD_THRESHOLD}")
    print(f"📡 Alert: {ALERT_URL}")
    print(f"📡 Simulation: {SIMULATION_RESULT_URL}")
    print(f"🌐 Open: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)