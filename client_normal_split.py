import os
import sys
import socket
import pickle
import struct
import time
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

# ---- fair-comparison utilities ----
from shared_utils import set_seed, get_or_create_indices

# ============================================================
# SERVER CONNECTION CONFIG
# ============================================================
SERVER_IP      = "127.0.0.1"
SERVER_PORT    = 8888
SOCKET_TIMEOUT = 120  # seconds

# ============================================================
# COMMUNICATION TRACKING
# ============================================================
SENT_BYTES = 0
RECV_BYTES = 0

# ============================================================
# SOCKET HELPERS
# ============================================================
def send_msg(sock, data):
    global SENT_BYTES
    msg = pickle.dumps(data)
    length_prefix = struct.pack(">I", len(msg))
    sock.sendall(length_prefix + msg)
    SENT_BYTES += len(length_prefix) + len(msg)

def recv_msg(sock):
    global RECV_BYTES
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recvall(sock, msg_len)
    if data is None:
        return None
    RECV_BYTES += 4 + msg_len
    return pickle.loads(data)

def recvall(sock, n):
    data = b""
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout:
            raise RuntimeError(
                f"[Client] Socket timed out waiting for {n - len(data)} more bytes "
                f"(received {len(data)}/{n})"
            )
        if not packet:
            return None
        data += packet
    return data

# ============================================================
# RESULTS SAVER
# ============================================================
def save_results(method, machine_id, acc, f1, time_taken, sent_mb, recv_mb):
    result = {
        "method":        method,
        "client":        machine_id,
        "accuracy":      float(acc),
        "f1":            float(f1),
        "time":          float(time_taken),
        "sent_MB":       float(sent_mb),
        "recv_MB":       float(recv_mb),
        "total_comm_MB": float(sent_mb + recv_mb),
    }
    os.makedirs("logs", exist_ok=True)
    file_path = f"logs/{method}_{machine_id}.json"
    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[{machine_id}] Results saved to {file_path}")

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class FrontModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,  32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 512)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 256)
        return x


class ServerModel(nn.Module):
    """Replica used only for local evaluation after training."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc1  = nn.Linear(128, 64)
        self.fc2  = nn.Linear(64, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.gap(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
        return x

# ============================================================
# DATA LOADERS
# ============================================================
def load_data(machine_id, base_dir="new_train"):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, machine_id, folder)
        if not os.path.exists(path):
            print(f"[{machine_id}] Warning: path not found: {path}")
            continue
        files = sorted(f for f in os.listdir(path) if f.endswith(".h5"))
        for f in tqdm(files, desc=f"Loading {machine_id}/{folder}"):
            with h5py.File(os.path.join(path, f), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        raise RuntimeError(f"[{machine_id}] No training data found in {base_dir}/{machine_id}")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    # FIX: use shared, saved indices so both methods see the same data order
    idx = get_or_create_indices(len(X), machine_id, split="train")
    return X[idx], y[idx]


def load_global_test(base_dir="new_test"):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            print(f"Warning: test path not found: {path}")
            continue
        files = sorted(f for f in os.listdir(path) if f.endswith(".h5"))
        for f in tqdm(files, desc=f"Loading test/{folder}"):
            with h5py.File(os.path.join(path, f), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        raise RuntimeError(f"No test data found in {base_dir}")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    # FIX: shared test indices — generated once, reused by both methods
    idx = get_or_create_indices(len(X), machine_id="global", split="test")
    return X[idx], y[idx]

# ============================================================
# CLIENT MAIN
# ============================================================
def run_client():
    global SENT_BYTES, RECV_BYTES
    SENT_BYTES = 0
    RECV_BYTES = 0

    # FIX: seed everything before any model init or data loading
    set_seed()

    start_time = time.time()

    if len(sys.argv) < 2:
        print("Usage: python client_normal_split.py <M01|M02|M03>")
        sys.exit(1)

    machine_id = sys.argv[1]
    print(f"[{machine_id}] Starting client (normal split learning)...")

    # ---- Load training data ----
    X_train, y_train = load_data(machine_id)
    print(f"[{machine_id}] Loaded {len(X_train)} training samples")

    # ---- Connect to server ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.connect((SERVER_IP, SERVER_PORT))
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f"[{machine_id}] Could not connect to server at {SERVER_IP}:{SERVER_PORT} — {e}")
        sys.exit(1)

    send_msg(sock, {"id": machine_id, "count": len(X_train)})

    cfg = recv_msg(sock)
    if cfg is None:
        raise RuntimeError(f"[{machine_id}] Did not receive config from server.")
    num_batches = cfg["num_batches"]
    batch_size  = cfg["batch_size"]
    effective   = cfg["effective_samples"]
    epochs      = cfg["epochs"]
    print(
        f"[{machine_id}] Config — batch_size={batch_size}, "
        f"num_batches={num_batches}, effective={effective}, epochs={epochs}"
    )

    X_tr = X_train[:effective]
    y_tr = y_train[:effective]

    # FIX: init front model AFTER seed is set → same weights as U-shape run
    front     = FrontModel()
    opt_front = optim.Adam(front.parameters(), lr=0.001)

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(epochs):
        # FIX: per-epoch shuffle uses a deterministic derived seed
        rng  = np.random.default_rng(42 + epoch)
        idx  = rng.permutation(effective)
        X_ep = X_tr[idx]
        y_ep = y_tr[idx]

        epoch_loss = 0.0
        front.train()

        for b in range(num_batches):
            start = b * batch_size
            end   = start + batch_size

            x_batch = torch.tensor(
                X_ep[start:end].transpose(0, 2, 1), dtype=torch.float32
            )
            y_batch = y_ep[start:end].reshape(-1, 1)

            opt_front.zero_grad()
            act = front(x_batch)

            send_msg(sock, {
                "activation": act.detach().numpy(),
                "labels":     y_batch,
            })

            msg = recv_msg(sock)
            if msg is None:
                raise RuntimeError(f"[{machine_id}] Server disconnected during batch {b}.")
            upstream_np = msg["upstream_grad"]
            epoch_loss += msg["loss"]

            act.backward(torch.tensor(upstream_np, dtype=torch.float32))
            opt_front.step()

        send_msg(sock, {"epoch_done": True})
        avg = epoch_loss / num_batches
        print(f"[{machine_id}] Epoch {epoch + 1}/{epochs} | Avg Loss: {avg:.4f}")

    # ============================================================
    # WAIT FOR FINAL SERVER SIGNAL
    # ============================================================
    final_msg = recv_msg(sock)

    if final_msg is None or not final_msg.get("training_complete"):
        raise RuntimeError(f"[{machine_id}] Did not receive training completion signal.")

    print(f"[{machine_id}] Server confirmed model saved.")

    sock.close()
    print(f"[{machine_id}] Training complete. Socket closed.")

    # ============================================================
    # GLOBAL TEST EVALUATION
    # FIX: use relative path instead of hardcoded Windows absolute path
    # ============================================================
    print(f"\n[{machine_id}] Running global test evaluation...")

    server_weights = "server_model1.pth"   # <-- FIXED (was hardcoded Windows path)
    if not os.path.exists(server_weights):
        print(f"[{machine_id}] Warning: '{server_weights}' not found. Skipping evaluation.")
        return

    X_test, y_test = load_global_test()
    print(f"[{machine_id}] Loaded {len(X_test)} test samples")

    server_model = ServerModel()
    server_model.load_state_dict(torch.load(server_weights, weights_only=True))
    server_model.eval()
    front.eval()

    all_preds = []
    with torch.no_grad():
        num_test_batches = len(X_test) // batch_size

        for b in range(num_test_batches):
            s   = b * batch_size
            e   = s + batch_size
            x_t = torch.tensor(
                X_test[s:e].transpose(0, 2, 1), dtype=torch.float32
            )
            p = server_model(front(x_t))
            all_preds.extend(p.squeeze(1).numpy().tolist())

        remainder = len(X_test) - num_test_batches * batch_size
        if remainder > 0:
            x_t = torch.tensor(
                X_test[-remainder:].transpose(0, 2, 1), dtype=torch.float32
            )
            p = server_model(front(x_t))
            all_preds.extend(p.squeeze(1).numpy().tolist())

    preds_np  = np.array(all_preds)
    preds_bin = (preds_np > 0.5).astype(int)
    y_true    = y_test[:len(preds_bin)].astype(int)

    acc = (preds_bin == y_true).mean()
    f1  = f1_score(y_true, preds_bin, zero_division=0)
    cm  = confusion_matrix(y_true, preds_bin)

    print(f"[{machine_id}] Test Accuracy  : {acc:.4f}")
    print(f"[{machine_id}] Test F1 Score  : {f1:.4f}")
    print(f"[{machine_id}] Confusion Matrix:\n{cm}")

    sent_mb = SENT_BYTES / (1024 * 1024)
    recv_mb = RECV_BYTES / (1024 * 1024)
    print(f"[{machine_id}] Sent: {sent_mb:.2f} MB | Received: {recv_mb:.2f} MB")

    torch.save(front.state_dict(), f"front_{machine_id}_normal.pth")
    print(f"[{machine_id}] Front model weights saved.")

    save_results(
        method     = "normal_split",
        machine_id = machine_id,
        acc        = acc,
        f1         = f1,
        time_taken = time.time() - start_time,
        sent_mb    = sent_mb,
        recv_mb    = recv_mb,
    )


if __name__ == "__main__":
    run_client()