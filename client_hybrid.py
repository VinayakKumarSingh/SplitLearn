import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
import struct
import json
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

# ---- fair-comparison utilities ----
from shared_utils import set_seed, get_or_create_indices


# Add to top of client_split.py and client_normal_split.py
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
tqdm.tqdm = lambda *args, **kwargs: tqdm.tqdm(*args, **kwargs, disable=True)
# Or for older Python:
# import functools
# print = functools.partial(print, flush=True)

# ============================================================
# COMMUNICATION TRACKING
# ============================================================
SENT_BYTES = 0
RECV_BYTES = 0

# ============================================================
# SERVER CONNECTION CONFIG
# ============================================================
SERVER_IP   = "127.0.0.1"
SERVER_PORT = 8888
SOCKET_TIMEOUT = 120  # seconds

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
#
# Full model (split across client-server-client):
#   Conv1D(32,k=5) -> Pool -> Conv1D(64,k=5) -> Pool        [Client Front]
#   -> Conv1D(128,k=3) -> GAP -> Dense(64,relu)              [Server Middle]
#   -> Dropout(0.3) -> Dense(1) -> Sigmoid                   [Client Tail]
#
# Tensor shapes:
#   Input            : (B, 3, 1024)
#   After Front      : (B, 64, 256)
#   After Middle     : (B, 64)
#   After Tail       : (B, 1)
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


class TailModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.sigmoid(self.fc(x))
        return x  # (B, 1)


# Middle is defined on the client only for local test evaluation
# (loads the shared weights saved by the server after training)
class MiddleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))   # (B, 128, 256)
        x = self.gap(x).squeeze(-1)   # (B, 128)
        x = self.relu2(self.fc(x))    # (B, 64)
        return x

# ============================================================
# DATA LOADERS
# ============================================================
def load_data(machine_id, base_dir):
    """Load per-machine training data from <base_dir>/<machine_id>/good|bad/."""
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
        raise RuntimeError(f"[{machine_id}] No data found in {base_dir}/{machine_id}")
    X = np.stack(X).astype(np.float32)  # (N, 1024, 3)
    y = np.array(y, dtype=np.float32)
    # FIX: shared indices — same shuffle as normal_split for this machine
    idx = get_or_create_indices(len(X), machine_id, split="train")
    return X[idx], y[idx]


def load_global_test(base_dir):
    """Load shared test data from <base_dir>/good|bad/ (no machine subfolder)."""
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
    X = np.stack(X).astype(np.float32)  # (N, 1024, 3)
    y = np.array(y, dtype=np.float32)
    # FIX: shared test indices — same order as normal_split evaluation
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
        print("Usage: python client_split.py <M01|M02|M03>")
        sys.exit(1)

    machine_id = sys.argv[1]
    print(f"[{machine_id}] Starting client (U-shape split learning)...")

    # ---- Load training data ----
    X_train, y_train = load_data(machine_id, base_dir="new_train")
    print(f"[{machine_id}] Loaded {len(X_train)} training samples")

    # ---- Connect to server ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.connect((SERVER_IP, SERVER_PORT))
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f"[{machine_id}] Could not connect to server at {SERVER_IP}:{SERVER_PORT} — {e}")
        sys.exit(1)

    # Send machine ID and sample count
    send_msg(sock, {"id": machine_id, "count": len(X_train)})

    # ---- Receive training config ----
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

    # Truncate to effective samples
    X_tr = X_train[:effective]
    y_tr = y_train[:effective]

    # ---- Init models and optimisers ----
    # FIX: model init AFTER seed → same initial weights as normal_split's front
    front     = FrontModel()
    tail      = TailModel()
    opt_front = optim.Adam(front.parameters(), lr=0.001)
    opt_tail  = optim.Adam(tail.parameters(),  lr=0.001)
    bce       = nn.BCELoss()

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(epochs):
        # FIX: per-epoch shuffle uses same derived seed as normal_split
        rng  = np.random.default_rng(42 + epoch)
        idx  = rng.permutation(effective)
        X_ep = X_tr[idx]
        y_ep = y_tr[idx]

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        front.train()
        tail.train()

        # Progress bar for batches with formatted output (from Code 1)
        pbar = tqdm(range(num_batches), desc=f"[{machine_id}] Epoch {epoch+1}/{epochs}", 
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for b in pbar:
            start = b * batch_size
            end   = start + batch_size

            # Prepare batch — transpose (B, 1024, 3) → (B, 3, 1024) for Conv1d
            x_batch = torch.tensor(
                X_ep[start:end].transpose(0, 2, 1), dtype=torch.float32
            )                                                   # (B, 3, 1024)
            y_batch = torch.tensor(
                y_ep[start:end], dtype=torch.float32
            ).unsqueeze(1)                                      # (B, 1)

            # --------------------------------------------------
            # FORWARD: front layers
            # act retains grad_fn so backward can flow through it
            # --------------------------------------------------
            opt_front.zero_grad()
            opt_tail.zero_grad()

            act    = front(x_batch)                             # (B, 64, 256) — retains grad_fn
            act_np = act.detach().numpy()

            # Send activation batch to server
            send_msg(sock, {"activation": act_np})

            # Receive server middle output
            msg = recv_msg(sock)
            if msg is None:
                raise RuntimeError(f"[{machine_id}] Server disconnected during forward pass.")
            server_out_np = msg["server_out"]                   # (B, 64)

            # Recreate as leaf tensor so tail backward gives grad
            server_out = torch.tensor(
                server_out_np, dtype=torch.float32, requires_grad=True
            )

            # --------------------------------------------------
            # FORWARD: tail layers + loss
            # --------------------------------------------------
            pred = tail(server_out)                             # (B, 1)
            loss = bce(pred, y_batch)
            epoch_loss += loss.item()

            # Calculate batch accuracy (from Code 1)
            pred_binary = (pred.detach().numpy() > 0.5).astype(np.float32)
            batch_correct = (pred_binary == y_batch.numpy()).sum()
            epoch_correct += batch_correct
            epoch_total += batch_size

            # Update progress bar with current batch metrics (from Code 1)
            batch_acc = batch_correct / batch_size
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.3f}'})

            # --------------------------------------------------
            # BACKWARD: tail — get gradient w.r.t server_out
            # --------------------------------------------------
            loss.backward()
            tail_grad_np = server_out.grad.detach().cpu().numpy()  # (B, 64)
            opt_tail.step()

            # Send tail gradient and loss to server
            send_msg(sock, {"tail_grad": tail_grad_np, "loss": loss.item()})

            # Receive upstream gradient from server
            msg2 = recv_msg(sock)
            if msg2 is None:
                raise RuntimeError(f"[{machine_id}] Server disconnected during backward pass.")
            upstream_np     = msg2["upstream_grad"]              # (B, 64, 256)
            upstream_tensor = torch.tensor(upstream_np, dtype=torch.float32)

            # --------------------------------------------------
            # BACKWARD: front — use upstream grad to complete
            # backprop through front layers
            # --------------------------------------------------
            act.backward(upstream_tensor)
            opt_front.step()

        # Epoch done signal to server
        send_msg(sock, {"epoch_done": True})
        avg_loss = epoch_loss / num_batches
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        print(f"[{machine_id}] Epoch {epoch+1}/{epochs} done | Avg Loss: {avg_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # ============================================================
    # WAIT FOR FINAL SERVER SIGNAL
    # ============================================================
    final_msg = recv_msg(sock)

    if final_msg is None or not final_msg.get("training_complete"):
        raise RuntimeError(f"[{machine_id}] Did not receive training completion signal.")

    print(f"[{machine_id}] Server confirmed middle model saved.")

    sock.close()
    print(f"[{machine_id}] Training complete. Socket closed.")

    # ============================================================
    # GLOBAL TEST EVALUATION
    # All three clients evaluate but each uses its own front+tail
    # paired with the saved shared middle
    # ============================================================
    print(f"\n[{machine_id}] Running global test evaluation...")

    middle_weights = "middle_model.pth"
    if not os.path.exists(middle_weights):
        print(f"[{machine_id}] Warning: '{middle_weights}' not found. Skipping evaluation.")
        return

    X_test, y_test = load_global_test(base_dir="new_test")
    print(f"[{machine_id}] Loaded {len(X_test)} test samples")

    middle = MiddleModel()
    middle.load_state_dict(torch.load(middle_weights, weights_only=True))
    middle.eval()
    front.eval()
    tail.eval()

    all_preds = []
    with torch.no_grad():
        num_test_batches = len(X_test) // batch_size
        if num_test_batches == 0:
            print(
                f"[{machine_id}] Warning: test set ({len(X_test)} samples) smaller than "
                f"batch_size ({batch_size}). Evaluating as single batch."
            )

        for b in range(num_test_batches):
            start  = b * batch_size
            end    = start + batch_size
            x_t    = torch.tensor(
                X_test[start:end].transpose(0, 2, 1), dtype=torch.float32
            )                                                   # (B, 3, 1024)
            act    = front(x_t)                                 # (B, 64, 256)
            out    = middle(act)                                # (B, 64)
            p      = tail(out)                                  # (B, 1)
            all_preds.extend(p.squeeze(1).numpy().tolist())

        # Handle leftover samples
        remainder = len(X_test) - num_test_batches * batch_size
        if remainder > 0:
            x_t = torch.tensor(
                X_test[-remainder:].transpose(0, 2, 1), dtype=torch.float32
            )
            act = front(x_t)
            out = middle(act)
            p   = tail(out)
            all_preds.extend(p.squeeze(1).numpy().tolist())

    preds_np  = np.array(all_preds)
    preds_bin = (preds_np > 0.5).astype(int)
    y_true    = y_test[:len(preds_bin)].astype(int)

    acc = (preds_bin == y_true).mean()
    f1  = f1_score(y_true, preds_bin, zero_division=0)
    cm  = confusion_matrix(y_true, preds_bin)

    print(f"[{machine_id}] Test Accuracy : {acc:.4f}")
    print(f"[{machine_id}] Test F1 Score : {f1:.4f}")
    print(f"[{machine_id}] Confusion Matrix:\n{cm}")

    sent_mb = SENT_BYTES / (1024 * 1024)
    recv_mb = RECV_BYTES / (1024 * 1024)
    print(f"[{machine_id}] Sent: {sent_mb:.2f} MB | Received: {recv_mb:.2f} MB")

    torch.save(front.state_dict(), f"front_{machine_id}_ushape.pth")
    print(f"[{machine_id}] Front model weights saved.")

    torch.save(tail.state_dict(), f"tail_{machine_id}_ushape.pth")  # ← ADD THIS
    print(f"[{machine_id}] Tail model weights saved.")

    save_results(
        method     = "u_shape",
        machine_id = machine_id,
        acc        = acc,
        f1         = f1,
        time_taken = time.time() - start_time,  # ← Timing unchanged: starts after set_seed(), includes all
        sent_mb    = sent_mb,
        recv_mb    = recv_mb,
    )


if __name__ == "__main__":
    run_client()