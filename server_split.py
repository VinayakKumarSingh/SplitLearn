import socket
import pickle
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# CONFIG
# ============================================================
NUM_CLIENTS    = 3
PORT           = 8888
EPOCHS         = 3
SOCKET_TIMEOUT = 120  # seconds — per-client recv timeout

# ============================================================
# MIDDLE MODEL
#
# Receives stacked activations from all N clients: (N·B, 64, 256)
# Outputs combined embeddings:                     (N·B, 64)
# ============================================================
class MiddleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))   # (N·B, 128, 256)
        x = self.gap(x).squeeze(-1)   # (N·B, 128)
        x = self.relu2(self.fc(x))    # (N·B, 64)
        return x

# ============================================================
# SOCKET HELPERS
# ============================================================
def send_msg(sock, data):
    msg = pickle.dumps(data)
    sock.sendall(struct.pack(">I", len(msg)) + msg)

def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recvall(sock, msg_len)
    if data is None:
        return None
    return pickle.loads(data)

def recvall(sock, n):
    data = b""
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout:
            raise RuntimeError(
                f"[Server] Socket timed out waiting for {n - len(data)} more bytes "
                f"(received {len(data)}/{n})"
            )
        if not packet:
            return None
        data += packet
    return data

# ============================================================
# BATCH SIZE AUTO-SELECTION
# ============================================================
def auto_batch_size(min_samples: int) -> int:
    if min_samples < 100:
        return 8
    elif min_samples < 500:
        return 16
    elif min_samples < 2000:
        return 32
    elif min_samples < 5000:
        return 64
    else:
        return 128

# ============================================================
# SERVER MAIN
# ============================================================
def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", PORT))
    server_socket.listen(NUM_CLIENTS)
    print(f"[Server] Listening on port {PORT}, waiting for {NUM_CLIENTS} client(s)...")

    # ---- Accept all clients and gather handshake info ----
    client_sockets = {}
    counts = {}
    for _ in range(NUM_CLIENTS):
        conn, addr = server_socket.accept()
        conn.settimeout(SOCKET_TIMEOUT)
        msg = recv_msg(conn)
        if msg is None:
            raise RuntimeError("[Server] Received empty handshake.")
        cid   = msg["id"]
        count = msg["count"]
        client_sockets[cid] = conn
        counts[cid] = count
        print(f"[Server] Connected: {cid} @ {addr} | samples={count}")

    # ---- Decide batch size and effective samples ----
    min_samples = min(counts.values())
    batch_size  = auto_batch_size(min_samples)
    num_batches = min_samples // batch_size
    effective   = num_batches * batch_size

    if num_batches == 0:
        raise RuntimeError(
            f"[Server] Not enough samples to form even one batch "
            f"(min_samples={min_samples}, batch_size={batch_size})."
        )

    print(f"[Server] Auto batch_size={batch_size} | "
          f"effective_samples={effective} | num_batches={num_batches}")

    # Broadcast training config to every client
    cfg = {
        "num_batches":       num_batches,
        "batch_size":        batch_size,
        "effective_samples": effective,
        "epochs":            EPOCHS,
    }
    for conn in client_sockets.values():
        send_msg(conn, cfg)

    # ---- Init middle model and optimiser ----
    middle     = MiddleModel()
    opt_middle = optim.Adam(middle.parameters(), lr=0.001)
    client_order = sorted(client_sockets.keys())  # deterministic ordering

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(EPOCHS):
        print(f"\n[Server] === Epoch {epoch + 1}/{EPOCHS} ===")
        epoch_losses = []
        middle.train()

        for batch_idx in range(num_batches):

            # --------------------------------------------------
            # STEP 1: Collect one activation batch from every client
            #         Shape per client: (B, 64, 256)
            # --------------------------------------------------
            batch_acts = []
            for cid in client_order:
                msg = recv_msg(client_sockets[cid])
                if msg is None:
                    raise RuntimeError(f"[Server] Client {cid} disconnected (activation step).")
                batch_acts.append(msg["activation"])  # (B, 64, 256)

            # --------------------------------------------------
            # STEP 2: Stack → (N·B, 64, 256), forward through middle
            # act_tensor.requires_grad_(True) ensures its .grad is
            # populated after backward, giving the upstream gradient
            # to send back to the clients.
            # --------------------------------------------------
            combined_np = np.concatenate(batch_acts, axis=0)      # (N·B, 64, 256)
            act_tensor  = torch.tensor(combined_np, dtype=torch.float32, requires_grad=True)

            opt_middle.zero_grad()
            server_out    = middle(act_tensor)                     # (N·B, 64)
            server_out_np = server_out.detach().numpy()

            # --------------------------------------------------
            # STEP 3: Split output and send each chunk to its client
            # --------------------------------------------------
            split_outs = np.split(server_out_np, NUM_CLIENTS, axis=0)  # each (B, 64)
            for i, cid in enumerate(client_order):
                send_msg(client_sockets[cid], {"server_out": split_outs[i]})

            # --------------------------------------------------
            # STEP 4: Collect tail gradients ∂L/∂server_out
            #         Shape per client: (B, 64)
            # --------------------------------------------------
            batch_grads = []
            for cid in client_order:
                msg = recv_msg(client_sockets[cid])
                if msg is None:
                    raise RuntimeError(f"[Server] Client {cid} disconnected (tail-grad step).")
                batch_grads.append(msg["tail_grad"])  # (B, 64)
                epoch_losses.append(msg["loss"])

            # --------------------------------------------------
            # STEP 5: Stack gradients → (N·B, 64), backward through
            # middle, update middle weights.
            # Using server_out.backward(stacked_grads) applies the
            # chain rule correctly through the middle layers and
            # populates act_tensor.grad with ∂L/∂act.
            # --------------------------------------------------
            stacked_grads = torch.tensor(
                np.concatenate(batch_grads, axis=0), dtype=torch.float32
            )                                                       # (N·B, 64)
            server_out.backward(stacked_grads)
            opt_middle.step()

            # --------------------------------------------------
            # STEP 6: Send upstream gradient ∂L/∂act back to clients
            #         Shape: (N·B, 64, 256) → split into N × (B, 64, 256)
            # --------------------------------------------------
            upstream_np     = act_tensor.grad.detach().numpy()     # (N·B, 64, 256)
            split_upstreams = np.split(upstream_np, NUM_CLIENTS, axis=0)
            for i, cid in enumerate(client_order):
                send_msg(client_sockets[cid], {"upstream_grad": split_upstreams[i]})

        # --------------------------------------------------
        # EPOCH END: wait for all clients to confirm done
        # --------------------------------------------------
        for cid in client_order:
            msg = recv_msg(client_sockets[cid])
            if msg is None:
                raise RuntimeError(f"[Server] Client {cid} disconnected at epoch end.")

        avg_loss = float(np.mean(epoch_losses))
        print(f"[Server] Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")

    # ---- Save shared middle model ----
    torch.save(middle.state_dict(), "middle_model.pth")
    print("[Server] Middle model saved to middle_model.pth")

    # ============================================================
    # FINAL SYNC MESSAGE
    # ============================================================
    for cid in client_order:
        send_msg(client_sockets[cid], {
            "training_complete": True
        })

    print("[Server] Sent training completion signal to all clients.")

    # ---- Clean up ----
    for conn in client_sockets.values():
        conn.close()
    server_socket.close()
    print("[Server] Shutdown complete.")


if __name__ == "__main__":
    run_server()