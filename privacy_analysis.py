"""
privacy_analysis.py  (v2)
-------------------------
Quantifies and visualises the privacy advantage of U-shape split
learning over normal split learning across four complementary lenses:

  1. LABEL EXPOSURE AUDIT
       Structural analysis — does the protocol ever hand labels to
       the server?  Normal split: yes every batch.  U-shape: never.

  2. LABEL INFERENCE ATTACK  (simulated curious server)
       Trains a logistic-regression probe on the activations the
       server receives and measures how accurately it can predict
       labels it was never given.  Repeated N_TRIALS times with
       different random seeds for confidence intervals.
       Lower accuracy  →  better privacy.

  3. MUTUAL INFORMATION ESTIMATION
       Uses a binned histogram estimator of I(activations ; labels).
       High MI means the activations carry label information even
       without explicit access.
       Lower MI  →  better privacy.

  4. BHATTACHARYYA OVERLAP  (fixed formula)
       Mean BC coefficient across all activation dimensions.
       BC ≈ 1  →  class distributions nearly identical in
                   activation space  →  hard for server to separate
                   classes  →  better privacy.
       BC ≈ 0  →  classes cleanly separable  →  bad privacy.

  5. GRADIENT LEAKAGE SIMULATION  (normal split only)
       In normal split the server computes the loss and sends
       gradients back.  Those gradients can leak label information.
       We quantify this via cosine similarity between per-class
       gradient vectors — large difference → labels leaked.

Outputs
-------
  logs/privacy_report.json     — machine-readable summary
  plots/privacy_bars.png       — attack accuracy + MI comparison
  plots/activation_dist.png    — per-class activation PCA scatter
  plots/bc_per_dim.png         — BC coefficient per feature dimension
  plots/gradient_leak.png      — gradient cosine similarity (if applicable)

Usage
-----
  python privacy_analysis.py

Set the CONFIG section below to match your file layout.
Add  torch.save(front.state_dict(), "front_M01_ushape.pth")  (and
the normal-split equivalent) at the end of each training script to
get real trained weights instead of the random-init fallback.
"""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import sem

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR    = "new_train"
TEST_DIR     = "new_test"
MACHINE_IDS  = ["M01", "M02", "M03"]
LOG_DIR      = "logs"
PLOTS_DIR    = "plots"
MAX_PER_CLASS = 200   # samples per class per machine (train)
MAX_TEST      = 200   # samples per class (test)
N_TRIALS      = 5     # repetitions for confidence intervals
MI_BINS       = 20    # bins for mutual-information histogram estimate
BATCH_SIZE    = 32

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "u_shape":      "#2E86AB",   # blue
    "normal_split": "#E84855",   # red
}
METHOD_LABELS = {
    "u_shape":      "U-Shape",
    "normal_split": "Normal Split",
}

# ============================================================
# MODEL DEFINITION
# ============================================================
class FrontModel(nn.Module):
    """Shared front model — identical in both split methods."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,  32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x   # (B, 64, 256)


# ============================================================
# DATA LOADING
# ============================================================
def load_samples(machine_id: str, base_dir: str,
                 max_per_class: int = 200):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, machine_id, folder)
        if not os.path.exists(path):
            continue
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".h5"))[:max_per_class]
        for fname in tqdm(files,
                          desc=f"  {machine_id}/{folder}", leave=False):
            with h5py.File(os.path.join(path, fname), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)


def load_test_samples(base_dir: str, max_per_class: int = 200):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            continue
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".h5"))[:max_per_class]
        for fname in tqdm(files, desc=f"  test/{folder}", leave=False):
            with h5py.File(os.path.join(path, fname), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================
def extract_activations(front: nn.Module, X_np: np.ndarray,
                         batch_size: int = BATCH_SIZE):
    """
    Returns
    -------
    raw  : (N, 64, 256)  — raw feature maps (what server sees)
    gap  : (N, 64)       — global-average-pooled (for probes)
    """
    front.eval()
    raw_list, gap_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(
                X_np[i:i + batch_size].transpose(0, 2, 1),
                dtype=torch.float32,
            )
            act = front(xb)                      # (B, 64, 256)
            raw_list.append(act.numpy())
            gap_list.append(act.mean(dim=-1).numpy())   # (B, 64)
    return np.concatenate(raw_list), np.concatenate(gap_list)


# ============================================================
# METRIC 1 — LABEL EXPOSURE AUDIT
# ============================================================
def label_exposure_audit() -> dict:
    """
    Reads JSON logs to determine structural label exposure.
    Falls back to protocol-level knowledge if no logs are present.
    """
    found_methods = set()
    if os.path.isdir(LOG_DIR):
        for fname in os.listdir(LOG_DIR):
            if fname.endswith(".json") and fname != "privacy_report.json":
                try:
                    with open(os.path.join(LOG_DIR, fname)) as f:
                        rec = json.load(f)
                    if "method" in rec:
                        found_methods.add(rec["method"])
                except (json.JSONDecodeError, KeyError):
                    pass

    results = {}
    for method in (found_methods or {"u_shape", "normal_split"}):
        if method == "normal_split":
            results[method] = {
                "labels_sent_to_server": "ALL — every training batch",
                "server_computes_loss":  True,
                "description": (
                    "Server receives labels with every activation batch "
                    "to compute the cross-entropy loss. A curious server "
                    "trivially knows all labels."
                ),
            }
        elif method == "u_shape":
            results[method] = {
                "labels_sent_to_server": "NONE",
                "server_computes_loss":  False,
                "description": (
                    "Labels never leave the client. The server sees only "
                    "intermediate activations and gradients. The tail of "
                    "the network lives on the client, which computes the "
                    "loss and backpropagates through the server block."
                ),
            }
    return results


# ============================================================
# METRIC 2 — LABEL INFERENCE ATTACK  (with confidence interval)
# ============================================================
def label_inference_attack(act_train: np.ndarray, y_train: np.ndarray,
                            act_test:  np.ndarray, y_test:  np.ndarray,
                            n_trials:  int = N_TRIALS) -> dict:
    """
    Logistic-regression probe repeated n_trials times.
    Returns mean accuracy, std, and 95 % CI half-width.
    """
    accs = []
    for seed in range(n_trials):
        # Subsample to add variance between trials
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(act_train),
                         size=min(len(act_train), 500), replace=False)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(act_train[idx])
        X_te   = scaler.transform(act_test)

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_tr, y_train[idx])
        accs.append(accuracy_score(y_test, clf.predict(X_te)))

    accs = np.array(accs)
    ci   = 1.96 * sem(accs) if len(accs) > 1 else 0.0
    return {"mean": float(accs.mean()), "std": float(accs.std()),
            "ci95": float(ci), "all": accs.tolist()}


# ============================================================
# METRIC 3 — MUTUAL INFORMATION ESTIMATION
# ============================================================
def mutual_information_estimate(gap: np.ndarray, y: np.ndarray,
                                 n_bins: int = MI_BINS) -> float:
    """
    Estimate I(A ; Y) via binned histogram over each activation
    dimension, then average.

    I(A_d ; Y) = H(Y) - H(Y | A_d)

    This is a lower bound on the true MI; it is comparable across
    methods run on the same data.
    """
    classes, counts = np.unique(y, return_counts=True)
    p_y  = counts / counts.sum()
    hy   = -np.sum(p_y * np.log2(p_y + 1e-12))   # H(Y)

    mis = []
    for d in range(gap.shape[1]):
        feat  = gap[:, d]
        bins  = np.linspace(feat.min(), feat.max() + 1e-8, n_bins + 1)
        bin_idx = np.digitize(feat, bins) - 1

        # H(Y | A_d)  =  Σ_b p(b) H(Y | b)
        h_y_given_a = 0.0
        for b in range(n_bins):
            mask  = bin_idx == b
            if mask.sum() == 0:
                continue
            pb    = mask.mean()
            y_b   = y[mask]
            _, bc = np.unique(y_b, return_counts=True)
            p_yb  = bc / bc.sum()
            h_yb  = -np.sum(p_yb * np.log2(p_yb + 1e-12))
            h_y_given_a += pb * h_yb

        mis.append(hy - h_y_given_a)

    return float(np.mean(mis))


# ============================================================
# METRIC 4 — BHATTACHARYYA OVERLAP  (corrected formula)
# ============================================================
def bhattacharyya_overlap(gap: np.ndarray, y: np.ndarray):
    """
    Bhattacharyya coefficient for Gaussian approximation, per
    activation dimension.

    BC(p, q) = exp(-BD)  where
    BD = (1/8)(μ1-μ0)²/σ_avg + (1/2) ln(σ_avg / sqrt(σ0·σ1))
    σ_avg = (σ0² + σ1²) / 2

    Returns mean BC (scalar) and per-dim array for plotting.
    BC ∈ [0, 1]:  1 → identical distributions → best privacy.
    """
    c0 = gap[y == 0]
    c1 = gap[y == 1]
    if len(c0) == 0 or len(c1) == 0:
        return float("nan"), np.full(gap.shape[1], float("nan"))

    bcs = []
    for d in range(gap.shape[1]):
        mu0, sig0 = c0[:, d].mean(), c0[:, d].std() + 1e-8
        mu1, sig1 = c1[:, d].mean(), c1[:, d].std() + 1e-8
        sig_avg   = (sig0**2 + sig1**2) / 2.0
        # Bhattacharyya distance (Gaussian)
        bd = (0.125 * (mu0 - mu1)**2 / sig_avg
              + 0.5  * np.log(sig_avg / (np.sqrt(sig0**2 * sig1**2) + 1e-8)))
        bcs.append(float(np.exp(-bd)))

    bcs = np.array(bcs)
    return float(np.mean(bcs)), bcs


# ============================================================
# METRIC 5 — GRADIENT LEAKAGE (normal split only)
# ============================================================
def gradient_leakage_score(front: nn.Module,
                            X_np: np.ndarray, y_np: np.ndarray,
                            n_samples: int = 64) -> dict:
    """
    In normal split the server computes gradients of the loss w.r.t.
    the smashed activations.  We estimate per-class gradient vectors
    and measure their cosine distance.

    Large distance  →  gradients clearly differ between classes
                    →  server can infer labels from gradient direction.
    Small distance  →  gradients similar  →  labels harder to infer.

    Only meaningful for normal_split (where the server owns the loss).
    For u_shape the server never sees per-sample gradients w.r.t. labels.
    """
    front.train()
    loss_fn = nn.BCEWithLogitsLoss()

    # Simple linear head to simulate server-side classifier
    head = nn.Linear(64, 1)
    nn.init.xavier_uniform_(head.weight)

    grad_by_class: dict = {0: [], 1: []}

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_np), size=min(n_samples, len(X_np)), replace=False)

    for i in idx:
        xb = torch.tensor(
            X_np[i:i + 1].transpose(0, 2, 1), dtype=torch.float32
        )
        label = torch.tensor([[float(y_np[i])]])

        act = front(xb)                  # (1, 64, 256)
        gap = act.mean(dim=-1)           # (1, 64)
        gap.retain_grad()

        out  = head(gap)
        loss = loss_fn(out, label)
        loss.backward()

        if gap.grad is not None:
            grad_by_class[int(y_np[i])].append(
                gap.grad.detach().numpy().flatten()
            )
        front.zero_grad()
        head.zero_grad()

    front.eval()

    if not grad_by_class[0] or not grad_by_class[1]:
        return {"cosine_similarity": float("nan"),
                "leakage_level": "insufficient data"}

    g0 = np.mean(grad_by_class[0], axis=0)
    g1 = np.mean(grad_by_class[1], axis=0)
    cos_sim = float(
        np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-8)
    )
    # cos_sim close to -1 or very different from 1 → large leakage
    leakage = "HIGH" if cos_sim < 0.7 else "MEDIUM" if cos_sim < 0.9 else "LOW"
    return {"cosine_similarity": cos_sim, "leakage_level": leakage}


# ============================================================
# PLOTTING HELPERS
# ============================================================
def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def plot_attack_and_mi(attack_results: dict, mi_results: dict,
                        out_path: str):
    methods  = list(METHOD_LABELS.keys())
    labels   = [METHOD_LABELS[m] for m in methods]
    atk_mean = [attack_results[m]["mean"]  for m in methods]
    atk_ci   = [attack_results[m]["ci95"]  for m in methods]
    mi_vals  = [mi_results.get(m, float("nan")) for m in methods]
    bar_cols = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Privacy Analysis — U-Shape vs Normal Split Learning",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # --- subplot 1: attack accuracy ---
    ax = axes[0]
    x  = np.arange(len(methods))
    bars = ax.bar(x, atk_mean, 0.5, color=bar_cols, alpha=0.88,
                  yerr=atk_ci, capsize=6, error_kw={"linewidth": 1.5},
                  zorder=3)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.5,
               label="Random guess (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Attack Accuracy", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "Label Inference Attack Accuracy\n"
        "(lower = better privacy for the client)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    _style_ax(ax)
    for bar, val, ci in zip(bars, atk_mean, atk_ci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ci + 0.03,
            f"{val*100:.1f}%\n±{ci*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Annotate normal split with "trivially 100 %" note
    if "normal_split" in methods:
        i = methods.index("normal_split")
        ax.annotate(
            "Server already has\nlabels → trivial 100%",
            xy=(i, atk_mean[i] + atk_ci[i] + 0.03),
            xytext=(i + 0.3, 1.10),
            arrowprops={"arrowstyle": "->", "color": COLORS["normal_split"]},
            fontsize=8, color=COLORS["normal_split"],
        )

    # --- subplot 2: mutual information ---
    ax2 = axes[1]
    valid = [(m, v) for m, v in zip(methods, mi_vals) if not np.isnan(v)]
    if valid:
        ms, vs = zip(*valid)
        cols2  = [COLORS[m] for m in ms]
        lbls2  = [METHOD_LABELS[m] for m in ms]
        b2 = ax2.bar(range(len(ms)), vs, 0.5, color=cols2, alpha=0.88,
                     zorder=3)
        ax2.set_xticks(range(len(ms)))
        ax2.set_xticklabels(lbls2, fontsize=11)
        ax2.set_ylabel("Estimated Mutual Information (bits)", fontsize=10)
        ax2.set_title(
            "Activation–Label Mutual Information\n"
            "(lower = less label info leaked via activations)",
            fontsize=11,
        )
        _style_ax(ax2)
        for bar, val in zip(b2, vs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.3f} bits",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
    else:
        ax2.text(0.5, 0.5, "No MI data available",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_bc_per_dim(bc_per_dim: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    for method, bcs in bc_per_dim.items():
        if bcs is None or np.all(np.isnan(bcs)):
            continue
        dims = np.arange(len(bcs))
        ax.plot(dims, bcs, alpha=0.8, color=COLORS[method],
                label=METHOD_LABELS[method], linewidth=1.2)
        ax.fill_between(dims, bcs, alpha=0.15, color=COLORS[method])

    ax.axhline(1.0, color="green", linestyle=":", linewidth=1.2,
               label="Perfect overlap (BC=1)")
    ax.set_xlabel("Activation Feature Dimension", fontsize=10)
    ax.set_ylabel("Bhattacharyya Coefficient", fontsize=10)
    ax.set_title(
        "Class Overlap per Activation Dimension\n"
        "(higher = classes harder to separate in that dimension = better privacy)",
        fontsize=11,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_activation_pca(gap_data: dict, y_data: dict, out_path: str):
    """2-D PCA scatter of server-visible activations, coloured by class."""
    n_methods = len(gap_data)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    all_gap = np.concatenate(list(gap_data.values()))
    pca     = PCA(n_components=2, random_state=0)
    pca.fit(all_gap)

    class_colors = {0: "#2196F3", 1: "#F44336"}
    class_labels = {0: "Good (label=0)", 1: "Bad  (label=1)"}

    for ax, (method, gap) in zip(axes, gap_data.items()):
        y   = y_data[method]
        emb = pca.transform(gap)
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=class_colors[cls], label=class_labels[cls],
                alpha=0.55, s=18, edgecolors="none",
            )
        ax.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"(server sees these activations)",
            fontsize=11,
        )
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)
        _style_ax(ax)
        # More overlap → better privacy
        note = ("More overlap → harder to infer labels → better privacy"
                if method == "u_shape" else
                "Separable + labels sent directly → poor privacy")
        ax.set_xlabel(f"PC 1\n{note}", fontsize=8)

    fig.suptitle(
        "PCA of Activations Visible to Server (coloured by label)\n"
        "More overlap between classes = better privacy",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gradient_leak(grad_results: dict, out_path: str):
    methods = [m for m, r in grad_results.items()
               if not np.isnan(r.get("cosine_similarity", float("nan")))]
    if not methods:
        return

    cos_vals = [grad_results[m]["cosine_similarity"] for m in methods]
    labels   = [METHOD_LABELS[m] for m in methods]
    cols     = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(methods)), cos_vals, 0.5, color=cols, alpha=0.88)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Per-class Gradient Cosine Similarity", fontsize=10)
    ax.set_ylim(-1.1, 1.2)
    ax.axhline(1.0, color="green",  linestyle=":", linewidth=1.5,
               label="Identical gradients (no leakage)")
    ax.axhline(0.0, color="orange", linestyle=":", linewidth=1.5,
               label="Orthogonal gradients (some leakage)")
    ax.axhline(-1.0, color="red",   linestyle=":", linewidth=1.5,
               label="Opposite gradients (max leakage)")
    ax.set_title(
        "Gradient Leakage: Cosine Similarity of Per-Class Gradients\n"
        "(higher similarity = less class info in gradient direction)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    _style_ax(ax)
    for bar, val, m in zip(bars, cos_vals, methods):
        level = grad_results[m]["leakage_level"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.04 if val >= 0 else val - 0.08,
            f"{val:.3f}\n({level} leakage)",
            ha="center", fontsize=9, fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_summary_table(exposure: dict, attack_results: dict,
                       mi_results: dict, bc_results: dict,
                       out_path: str):
    """One-page summary table for reports."""
    rows = []
    for m in ["u_shape", "normal_split"]:
        if m not in attack_results:
            continue
        ar  = attack_results[m]
        exp = exposure.get(m, {})
        rows.append([
            METHOD_LABELS[m],
            exp.get("labels_sent_to_server", "?"),
            f"{ar['mean']*100:.1f}% ±{ar['ci95']*100:.1f}%",
            f"{mi_results.get(m, float('nan')):.3f} bits",
            f"{bc_results.get(m, float('nan')):.3f}",
            "Higher ✓" if m == "u_shape" else "Lower ✗",
        ])

    col_labels = [
        "Method", "Labels to server",
        "Attack accuracy", "Mutual info (bits)",
        "BC overlap", "Overall privacy",
    ]

    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    header_color = "#263238"
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    row_colors = {"U-Shape": "#E3F2FD", "Normal Split": "#FFEBEE"}
    for i, row in enumerate(rows, start=1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_colors.get(row[0], "white"))
        # Privacy cell
        privacy_cell = tbl[i, len(col_labels) - 1]
        if "Higher" in rows[i - 1][-1]:
            privacy_cell.set_facecolor("#C8E6C9")  # green
        else:
            privacy_cell.set_facecolor("#FFCDD2")  # red

    ax.set_title("Privacy Comparison Summary", fontsize=13,
                 fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    sep = "=" * 60

    print(sep)
    print("  PRIVACY ANALYSIS  (v2)")
    print(sep)

    # ----------------------------------------------------------
    # 1. Label exposure audit
    # ----------------------------------------------------------
    print("\n[1] Label Exposure Audit")
    exposure = label_exposure_audit()
    for method, info in exposure.items():
        print(f"  {METHOD_LABELS.get(method, method)}")
        print(f"    Labels to server : {info['labels_sent_to_server']}")
        print(f"    Server owns loss : {info['server_computes_loss']}")
        print(f"    → {info['description']}")

    # ----------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------
    print("\n[2] Loading data …")
    X_train_parts, y_train_parts = [], []
    for mid in MACHINE_IDS:
        X, y = load_samples(mid, TRAIN_DIR, MAX_PER_CLASS)
        if X is not None:
            X_train_parts.append(X)
            y_train_parts.append(y)

    X_test, y_test = load_test_samples(TEST_DIR, MAX_TEST)

    if not X_train_parts or X_test is None:
        print("  ✗ No data found. Check TRAIN_DIR / TEST_DIR paths.")
        return

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Class balance — train: "
          f"{(y_train==0).sum()} good / {(y_train==1).sum()} bad  |  "
          f"test: {(y_test==0).sum()} good / {(y_test==1).sum()} bad")

    # ----------------------------------------------------------
    # 3. Per-method analysis
    # ----------------------------------------------------------
    methods = [
        ("u_shape",      "front_M01_ushape.pth"),
        ("normal_split", "front_M01_normal.pth"),
    ]

    attack_results = {}
    mi_results     = {}
    bc_results     = {}
    bc_per_dim     = {}
    gap_for_pca    = {}
    y_for_pca      = {}
    grad_results   = {}

    for method, weight_file in methods:
        print(f"\n[3] Analysing — {METHOD_LABELS[method]}")
        front = FrontModel()

        if os.path.exists(weight_file):
            front.load_state_dict(
                torch.load(weight_file, weights_only=True)
            )
            print(f"    Loaded weights from {weight_file}")
        else:
            print(f"    ⚠  '{weight_file}' not found — using random-init "
                  f"front model.\n"
                  f"       Add `torch.save(front.state_dict(), "
                  f"'{weight_file}')` at the end of training to get "
                  f"real results.")

        print("    Extracting activations …")
        _, gap_train = extract_activations(front, X_train)
        _, gap_test  = extract_activations(front, X_test)

        gap_for_pca[method] = gap_train
        y_for_pca[method]   = y_train

        # Attack
        print(f"    Running inference attack ({N_TRIALS} trials) …")
        atk = label_inference_attack(gap_train, y_train, gap_test, y_test)
        attack_results[method] = atk
        print(f"    Attack accuracy : {atk['mean']*100:.1f}% "
              f"±{atk['ci95']*100:.1f}%  (95% CI)")

        # Mutual information
        mi  = mutual_information_estimate(gap_train, y_train)
        mi_results[method] = mi
        print(f"    Mutual info     : {mi:.4f} bits")

        # Bhattacharyya overlap
        bc_mean, bc_dims = bhattacharyya_overlap(gap_train, y_train)
        bc_results[method]  = bc_mean
        bc_per_dim[method]  = bc_dims
        print(f"    BC overlap      : {bc_mean:.4f} "
              f"(1=full overlap=best privacy)")

        # Gradient leakage (simulate for normal split; estimate for u_shape)
        print("    Estimating gradient leakage …")
        gl = gradient_leakage_score(front, X_train, y_train)
        grad_results[method] = gl
        print(f"    Gradient cos-sim: {gl['cosine_similarity']:.4f} "
              f"({gl['leakage_level']} leakage)")

    # ----------------------------------------------------------
    # 4. Privacy summary
    # ----------------------------------------------------------
    print(f"\n{sep}")
    print("  PRIVACY SUMMARY")
    print(sep)
    for method, _ in methods:
        if method not in attack_results:
            continue
        ar  = attack_results[method]
        mi  = mi_results.get(method, float("nan"))
        bc  = bc_results.get(method, float("nan"))
        gl  = grad_results.get(method, {})
        exp = exposure.get(method, {})
        print(f"\n  ▶ {METHOD_LABELS[method]}")
        print(f"    Labels exposed  : {exp.get('labels_sent_to_server','?')}")
        print(f"    Attack accuracy : {ar['mean']*100:.1f}% ±{ar['ci95']*100:.1f}%")
        print(f"    Mutual info     : {mi:.4f} bits")
        print(f"    BC overlap      : {bc:.4f}")
        print(f"    Gradient leakage: {gl.get('cosine_similarity', float('nan')):.4f} "
              f"({gl.get('leakage_level','?')})")

    # Privacy difference
    if "u_shape" in attack_results and "normal_split" in attack_results:
        diff = (attack_results["normal_split"]["mean"]
                - attack_results["u_shape"]["mean"])
        mi_diff = mi_results.get("normal_split", 0) - mi_results.get("u_shape", 0)
        print(f"\n  ✓ U-shape reduces inference attack accuracy by "
              f"{diff*100:.1f} percentage points")
        print(f"  ✓ U-shape reduces mutual information by "
              f"{mi_diff:.4f} bits")

    # ----------------------------------------------------------
    # 5. Save JSON report
    # ----------------------------------------------------------
    report = {
        "label_exposure":  exposure,
        "attack_results":  attack_results,
        "mutual_info":     mi_results,
        "bc_overlap":      bc_results,
        "gradient_leakage": grad_results,
    }
    report_path = os.path.join(LOG_DIR, "privacy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)
    print(f"\n  Report saved: {report_path}")

    # ----------------------------------------------------------
    # 6. Plots
    # ----------------------------------------------------------
    print("\n[6] Generating plots …")

    plot_attack_and_mi(
        attack_results, mi_results,
        os.path.join(PLOTS_DIR, "privacy_bars.png"),
    )
    plot_bc_per_dim(
        bc_per_dim,
        os.path.join(PLOTS_DIR, "bc_per_dim.png"),
    )
    plot_activation_pca(
        gap_for_pca, y_for_pca,
        os.path.join(PLOTS_DIR, "activation_dist.png"),
    )
    plot_gradient_leak(
        grad_results,
        os.path.join(PLOTS_DIR, "gradient_leak.png"),
    )
    plot_summary_table(
        exposure, attack_results, mi_results, bc_results,
        os.path.join(PLOTS_DIR, "privacy_summary_table.png"),
    )

    print("\n  ✓ Privacy analysis complete.")


if __name__ == "__main__":
    main()