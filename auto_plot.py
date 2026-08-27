"""
auto_plot.py
------------
Reads comparison_summary.csv (produced by auto_compare.py) and
logs/all_clients.csv, then generates a set of publication-quality
comparison plots saved to plots/.

Output files
------------
  plots/accuracy.png
  plots/f1.png
  plots/time.png
  plots/comm.png
  plots/radar.png
  plots/per_client_accuracy.png
  plots/per_client_f1.png
  plots/summary_grid.png   ← all four bar charts in one figure
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch

# ============================================================
# CONFIG
# ============================================================
SUMMARY_CSV    = "comparison_summary.csv"
ALL_CLIENTS_CSV = os.path.join("logs", "all_clients.csv")
PLOTS_DIR      = "plots"

# Colour palette — one per method (extended automatically if needed)
BASE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
for path in [SUMMARY_CSV]:
    if not os.path.exists(path):
        print(f"Error: '{path}' not found. Run auto_compare.py first.")
        sys.exit(1)

summary = pd.read_csv(SUMMARY_CSV)
methods = summary["method"].tolist()
n       = len(methods)
colors  = (BASE_COLORS * ((n // len(BASE_COLORS)) + 1))[:n]

# Per-client data (optional — used for error bars and strip plots)
has_clients = os.path.exists(ALL_CLIENTS_CSV)
clients_df  = pd.read_csv(ALL_CLIENTS_CSV) if has_clients else None

# ============================================================
# HELPER: compute per-method std for error bars
# ============================================================
def get_std(col):
    if clients_df is not None and col in clients_df.columns:
        return clients_df.groupby("method")[col].std().reindex(methods).fillna(0).values
    return np.zeros(n)

# ============================================================
# HELPER: styled bar chart
# ============================================================
def bar_chart(ax, values, yerr, ylabel, title, fmt="{:.3f}", pct=False):
    x = np.arange(n)
    bars = ax.bar(x, values, color=colors, width=0.55,
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.2, "color": "#555"},
                  zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels on bars
    for bar, val in zip(bars, values):
        label = (f"{val*100:.1f}%" if pct else fmt.format(val))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.02),
            label, ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Individual client dots (if available)
    if clients_df is not None:
        col_map = {
            "Accuracy":           "accuracy",
            "F1 Score":           "f1",
            "Training Time (s)":  "time",
            "Communication (MB)": "total_comm_MB",
        }
        col = col_map.get(ylabel)
        if col and col in clients_df.columns:
            for xi, method in enumerate(methods):
                pts = clients_df[clients_df["method"] == method][col].values
                ax.scatter(
                    np.full_like(pts, xi, dtype=float) +
                    np.random.uniform(-0.12, 0.12, len(pts)),
                    pts, color="black", s=22, zorder=5, alpha=0.6
                )

    # Set y-axis floor slightly below minimum
    floor = max(0, min(values) - max(values) * 0.15)
    ax.set_ylim(bottom=floor)
    return bars

# ============================================================
# INDIVIDUAL PLOTS
# ============================================================
metrics = [
    ("accuracy",      "Accuracy",           "Accuracy",           True,  "{:.3f}"),
    ("f1",            "F1 Score",           "F1 Score",           False, "{:.3f}"),
    ("time",          "Training Time (s)",  "Training Time (s)",  False, "{:.1f}"),
    ("total_comm_MB", "Communication (MB)", "Communication (MB)", False, "{:.2f}"),
]
fnames = ["accuracy", "f1", "time", "comm"]

for (col, ylabel, title, pct, fmt), fname in zip(metrics, fnames):
    if col not in summary.columns:
        print(f"  Skipping '{col}' — not in summary CSV.")
        continue
    values = summary[col].values
    yerr   = get_std(col)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.6), 4.5))
    bar_chart(ax, values, yerr, ylabel, title, fmt=fmt, pct=pct)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{fname}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ============================================================
# 2×2 SUMMARY GRID
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Split Learning — Method Comparison", fontsize=14, fontweight="bold", y=1.01)

for ax, (col, ylabel, title, pct, fmt) in zip(axes.flat, metrics):
    if col not in summary.columns:
        ax.set_visible(False)
        continue
    bar_chart(ax, summary[col].values, get_std(col), ylabel, title, fmt=fmt, pct=pct)

fig.tight_layout()
out = os.path.join(PLOTS_DIR, "summary_grid.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ============================================================
# RADAR / SPIDER CHART
# Normalises each metric to [0, 1] so all axes are comparable.
# For time and comm, lower is better so we invert the scale.
# ============================================================
radar_metrics = []
radar_labels  = []
invert_cols   = {"time", "total_comm_MB"}

for col, label, _, _, _ in metrics:
    if col in summary.columns:
        radar_metrics.append(col)
        radar_labels.append(label)

if len(radar_metrics) >= 3:
    # Normalise
    norm = summary[radar_metrics].copy()
    for col in radar_metrics:
        mn, mx = norm[col].min(), norm[col].max()
        if mx == mn:
            norm[col] = 1.0
        elif col in invert_cols:
            norm[col] = 1 - (norm[col] - mn) / (mx - mn)
        else:
            norm[col] = (norm[col] - mn) / (mx - mn)

    num_vars = len(radar_metrics)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_title("Method Comparison (normalised)\nHigher = better on all axes",
                 fontsize=11, fontweight="bold", pad=20)

    for i, (method, color) in enumerate(zip(methods, colors)):
        values_r = norm.iloc[i][radar_metrics].tolist()
        values_r += values_r[:1]
        ax.plot(angles, values_r, color=color, linewidth=2, label=method)
        ax.fill(angles, values_r, color=color, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ============================================================
# PER-CLIENT GROUPED BAR CHARTS  (accuracy and f1)
# ============================================================
if has_clients:
    for col, ylabel, title, pct, fmt in metrics[:2]:  # accuracy + f1 only
        if col not in clients_df.columns:
            continue

        client_ids = sorted(
            clients_df["client"]
            .dropna()
            .astype(str)
            .unique()
        )
        x          = np.arange(len(client_ids))
        width      = 0.8 / n

        fig, ax = plt.subplots(figsize=(max(6, len(client_ids) * 2), 4.5))
        for mi, (method, color) in enumerate(zip(methods, colors)):
            sub    = clients_df[clients_df["method"] == method].set_index("client")
            vals   = [sub.loc[c, col] if c in sub.index else 0.0 for c in client_ids]
            offset = (mi - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width=width * 0.9,
                            color=color, label=method, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(client_ids, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Per-Client {title}", fontsize=12, fontweight="bold", pad=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fname = "per_client_accuracy" if col == "accuracy" else "per_client_f1"
        out   = os.path.join(PLOTS_DIR, f"{fname}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

print(f"\nAll plots saved to '{PLOTS_DIR}/'")