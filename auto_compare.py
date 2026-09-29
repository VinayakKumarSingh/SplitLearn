"""
auto_compare.py
---------------
Reads all JSON log files from the logs/ directory, builds a per-client
table and a per-method summary, prints both, and saves two CSVs:
  - logs/all_clients.csv   : one row per (method, client)
  - comparison_summary.csv : one row per method (averages across clients)
"""

import os
import json
import pandas as pd

LOG_DIR = "logs"

# ============================================================
# LOAD ALL LOG FILES
# ============================================================
records = []
for filename in sorted(os.listdir(LOG_DIR)):
    if not filename.endswith(".json") or filename == "privacy_report.json":
        continue
    path = os.path.join(LOG_DIR, filename)
    try:
        with open(path) as f:
            record = json.load(f)
        if isinstance(record, dict) and "method" in record and "client" in record:
            records.append(record)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {path} — {e}")

if not records:
    print(f"No JSON log files found in '{LOG_DIR}/'. Run training first.")
    raise SystemExit(1)

df = pd.DataFrame(records)

# ============================================================
# NORMALISE SCHEMA
# Older logs may have a single "communication" column instead of
# "sent_MB" / "recv_MB" / "total_comm_MB".  Map them gracefully.
# ============================================================
if "communication" in df.columns and "total_comm_MB" not in df.columns:
    df.rename(columns={"communication": "total_comm_MB"}, inplace=True)
    df["sent_MB"] = float("nan")
    df["recv_MB"] = float("nan")

# Ensure all expected columns exist (fill with NaN if absent)
for col in ["sent_MB", "recv_MB", "total_comm_MB"]:
    if col not in df.columns:
        df[col] = float("nan")

# ============================================================
# PER-CLIENT TABLE
# ============================================================
client_cols = ["method", "client", "accuracy", "f1", "time",
               "sent_MB", "recv_MB", "total_comm_MB"]
client_cols = [c for c in client_cols if c in df.columns]
df_display  = df[client_cols].copy()

# Pretty-print formatting
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("\n" + "=" * 60)
print("  RAW CLIENT RESULTS")
print("=" * 60)
print(df_display.to_string(index=False))

# ============================================================
# PER-METHOD SUMMARY  (mean ± std across clients)
# ============================================================
agg_cols = {
    "accuracy":      ["mean", "std"],
    "f1":            ["mean", "std"],
    "time":          ["mean", "std"],
    "total_comm_MB": ["mean", "std"],
}
# Only aggregate columns that actually exist in the dataframe
agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

summary_multi = df.groupby("method").agg(agg_cols)
summary_multi.columns = ["_".join(c) for c in summary_multi.columns]
summary_multi = summary_multi.reset_index()

# Flat summary (mean only) for CSV / downstream plotting
summary_flat = df.groupby("method").agg(
    accuracy      = ("accuracy",      "mean"),
    f1            = ("f1",            "mean"),
    time          = ("time",          "mean"),
    total_comm_MB = ("total_comm_MB", "mean"),
).reset_index()

print("\n" + "=" * 60)
print("  AVERAGE COMPARISON  (mean ± std across clients)")
print("=" * 60)
print(summary_multi.to_string(index=False))

# ============================================================
# RANK METHODS
# ============================================================
ranked = summary_flat.copy()
ranked["accuracy_rank"]  = ranked["accuracy"].rank(ascending=False).astype(int)
ranked["f1_rank"]        = ranked["f1"].rank(ascending=False).astype(int)
ranked["time_rank"]      = ranked["time"].rank(ascending=True).astype(int)
ranked["comm_rank"]      = ranked["total_comm_MB"].rank(ascending=True).astype(int)
ranked["overall_rank"]   = (
    ranked["accuracy_rank"] + ranked["f1_rank"] +
    ranked["time_rank"]     + ranked["comm_rank"]
).rank(ascending=True).astype(int)

print("\n" + "=" * 60)
print("  RANKINGS  (lower = better)")
print("=" * 60)
rank_cols = ["method", "accuracy_rank", "f1_rank", "time_rank",
             "comm_rank", "overall_rank"]
print(ranked[rank_cols].sort_values("overall_rank").to_string(index=False))

# ============================================================
# SAVE CSVs
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
all_clients_path = os.path.join(LOG_DIR, "all_clients.csv")
summary_path     = "comparison_summary.csv"

df_display.to_csv(all_clients_path, index=False)
summary_flat.to_csv(summary_path,   index=False)

print(f"\nSaved: {all_clients_path}")
print(f"Saved: {summary_path}")