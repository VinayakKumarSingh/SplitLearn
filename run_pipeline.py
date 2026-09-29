#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Convenience script to run any component of the SplitLearn system:
  1. Train U-Shape Split Learning (Server + 3 Clients)
  2. Train Normal Split Learning (Server + 3 Clients)
  3. Run Comparison & Plots (auto_compare.py & auto_plot.py)
  4. Run Privacy Analysis (privacy_analysis.py)
  5. Run Full End-to-End Pipeline (1 -> 2 -> 3 -> 4)
  6. Launch Analysis Dashboard (app.py on http://127.0.0.1:5050)
  7. Launch Factory Portal (portal_modified.py on http://127.0.0.1:5051)
  8. Launch Real-Time Inference Demo (flask_frontend_split.py on http://127.0.0.1:8080)
"""

import os
import sys
import time
import subprocess
import argparse

PYTHON = sys.executable

def run_ushape():
    print("\n" + "=" * 60)
    print("▶ 1. RUNNING U-SHAPE SPLIT LEARNING")
    print("=" * 60)
    server = subprocess.Popen([PYTHON, "server_split.py"])
    time.sleep(1)
    c1 = subprocess.Popen([PYTHON, "client_split.py", "M01"])
    c2 = subprocess.Popen([PYTHON, "client_split.py", "M02"])
    c3 = subprocess.Popen([PYTHON, "client_split.py", "M03"])
    
    c1.wait()
    c2.wait()
    c3.wait()
    server.wait()
    print("✅ U-Shape Split Learning finished successfully.")


def run_normal():
    print("\n" + "=" * 60)
    print("▶ 2. RUNNING NORMAL SPLIT LEARNING")
    print("=" * 60)
    server = subprocess.Popen([PYTHON, "server_normal_split.py"])
    time.sleep(1)
    c1 = subprocess.Popen([PYTHON, "client_normal_split.py", "M01"])
    c2 = subprocess.Popen([PYTHON, "client_normal_split.py", "M02"])
    c3 = subprocess.Popen([PYTHON, "client_normal_split.py", "M03"])
    
    c1.wait()
    c2.wait()
    c3.wait()
    server.wait()
    print("✅ Normal Split Learning finished successfully.")


def run_compare_and_plots():
    print("\n" + "=" * 60)
    print("▶ 3. GENERATING COMPARISONS & PLOTS")
    print("=" * 60)
    subprocess.run([PYTHON, "auto_compare.py"], check=True)
    subprocess.run([PYTHON, "auto_plot.py"], check=True)
    print("✅ Comparison tables and plots generated in logs/ and plots/.")


def run_privacy():
    print("\n" + "=" * 60)
    print("▶ 4. RUNNING PRIVACY ANALYSIS")
    print("=" * 60)
    subprocess.run([PYTHON, "privacy_analysis.py"], check=True)
    print("✅ Privacy report and plots saved.")


def run_full():
    run_ushape()
    run_normal()
    run_compare_and_plots()
    run_privacy()
    print("\n" + "=" * 60)
    print("🎉 FULL SPLITLEARN PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SplitLearn Execution Runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=["full", "ushape", "normal", "compare", "privacy", "dashboard", "portal", "inference"],
        help="Pipeline mode to execute (default: full)"
    )
    args = parser.parse_args()

    if args.mode == "ushape":
        run_ushape()
    elif args.mode == "normal":
        run_normal()
    elif args.mode == "compare":
        run_compare_and_plots()
    elif args.mode == "privacy":
        run_privacy()
    elif args.mode == "dashboard":
        print("Launching Analysis Dashboard on http://127.0.0.1:5050 ...")
        subprocess.run([PYTHON, "app.py"])
    elif args.mode == "portal":
        print("Launching Factory Portal on http://127.0.0.1:5051 ...")
        subprocess.run([PYTHON, "portal_modified.py"])
    elif args.mode == "inference":
        print("Launching Inference Service on http://127.0.0.1:8080 ...")
        subprocess.run([PYTHON, "flask_frontend_split.py"])
    else:
        run_full()


if __name__ == "__main__":
    main()
