"""
portal_app.py
-------------
MVP Factory Portal: Login / Sign-up, H5 Upload, Env Check, Auto-Run Client, Live Logs.
✅ FIXED: SSE race condition, Windows file-lock, scikit-learn import, unbuffered streaming.

Install:  pip install flask h5py
Run:      python portal_app.py
Open:     http://127.0.0.1:5051
"""

import os
import sys
import json
import secrets
import subprocess
import threading
import shutil
import time
from queue import Queue, Empty
from pathlib import Path
from flask import (
    Flask, render_template_string, request,
    redirect, url_for, session, jsonify, Response, stream_with_context
)
import h5py
import numpy as np

# ============================================================
# CONFIG & PATHS
# ============================================================
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "new_train"
DB_FILE = BASE_DIR / "factories.json"

REQUIRED_PACKAGES_PIP   = ["torch", "numpy", "h5py", "tqdm", "scikit-learn", "flask"]
REQUIRED_PACKAGES_IMPORT = ["torch", "numpy", "h5py", "tqdm", "sklearn", "flask"]

active_jobs = {}

# ============================================================
# FLASK LOGGING
# ============================================================
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)

@app.before_request
def log_request_info():
    app.logger.info(f"→ {request.method} {request.path}")

@app.after_request
def log_response_info(response):
    app.logger.info(f"← {response.status} for {request.path}")
    return response

# ============================================================
# DATABASE HELPERS
# ============================================================
def load_factories():
    if not DB_FILE.exists():
        return {}
    with open(DB_FILE) as f:
        return json.load(f)

def save_factories(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

# ============================================================
# HTML + CSS + JS
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SplitLearn — Factory Portal</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.30.0/dist/tabler-icons.min.css">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f8f7f4; --surface: #ffffff; --border: rgba(0,0,0,0.09);
    --text: #1a1a1a; --muted: #666; --accent: #2563eb;
    --green: #2a9d5c; --red: #E84855; --yellow: #f59e0b; --radius: 10px;
  }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6; }
  .layout { display: flex; min-height: 100vh; }
  .sidebar { width: 220px; min-width: 220px; background: var(--surface); border-right: 0.5px solid var(--border); display: flex; flex-direction: column; padding: 1.5rem 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; }
  .sidebar-logo { padding: 0 1.25rem 1.5rem; font-size: 15px; font-weight: 600; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px; border-bottom: 0.5px solid var(--border); margin-bottom: 0.75rem; }
  .sidebar-logo .dot { width: 26px; height: 26px; border-radius: 6px; background: var(--accent); display: flex; align-items: center; justify-content: center; color: white; font-size: 13px; }
  .nav-item { display: flex; align-items: center; gap: 10px; padding: 0.55rem 1.25rem; cursor: pointer; color: var(--muted); font-size: 13.5px; border-left: 3px solid transparent; transition: background 0.12s, color 0.12s; }
  .nav-item:hover  { background: var(--bg); color: var(--text); }
  .nav-item.active { color: var(--accent); background: #eff6ff; border-left-color: var(--accent); }
  .nav-item i { font-size: 17px; }
  .sidebar-footer { margin-top: auto; padding: 1rem 1.25rem; border-top: 0.5px solid var(--border); font-size: 12px; color: var(--muted); }
  .sidebar-footer strong { display: block; color: var(--text); margin-bottom: 2px; }
  .logout-btn { display: inline-flex; align-items: center; gap: 5px; margin-top: 6px; padding: 4px 10px; border: 0.5px solid var(--border); border-radius: 5px; font-size: 12px; cursor: pointer; background: none; color: var(--muted); }
  .logout-btn:hover { background: var(--bg); }
  .main { flex: 1; padding: 2rem 2.5rem; overflow: auto; max-width: 1100px; }
  .page { display: none; } .page.active { display: block; }
  h2 { font-size: 18px; font-weight: 600; margin-bottom: 0.25rem; }
  .page-sub { color: var(--muted); font-size: 13px; margin-bottom: 1.75rem; }
  .card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .card-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 1rem; }
  .auth-wrap { max-width: 440px; margin: 3rem auto; }
  .auth-card { background: var(--surface); border: 0.5px solid var(--border); border-radius: 14px; padding: 2rem 2.25rem; }
  .auth-logo { text-align: center; margin-bottom: 1.75rem; }
  .auth-logo .big-dot { width: 46px; height: 46px; border-radius: 12px; background: var(--accent); display: inline-flex; align-items: center; justify-content: center; color: white; font-size: 22px; margin-bottom: 0.5rem; }
  .auth-logo h1 { font-size: 18px; font-weight: 700; }
  .auth-logo p  { font-size: 13px; color: var(--muted); }
  .tabs-auth { display: flex; gap: 4px; background: var(--bg); border-radius: 8px; padding: 4px; margin-bottom: 1.5rem; }
  .tab-auth { flex: 1; text-align: center; padding: 7px 0; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; color: var(--muted); border: none; background: none; transition: all 0.12s; }
  .tab-auth.active { background: var(--surface); color: var(--text); box-shadow: 0 1px 4px rgba(0,0,0,0.10); }
  label { display: block; font-size: 12.5px; font-weight: 600; color: var(--muted); margin-bottom: 4px; margin-top: 0.85rem; }
  input[type=text], input[type=password] { width: 100%; padding: 9px 12px; border: 0.5px solid var(--border); border-radius: 7px; font-size: 13.5px; background: var(--bg); color: var(--text); outline: none; transition: border-color 0.15s; }
  input[type=text]:focus, input[type=password]:focus { border-color: var(--accent); background: white; }
  .btn { display: block; width: 100%; padding: 10px; margin-top: 1.25rem; border-radius: 7px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { opacity: 0.88; }
  .btn-sm { width: auto; display: inline-flex; padding: 6px 12px; font-size: 12.5px; margin-top: 0; }
  .flash { padding: 9px 12px; border-radius: 7px; font-size: 13px; margin-bottom: 1rem; }
  .flash-err  { background: #fee2e2; color: #991b1b; border: 0.5px solid #fecaca; }
  .flash-ok   { background: #d1fae5; color: #065f46; border: 0.5px solid #a7f3d0; }
  .auth-switch { text-align: center; margin-top: 1rem; font-size: 12.5px; color: var(--muted); }
  .auth-switch a { color: var(--accent); text-decoration: none; font-weight: 500; }
  .terminal { background: #1e1e2e; color: #cdd6f4; border-radius: 8px; padding: 1rem 1.25rem; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6; margin-bottom: 0.75rem; position: relative; max-height: 350px; overflow-y: auto; }
  .terminal .comment { color: #6c7086; } .terminal .cmd { color: #a6e3a1; } .terminal .arg { color: #89b4fa; } .terminal .wait { color: #f9e2af; }
  .terminal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 0.5px solid rgba(255,255,255,0.08); }
  .term-title { color: #cba6f7; font-size: 12px; font-weight: 600; }
  .copy-btn, .action-btn { background: rgba(255,255,255,0.08); color: #cdd6f4; border: none; border-radius: 5px; padding: 3px 10px; font-size: 11px; cursor: pointer; display: flex; align-items: center; gap: 5px; }
  .copy-btn:hover, .action-btn:hover { background: rgba(255,255,255,0.15); }
  .upload-zone { border: 1.5px dashed var(--border); border-radius: var(--radius); padding: 2rem; text-align: center; color: var(--muted); cursor: pointer; transition: border-color 0.15s, background 0.15s; margin-bottom: 1rem; }
  .upload-zone:hover { border-color: var(--accent); background: #eff6ff; }
  .upload-zone i { font-size: 36px; margin-bottom: 0.5rem; display: block; }
  .file-list { font-size: 12.5px; max-height: 200px; overflow-y: auto; }
  .file-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 0.5px solid var(--border); gap: 8px; }
  .file-name { display: flex; align-items: center; gap: 6px; }
  .file-size { color: var(--muted); }
  .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 500; }
  .badge-done { background: #d1fae5; color: #065f46; }
  .badge-err  { background: #fee2e2; color: #991b1b; }
  .status-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .status-ok   { background: #d1fae5; color: #065f46; }
  .status-warn { background: #fef3c7; color: #b45309; }
  .status-err  { background: #fee2e2; color: #991b1b; }
  .notice { background: #eff6ff; border: 0.5px solid #bfdbfe; border-radius: var(--radius); padding: 0.75rem 1rem; font-size: 13px; color: #1e40af; margin-bottom: 1.25rem; display: flex; align-items: flex-start; gap: 8px; }
  .notice i { font-size: 16px; margin-top: 1px; flex-shrink: 0; }
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
  .empty i { font-size: 40px; display: block; margin-bottom: 0.75rem; opacity: 0.4; }
  .radio-group { display: flex; gap: 16px; margin-bottom: 8px; }
  .radio-group label { font-weight: 400; color: var(--text); display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .btn-row { display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap; }
  .btn-success { background: var(--green); color: white; }
  .btn-danger { background: var(--red); color: white; }
  .btn-warn { background: var(--yellow); color: white; }
</style>
</head>
<body>

{% if not logged_in %}
<div class="auth-wrap">
  <div class="auth-card">
    <div class="auth-logo">
      <div class="big-dot"><i class="ti ti-brand-python" aria-hidden="true"></i></div>
      <h1>SplitLearn</h1>
      <p>Factory federated learning portal</p>
    </div>
    {% if flash_msg %}
      <div class="flash {{ 'flash-ok' if flash_ok else 'flash-err' }}">{{ flash_msg }}</div>
    {% endif %}
    <div class="tabs-auth">
      <button class="tab-auth {% if mode=='login' %}active{% endif %}" onclick="switchAuth('login')">Login</button>
      <button class="tab-auth {% if mode=='signup' %}active{% endif %}" onclick="switchAuth('signup')">Sign up</button>
    </div>
    <form id="form-login" action="/login" method="post" style="{{ '' if mode=='login' else 'display:none' }}">
      <label>Factory ID</label>
      <input type="text" name="factory_id" placeholder="e.g. M01" required>
      <label>Password</label>
      <input type="password" name="password" placeholder="••••••••" required>
      <button type="submit" class="btn btn-primary">Login</button>
    </form>
    <form id="form-signup" action="/signup" method="post" style="{{ '' if mode=='signup' else 'display:none' }}">
      <label>Factory ID <span style="color:var(--muted); font-weight:400">(new unique ID, e.g. M04)</span></label>
      <input type="text" name="factory_id" placeholder="e.g. M04" required>
      <label>Password</label>
      <input type="password" name="password" placeholder="Choose a password" required>
      <label>Confirm password</label>
      <input type="password" name="password2" placeholder="Repeat password" required>
      <button type="submit" class="btn btn-primary">Create account</button>
    </form>
  </div>
</div>
<script>
function switchAuth(mode) {
  document.querySelectorAll('.tab-auth').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('form-login').style.display  = mode === 'login'  ? '' : 'none';
  document.getElementById('form-signup').style.display = mode === 'signup' ? '' : 'none';
}
</script>
{% else %}
<div class="layout">
<nav class="sidebar" aria-label="Portal navigation">
  <div class="sidebar-logo">
    <div class="dot"><i class="ti ti-brand-python" aria-hidden="true"></i></div>
    SplitLearn
  </div>
  <div class="nav-item active" onclick="showPage('upload')" data-page="upload">
    <i class="ti ti-database-plus" aria-hidden="true"></i> Upload & Run
  </div>
  <div class="nav-item" onclick="showPage('run')" data-page="run">
    <i class="ti ti-terminal-2" aria-hidden="true"></i> Manual Guide
  </div>
  <div class="sidebar-footer">
    <strong>{{ factory_id }}</strong>
    Logged in as factory
    <form action="/logout" method="post" style="display:inline">
      <button class="logout-btn" type="submit"><i class="ti ti-logout" aria-hidden="true"></i> Logout</button>
    </form>
  </div>
</nav>

<main class="main">
<section class="page active" id="page-upload">
  <h2>Automated Client Runner</h2>
  <p class="page-sub">Upload your H5 data, verify your environment, and launch the client directly from this portal.</p>

  <div class="notice">
    <i class="ti ti-info-circle" aria-hidden="true"></i>
    <span>All operations run locally on this machine. Python & pip must be installed and accessible in PATH.</span>
  </div>

  <div class="card">
    <div class="card-title">Step 1 — Check Environment</div>
    <div id="env-status" style="margin-bottom:8px; font-size:13px; color:var(--muted);">Checking...</div>
    <div class="btn-row">
      <button class="btn btn-sm btn-primary" onclick="checkEnv()"><i class="ti ti-refresh"></i> Check Now</button>
      <button class="btn btn-sm btn-warn" id="install-btn" onclick="installDeps()" disabled><i class="ti ti-package"></i> Install Missing</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Step 2 — Upload H5 Data</div>
    <div class="radio-group">
      <label><input type="radio" name="label" value="good" checked> <strong>Good</strong> (Normal vibration)</label>
      <label><input type="radio" name="label" value="bad"> <strong>Bad</strong> (Faulty vibration)</label>
    </div>
    <div class="upload-zone" onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault(); this.style.borderColor='var(--accent)'"
         ondragleave="this.style.borderColor=''"
         ondrop="handleDrop(event)">
      <i class="ti ti-upload" aria-hidden="true"></i>
      <p style="font-weight:500; margin-bottom:4px;">Drop .h5 files here or click to browse</p>
      <p style="font-size:12px;">Expected shape: (1024, 3) float32</p>
    </div>
    <input type="file" id="file-input" accept=".h5" multiple style="display:none" onchange="handleFiles(this.files)">
    <div id="file-list" class="file-list"></div>
    <button class="btn btn-sm btn-success" id="upload-btn" onclick="uploadFiles()" disabled><i class="ti ti-cloud-upload"></i> Upload & Validate</button>
  </div>

  <div class="card">
    <div class="card-title">Step 3 — Run Client & Live Logs</div>
    <div class="btn-row" style="margin-bottom: 0.75rem;">
      <button class="btn btn-sm btn-primary" id="run-btn" onclick="startClient()"><i class="ti ti-player-play"></i> Run Client</button>
      <button class="btn btn-sm btn-danger" id="stop-btn" onclick="stopClient()" disabled><i class="ti ti-player-stop"></i> Stop</button>
    </div>
    
    <div style="margin-top: 0.5rem; font-size: 12px; color: var(--muted);">
      <details>
        <summary style="cursor: pointer; color: var(--accent);">🔧 Advanced: Direct Run (no JS)</summary>
        <div style="margin-top: 0.5rem; padding: 0.5rem; background: var(--bg); border-radius: 6px;">
          <p>Click to run client via direct form submit (bypasses JavaScript):</p>
          <form action="/run/{{ factory_id }}" method="post" target="_blank">
            <button type="submit" class="btn btn-sm" style="background: #64748b; color: white;">
              <i class="ti ti-external-link"></i> Run via Form Submit
            </button>
          </form>
        </div>
      </details>
    </div>
    
    <div class="terminal" id="log-terminal">
      <div class="terminal-header">
        <span class="term-title">client output</span>
        <span id="job-status" class="status-badge status-err">Idle</span>
      </div>
      <div id="log-content" style="white-space: pre-wrap;">Waiting for client to start...</div>
    </div>
  </div>
</section>

<section class="page" id="page-run">
  <h2>Manual Run Guide</h2>
  <p class="page-sub">Open terminals manually and run the commands below.</p>
  <div class="notice"><i class="ti ti-info-circle"></i> Run normal split first to generate shared <code>indices/</code> folder.</div>
  <div style="margin-top:1rem;">
    <div class="card">
      <div class="card-title">Normal Split</div>
      <div class="terminal"><span class="comment"># Terminal 1</span><br><span class="cmd">python</span> <span class="arg">server_normal_split.py</span></div>
      <div class="terminal"><span class="comment"># Terminals 2-4</span><br><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> {{ factory_id }}</div>
    </div>
    <div class="card">
      <div class="card-title">U-Shape Split</div>
      <div class="terminal"><span class="comment"># Terminal 1</span><br><span class="cmd">python</span> <span class="arg">server_split.py</span></div>
      <div class="terminal"><span class="comment"># Terminals 2-4</span><br><span class="cmd">python</span> <span class="arg">client_split.py</span> {{ factory_id }}</div>
    </div>
  </div>
</section>
</main>
</div>

<script>
const FACTORY_ID = "{{ factory_id }}";
let pendingFiles = [];
let eventSource = null;

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelector(`[data-page="${name}"]`).classList.add('active');
  if(name === 'upload') checkEnv();
}

function switchAuth(mode) {}

function handleFiles(files) {
  pendingFiles = Array.from(files).filter(f => f.name.endsWith('.h5'));
  const list = document.getElementById('file-list');
  const btn = document.getElementById('upload-btn');
  if (!pendingFiles.length) {
    list.innerHTML = '<p style="color:#E84855; font-size:13px">No .h5 files selected.</p>';
    btn.disabled = true; return;
  }
  list.innerHTML = pendingFiles.map(f =>
    `<div class="file-row">
       <div class="file-name"><i class="ti ti-file-database"></i> ${f.name}</div>
       <div class="file-size">${(f.size/1024).toFixed(1)} KB</div>
       <span class="badge badge-done">ready</span>
     </div>`
  ).join('');
  btn.disabled = false;
}

function handleDrop(e) {
  e.preventDefault(); e.currentTarget.style.borderColor = '';
  handleFiles(e.dataTransfer.files);
}

async function checkEnv() {
  const status = document.getElementById('env-status');
  const installBtn = document.getElementById('install-btn');
  status.innerHTML = 'Checking Python & packages...';
  try {
    const res = await fetch('/check-env');
    const data = await res.json();
    if (!data.ok) {
      status.innerHTML = `<span class="status-badge status-err">❌ Missing: ${data.missing.join(', ')}</span>`;
      installBtn.disabled = false;
    } else {
      status.innerHTML = `<span class="status-badge status-ok">✅ Python ${data.python} | All packages ready</span>`;
      installBtn.disabled = true;
    }
  } catch(e) { status.innerHTML = `<span class="status-badge status-err">⚠️ Check failed</span>`; }
}

async function installDeps() {
  const status = document.getElementById('env-status');
  status.innerHTML = 'Installing packages...';
  try {
    await fetch('/install-deps', {method: 'POST'});
    await checkEnv();
  } catch(e) { status.innerHTML = `<span class="status-badge status-err">❌ Install failed</span>`; }
}

async function uploadFiles() {
  const btn = document.getElementById('upload-btn');
  const list = document.getElementById('file-list');
  const label = document.querySelector('input[name="label"]:checked').value;
  
  btn.disabled = true; btn.innerHTML = '<i class="ti ti-loader"></i> Uploading...';
  list.innerHTML = '';

  for (const file of pendingFiles) {
    const fd = new FormData();
    fd.append('file', file); fd.append('label', label);
    try {
      const res = await fetch('/upload', {method: 'POST', body: fd});
      const data = await res.json();
      const color = data.ok ? '#065f46' : '#991b1b';
      const badge = data.ok ? 'badge-done' : 'badge-err';
      const msg = data.ok ? '✓ Uploaded & Validated' : `✗ ${data.error}`;
      list.innerHTML += `<div class="file-row"><div class="file-name"><i class="ti ti-file-database"></i> ${file.name}</div><div style="color:${color}; font-weight:500" class="${badge}">${msg}</div></div>`;
    } catch(e) {
      list.innerHTML += `<div class="file-row"><div class="file-name">${file.name}</div><span class="badge badge-err">✗ Network Error</span></div>`;
    }
  }
  btn.innerHTML = '<i class="ti ti-cloud-upload"></i> Upload More'; btn.disabled = false;
  document.getElementById('run-btn').disabled = false;
}

// ✅ FIXED: Race condition resolved by awaiting /run BEFORE opening EventSource
async function startClient() {
    console.log("[DEBUG] startClient() called");
    const btn = document.getElementById('run-btn');
    const stopBtn = document.getElementById('stop-btn');
    const status = document.getElementById('job-status');
    const log = document.getElementById('log-content');
    
    if (btn.disabled) { log.textContent = '❌ Button disabled.'; return; }
    
    btn.disabled = true; stopBtn.disabled = false;
    status.className = 'status-badge status-warn'; status.textContent = 'Starting...';
    log.textContent = '[Requesting server to start client...]\n';

    // 1️⃣ FIRST: Start the process via POST (guarantees job is registered)
    try {
        const res = await fetch(`/run/${FACTORY_ID}`, {method: 'POST'});
        const data = await res.json();
        console.log('[DEBUG] /run response:', data);
        if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    } catch (e) {
        log.textContent += `❌ Failed to start: ${e.message}\n`;
        status.className = 'status-badge status-err'; status.textContent = 'Failed';
        btn.disabled = false; stopBtn.disabled = true; return;
    }

    log.textContent += '[✓ Client process started. Connecting to log stream...]\n';

    // 2️⃣ SECOND: Connect SSE stream AFTER process is guaranteed to exist
    if(eventSource) eventSource.close();
    eventSource = new EventSource(`/stream/${FACTORY_ID}`);
    
    eventSource.onmessage = (e) => {
        console.log('[DEBUG] SSE message:', e.data.substring(0, 100));
        if(e.data.trim() === '[DONE]') {
            eventSource.close();
            status.className = 'status-badge status-ok'; status.textContent = 'Finished';
            btn.disabled = false; stopBtn.disabled = true;
        } else {
            log.textContent += e.data + '\n';
            document.getElementById('log-terminal').scrollTop = 99999;
        }
    };
    
    eventSource.onerror = (err) => {
        console.error('[DEBUG] Stream error:', err);
        if (status.textContent !== 'Finished') {
            status.className = 'status-badge status-err'; status.textContent = 'Disconnected';
            btn.disabled = false; stopBtn.disabled = true;
        }
        eventSource.close();
    };
}

async function stopClient() {
  await fetch(`/stop/${FACTORY_ID}`, {method: 'POST'});
}
</script>
{% endif %}
</body>
</html>
"""

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/ping")
def ping():
    return jsonify(ok=True, timestamp=time.time()), 200

@app.route("/")
def index():
    logged_in = "factory_id" in session
    return render_template_string(HTML, logged_in=logged_in, factory_id=session.get("factory_id", ""), mode="login", flash_msg=None, flash_ok=False)

@app.route("/login", methods=["POST"])
def login():
    fid = request.form.get("factory_id", "").strip().upper()
    pwd = request.form.get("password", "")
    db = load_factories()
    if fid in db and db[fid] == pwd:
        session["factory_id"] = fid; return redirect(url_for("index"))
    return render_template_string(HTML, logged_in=False, factory_id="", mode="login", flash_msg="Invalid factory ID or password.", flash_ok=False)

@app.route("/signup", methods=["POST"])
def signup():
    fid = request.form.get("factory_id", "").strip().upper()
    pwd = request.form.get("password", "")
    pwd2 = request.form.get("password2", "")
    db = load_factories()
    if not fid: msg = "Factory ID cannot be empty."
    elif fid in db: msg = f"Factory ID '{fid}' is already registered."
    elif pwd != pwd2: msg = "Passwords do not match."
    elif len(pwd) < 4: msg = "Password must be at least 4 characters."
    else:
        db[fid] = pwd; save_factories(db); session["factory_id"] = fid; return redirect(url_for("index"))
    return render_template_string(HTML, logged_in=False, factory_id="", mode="signup", flash_msg=msg, flash_ok=False)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear(); return redirect(url_for("index"))

@app.route("/check-env")
def check_env():
    import importlib
    missing = [pip_name for pip_name, imp_name in zip(REQUIRED_PACKAGES_PIP, REQUIRED_PACKAGES_IMPORT)
               if importlib.util.find_spec(imp_name) is None]
    return jsonify(ok=len(missing)==0, python=f"{sys.version_info.major}.{sys.version_info.minor}", missing=missing)

@app.route("/install-deps", methods=["POST"])
def install_deps():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES_PIP, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return jsonify(ok=True)
    except subprocess.CalledProcessError as e:
        return jsonify(ok=False, error=f"pip install failed: {e.stderr.strip()}"), 500
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    import time, tempfile, shutil
    fid = session.get("factory_id")
    if not fid: return jsonify(ok=False, error="Not logged in"), 401
    label = request.form.get("label", "good")
    file = request.files.get("file")
    if not file or not file.filename.endswith(".h5"): return jsonify(ok=False, error="Only .h5 files allowed"), 400

    target = DATA_DIR / fid / label
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5", dir=target) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        for attempt in range(3):
            try:
                with h5py.File(tmp_path, "r") as hf:
                    ds = hf["vibration_data"]
                break
            except OSError as e:
                if "being used by another process" in str(e).lower() and attempt < 2:
                    time.sleep(0.4); continue
                return jsonify(ok=False, error=f"Validation failed: {e}"), 500
        save_path = target / file.filename
        if save_path.exists(): save_path = target / f"{time.time_ns()}_{file.filename}"
        shutil.move(str(tmp_path), str(save_path))
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    finally:
        if tmp_path.exists(): tmp_path.unlink(missing_ok=True)

@app.route("/run/<fid>", methods=["GET", "POST"])
def run_client(fid):
    if fid != session.get("factory_id"): return jsonify(ok=False, error="Unauthorized"), 403
    if fid in active_jobs and active_jobs[fid]["proc"].poll() is None: return jsonify(ok=False, error="Already running"), 409

    client_script = BASE_DIR / "client_hybrid.py"
    if not client_script.exists(): client_script = BASE_DIR / "client_normal_split.py"
    if not client_script.exists(): return jsonify(ok=False, error="client_split.py not found"), 404

    cmd = [sys.executable, "-u", str(client_script), fid]
    print(f"[PORTAL DEBUG] Starting: {' '.join(cmd)}", flush=True)
    print(f"[PORTAL DEBUG] Working dir: {BASE_DIR}", flush=True)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(BASE_DIR), creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except Exception as e:
        return jsonify(ok=False, error=f"Failed to start: {e}"), 500

    q = Queue()
    def reader():
        try:
            for line in iter(proc.stdout.readline, ""):
                if line: q.put(line.rstrip())
        finally:
            proc.stdout.close()
            q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    active_jobs[fid] = {"proc": proc, "queue": q, "thread": None, "status": "running"}
    time.sleep(0.3)
    if request.method == "GET": return jsonify(ok=True, message=f"Started {fid} (form)")
    return jsonify(ok=True, message=f"Started {fid}")

@app.route("/stop/<fid>", methods=["POST"])
def stop_client(fid):
    job = active_jobs.pop(fid, None)
    if not job: return jsonify(ok=True, message="Not running")
    try: job["proc"].terminate(); job["proc"].wait(timeout=3)
    except: job["proc"].kill(); job["proc"].wait()
    return jsonify(ok=True, message="Stopped")

@app.route("/stream/<fid>")
def stream_logs(fid):
    if fid not in active_jobs:
        return Response("data: [ERROR] Job not found\n\n", mimetype="text/event-stream")
    
    def generate():
        q = active_jobs[fid]["queue"]
        proc = active_jobs[fid]["proc"]
        while True:
            try:
                line = q.get(timeout=2.0)
                if line is None:
                    yield " ── Client finished ──\n"
                    yield " [DONE]\n\n"
                    break
                # SSE requires "data: " prefix
                yield f"data: {line}\n\n"
            except Empty:
                if proc.poll() is not None:
                    yield "data: [Process exited unexpectedly]\n"
                    yield " [DONE]\n\n"
                    break
            except GeneratorExit:
                break
            except Exception as e:
                yield f"data: [Stream error: {e}]\n"
                yield " [DONE]\n\n"
                break
                
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

if __name__ == "__main__":
    print("=" * 55)
    print("  SplitLearn Factory Portal (MVP + Diagnostics)")
    print("  http://127.0.0.1:5051")
    print("  Data dir:", DATA_DIR)
    print("=" * 55)
    app.run(host="127.0.0.1", port=5051, debug=False, threaded=True)