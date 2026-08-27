"""
portal_app.py
-------------
Page 1 — Factory Portal: Login / Sign-up, Data Upload, How to Run.

Install:  pip install flask
Run:      python portal_app.py
Open:     http://127.0.0.1:5051
"""

import os
import json
import secrets
from flask import (
    Flask, render_template_string, request,
    redirect, url_for, session, jsonify,
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)   # change to a fixed value in production

# Simple flat-file "database" — stores factory credentials as JSON
FACTORY_DB = "factories.json"


def load_factories():
    if not os.path.exists(FACTORY_DB):
        return {}
    with open(FACTORY_DB) as f:
        return json.load(f)


def save_factories(db):
    with open(FACTORY_DB, "w") as f:
        json.dump(db, f, indent=2)


# ── HTML ──────────────────────────────────────────────────────────────────────

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
    --bg:      #f8f7f4;
    --surface: #ffffff;
    --border:  rgba(0,0,0,0.09);
    --text:    #1a1a1a;
    --muted:   #666;
    --accent:  #2563eb;
    --green:   #2a9d5c;
    --red:     #E84855;
    --radius:  10px;
  }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }
  .layout { display: flex; min-height: 100vh; }

  /* ── sidebar ── */
  .sidebar {
    width: 220px; min-width: 220px;
    background: var(--surface);
    border-right: 0.5px solid var(--border);
    display: flex; flex-direction: column;
    padding: 1.5rem 0;
    position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }
  .sidebar-logo {
    padding: 0 1.25rem 1.5rem;
    font-size: 15px; font-weight: 600; letter-spacing: -0.3px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 0.5px solid var(--border); margin-bottom: 0.75rem;
  }
  .sidebar-logo .dot {
    width: 26px; height: 26px; border-radius: 6px;
    background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 13px;
  }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.55rem 1.25rem;
    cursor: pointer; color: var(--muted); font-size: 13.5px;
    border-left: 3px solid transparent;
    transition: background 0.12s, color 0.12s;
  }
  .nav-item:hover  { background: var(--bg); color: var(--text); }
  .nav-item.active { color: var(--accent); background: #eff6ff; border-left-color: var(--accent); }
  .nav-item i { font-size: 17px; }
  .sidebar-footer {
    margin-top: auto; padding: 1rem 1.25rem;
    border-top: 0.5px solid var(--border);
    font-size: 12px; color: var(--muted);
  }
  .sidebar-footer strong { display: block; color: var(--text); margin-bottom: 2px; }
  .logout-btn {
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 6px; padding: 4px 10px;
    border: 0.5px solid var(--border); border-radius: 5px;
    font-size: 12px; cursor: pointer; background: none; color: var(--muted);
  }
  .logout-btn:hover { background: var(--bg); }

  /* ── main ── */
  .main { flex: 1; padding: 2rem 2.5rem; overflow: auto; max-width: 1100px; }
  .page { display: none; }
  .page.active { display: block; }
  h2 { font-size: 18px; font-weight: 600; margin-bottom: 0.25rem; }
  .page-sub { color: var(--muted); font-size: 13px; margin-bottom: 1.75rem; }

  /* ── cards ── */
  .card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .card-title { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 1rem; }

  /* ── auth form ── */
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
  input[type=text], input[type=password] {
    width: 100%; padding: 9px 12px; border: 0.5px solid var(--border);
    border-radius: 7px; font-size: 13.5px; background: var(--bg); color: var(--text);
    outline: none; transition: border-color 0.15s;
  }
  input[type=text]:focus, input[type=password]:focus { border-color: var(--accent); background: white; }
  .btn { display: block; width: 100%; padding: 10px; margin-top: 1.25rem; border-radius: 7px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { opacity: 0.88; }
  .flash { padding: 9px 12px; border-radius: 7px; font-size: 13px; margin-bottom: 1rem; }
  .flash-err  { background: #fee2e2; color: #991b1b; border: 0.5px solid #fecaca; }
  .flash-ok   { background: #d1fae5; color: #065f46; border: 0.5px solid #a7f3d0; }
  .auth-switch { text-align: center; margin-top: 1rem; font-size: 12.5px; color: var(--muted); }
  .auth-switch a { color: var(--accent); text-decoration: none; font-weight: 500; }

  /* ── terminal ── */
  .terminal {
    background: #1e1e2e; color: #cdd6f4;
    border-radius: 8px; padding: 1rem 1.25rem;
    font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.8;
    margin-bottom: 0.75rem; position: relative;
  }
  .terminal .comment { color: #6c7086; }
  .terminal .cmd     { color: #a6e3a1; }
  .terminal .arg     { color: #89b4fa; }
  .terminal .wait    { color: #f9e2af; }
  .terminal-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.5rem; padding-bottom: 0.5rem;
    border-bottom: 0.5px solid rgba(255,255,255,0.08);
  }
  .term-title { color: #cba6f7; font-size: 12px; font-weight: 600; }
  .copy-btn {
    background: rgba(255,255,255,0.08); color: #cdd6f4;
    border: none; border-radius: 5px; padding: 3px 10px;
    font-size: 11px; cursor: pointer; display: flex; align-items: center; gap: 5px;
  }
  .copy-btn:hover { background: rgba(255,255,255,0.15); }

  /* ── steps ── */
  .step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--accent); color: white; font-size: 11px; font-weight: 600;
    margin-right: 8px; flex-shrink: 0;
  }
  .step-row { display: flex; align-items: flex-start; gap: 0; margin-bottom: 0.75rem; }

  /* ── tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 1.25rem; }
  .tab {
    padding: 6px 18px; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 500;
    border: 0.5px solid var(--border);
    background: var(--surface); color: var(--muted); transition: all 0.12s;
  }
  .tab.active { background: var(--accent); color: white; border-color: var(--accent); }

  /* ── upload ── */
  .upload-zone {
    border: 1.5px dashed var(--border); border-radius: var(--radius);
    padding: 2rem; text-align: center; color: var(--muted);
    cursor: pointer; transition: border-color 0.15s, background 0.15s; margin-bottom: 1rem;
  }
  .upload-zone:hover { border-color: var(--accent); background: #eff6ff; }
  .upload-zone i { font-size: 36px; margin-bottom: 0.5rem; display: block; }
  .file-list { font-size: 12.5px; }
  .file-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 0.5px solid var(--border); gap: 8px; }
  .file-name { display: flex; align-items: center; gap: 6px; }
  .file-size { color: var(--muted); }
  .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 500; }
  .badge-done { background: #d1fae5; color: #065f46; }

  /* ── notice ── */
  .notice {
    background: #eff6ff; border: 0.5px solid #bfdbfe;
    border-radius: var(--radius); padding: 0.75rem 1rem;
    font-size: 13px; color: #1e40af; margin-bottom: 1.25rem;
    display: flex; align-items: flex-start; gap: 8px;
  }
  .notice i { font-size: 16px; margin-top: 1px; flex-shrink: 0; }

  /* ── comparison table ── */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: var(--bg); font-weight: 600; padding: 8px 12px; text-align: left; border-bottom: 0.5px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 0.5px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .better { color: var(--green); font-weight: 600; }
  .worse  { color: var(--red);   font-weight: 500; }

  /* ── misc ── */
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
  .empty i { font-size: 40px; display: block; margin-bottom: 0.75rem; opacity: 0.4; }
</style>
</head>
<body>

{% if not logged_in %}
<!-- ═══════════════ AUTH SCREEN ═══════════════ -->
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

    <!-- LOGIN -->
    <form id="form-login" action="/login" method="post" style="{{ '' if mode=='login' else 'display:none' }}">
      <label>Factory ID</label>
      <input type="text" name="factory_id" placeholder="e.g. M01" required>
      <label>Password</label>
      <input type="password" name="password" placeholder="••••••••" required>
      <button type="submit" class="btn btn-primary">Login</button>
    </form>

    <!-- SIGN UP -->
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
<!-- ═══════════════ PORTAL (logged in) ═══════════════ -->
<div class="layout">

<!-- Sidebar -->
<nav class="sidebar" aria-label="Portal navigation">
  <div class="sidebar-logo">
    <div class="dot"><i class="ti ti-brand-python" aria-hidden="true"></i></div>
    SplitLearn
  </div>
  <div class="nav-item active" onclick="showPage('upload')" data-page="upload">
    <i class="ti ti-database-plus" aria-hidden="true"></i> Upload data
  </div>
  <div class="nav-item" onclick="showPage('run')" data-page="run">
    <i class="ti ti-terminal-2" aria-hidden="true"></i> How to run
  </div>

  <div class="sidebar-footer">
    <strong>{{ factory_id }}</strong>
    Logged in as factory
    <form action="/logout" method="post" style="display:inline">
      <button class="logout-btn" type="submit"><i class="ti ti-logout" aria-hidden="true"></i> Logout</button>
    </form>
  </div>
</nav>

<!-- Main -->
<main class="main">

<!-- ─────────── UPLOAD ─────────── -->
<section class="page active" id="page-upload">
  <h2>New factory data</h2>
  <p class="page-sub">Add your machine client to the federation. Follow the steps below — no code changes needed.</p>

  <div class="card">
    <div class="card-title">Step 1 — Prepare your data folder</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:1rem;">Your H5 files must be organised as shown:</p>
    <div class="terminal">
      <span class="comment"># Required directory structure</span><br>
      new_train/<br>
      &nbsp;&nbsp;<span class="arg">{{ factory_id }}</span>/          <span class="comment">← your factory ID</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;good/  <span class="comment">*.h5  (normal vibration)</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;bad/   <span class="comment">*.h5  (faulty vibration)</span>
    </div>
    <p style="font-size:13px; color:var(--muted);">Each H5 file must contain a dataset named <code>vibration_data</code> with shape <code>(1024, 3)</code>.</p>
  </div>

  <div class="card">
    <div class="card-title">Step 2 — Upload H5 files</div>
    <div class="upload-zone" onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault(); this.style.borderColor='var(--accent)'"
         ondragleave="this.style.borderColor=''"
         ondrop="handleDrop(event)">
      <i class="ti ti-upload" aria-hidden="true"></i>
      <p style="font-weight:500; margin-bottom:4px;">Drop H5 files here or click to browse</p>
      <p style="font-size:12px;">Only <code>.h5</code> files accepted</p>
    </div>
    <input type="file" id="file-input" accept=".h5" multiple style="display:none" onchange="handleFiles(this.files)">
    <div id="file-list" class="file-list"></div>
  </div>

  <div class="card">
    <div class="card-title">Step 3 — Register and run</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:1rem;">
      Once your data is in place, update the server config and start a client terminal:
    </p>
    <div class="terminal">
      <span class="comment"># In server_split.py (or server_normal_split.py), update:</span><br>
      <span class="arg">NUM_CLIENTS</span> = <span class="cmd">{{ next_client_num }}</span><br>
      <br>
      <span class="comment"># Then start your client terminal:</span><br>
      <span class="cmd">python</span> <span class="arg">client_split.py</span> {{ factory_id }}
    </div>
    <p style="font-size:12px; color:var(--muted); margin-top:0.75rem;">
      <i class="ti ti-info-circle" aria-hidden="true"></i>
      The server will wait until all <code>NUM_CLIENTS</code> terminals connect before training begins.
    </p>
  </div>

  <div class="card">
    <div class="card-title">H5 path validator</div>
    <p style="font-size:13px; color:var(--muted); margin-bottom:0.75rem;">Paste a filename to check the expected shape:</p>
    <div style="display:flex; gap:8px; align-items:center;">
      <input type="text" id="validate-input" placeholder="e.g.  {{ factory_id }}/good/sample_001.h5"
             style="flex:1; padding:8px 12px; border:0.5px solid var(--border); border-radius:6px; font-size:13px; background:var(--bg);">
      <button onclick="validateShape()" style="padding:8px 16px; border:0.5px solid var(--accent); background:var(--accent); color:white; border-radius:6px; cursor:pointer; font-size:13px;">Check</button>
    </div>
    <div id="validate-result" style="margin-top:0.75rem; font-size:13px;"></div>
  </div>
</section>

<!-- ─────────── HOW TO RUN ─────────── -->
<section class="page" id="page-run">
  <h2>How to run</h2>
  <p class="page-sub">Open 4 terminals in your project directory and run the commands below in order.</p>

  <div class="notice">
    <i class="ti ti-info-circle" aria-hidden="true"></i>
    <span>Run <strong>normal split first</strong> — it creates the <code>indices/</code> folder that ensures both methods see identical data.</span>
  </div>

  <div class="tabs" id="run-tabs">
    <div class="tab active" onclick="switchRun('normal')">Normal split</div>
    <div class="tab" onclick="switchRun('ushape')">U-shape</div>
    <div class="tab" onclick="switchRun('analysis')">Analysis</div>
  </div>

  <!-- Normal split -->
  <div id="run-normal">
    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminal 1 — Start the server</div>
        <div class="terminal">
          <div class="terminal-header">
            <span class="term-title">server</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-server')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button>
          </div>
          <span id="cmd-ns-server"><span class="cmd">python</span> <span class="arg">server_normal_split.py</span></span>
          <br><span class="wait"># wait until: Listening on port 8888, waiting for 3 client(s)...</span>
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminals 2, 3, 4 — Start each client</div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M01</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m01')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-ns-m01"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M01</span>
        </div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M02</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m02')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-ns-m02"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M02</span>
        </div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M03</span>
            <button class="copy-btn" onclick="copyCmd('cmd-ns-m03')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-ns-m03"><span class="cmd">python</span> <span class="arg">client_normal_split.py</span> M03</span>
        </div>
      </div>
    </div>
  </div>

  <!-- U-shape -->
  <div id="run-ushape" style="display:none">
    <div class="notice">
      <i class="ti ti-alert-circle" aria-hidden="true"></i>
      <span>Keep the <code>indices/</code> folder from the previous run — U-shape reuses those shuffle indices for a fair comparison.</span>
    </div>
    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminal 1 — Start the server</div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">server</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-server')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-us-server"><span class="cmd">python</span> <span class="arg">server_split.py</span></span>
          <br><span class="wait"># wait until: Listening on port 8888, waiting for 3 client(s)...</span>
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Terminals 2, 3, 4 — Start each client</div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M01</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m01')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-us-m01"><span class="cmd">python</span> <span class="arg">client_split.py</span> M01</span>
        </div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M02</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m02')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-us-m02"><span class="cmd">python</span> <span class="arg">client_split.py</span> M02</span>
        </div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">client M03</span>
            <button class="copy-btn" onclick="copyCmd('cmd-us-m03')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-us-m03"><span class="cmd">python</span> <span class="arg">client_split.py</span> M03</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Analysis -->
  <div id="run-analysis" style="display:none">
    <div class="step-row">
      <span class="step-num">1</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Generate comparison CSVs and plots</div>
        <div class="terminal">
          <div class="terminal-header"><span class="term-title">any terminal</span>
            <button class="copy-btn" onclick="copyCmd('cmd-compare')"><i class="ti ti-copy" aria-hidden="true"></i> copy</button></div>
          <span id="cmd-compare"><span class="cmd">python</span> <span class="arg">auto_compare.py</span>
<span class="cmd">python</span> <span class="arg">auto_plot.py</span>
<span class="cmd">python</span> <span class="arg">privacy_analysis.py</span></span>
        </div>
      </div>
    </div>
    <div class="step-row">
      <span class="step-num">2</span>
      <div style="flex:1">
        <div style="font-weight:600; margin-bottom:6px;">Open the analysis dashboard</div>
        <p style="color:var(--muted); font-size:13px;">Start <code>python dashboard_app.py</code> and open
           <a href="http://127.0.0.1:5050" target="_blank" style="color:var(--accent)">http://127.0.0.1:5050</a>
           to view results, plots, and privacy analysis.</p>
      </div>
    </div>
  </div>

  <!-- Tradeoff table -->
  <div class="card" style="margin-top: 1.5rem;">
    <div class="card-title">Expected tradeoffs</div>
    <table>
      <thead><tr><th>Dimension</th><th>Normal split</th><th>U-shape</th></tr></thead>
      <tbody>
        <tr><td>Accuracy / F1</td><td>Similar</td><td>Similar</td></tr>
        <tr><td>Label privacy</td><td class="worse">❌ Labels sent every batch</td><td class="better">✅ Labels never leave client</td></tr>
        <tr><td>Communication</td><td class="better">✅ Lower</td><td class="worse">Higher (extra tail grad)</td></tr>
        <tr><td>Client compute</td><td class="better">✅ Lower (front only)</td><td class="worse">Higher (front + tail)</td></tr>
        <tr><td>Server compute</td><td class="worse">Higher (owns loss)</td><td class="better">✅ Lower (middle only)</td></tr>
        <tr><td>Overall verdict</td><td>Simpler but privacy-leaking</td><td class="better" style="color:var(--blue, #2E86AB)">✅ Better: same accuracy, stronger privacy</td></tr>
      </tbody>
    </table>
  </div>
</section>

</main>
</div>

<script>
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelector(`[data-page="${name}"]`).classList.add('active');
}

function switchRun(tab) {
  ['normal','ushape','analysis'].forEach(t => {
    document.getElementById('run-' + t).style.display = t === tab ? 'block' : 'none';
  });
  document.querySelectorAll('#run-tabs .tab').forEach((el, i) => {
    el.classList.toggle('active', ['normal','ushape','analysis'][i] === tab);
  });
}

function copyCmd(id) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.textContent.trim()).catch(() => {});
  const btn = el.closest('.terminal').querySelector('.copy-btn');
  const orig = btn.innerHTML;
  btn.innerHTML = '<i class="ti ti-check" aria-hidden="true"></i> copied';
  setTimeout(() => btn.innerHTML = orig, 1500);
}

function handleFiles(files) {
  const list = document.getElementById('file-list');
  const valid = Array.from(files).filter(f => f.name.endsWith('.h5'));
  if (!valid.length) { list.innerHTML = '<p style="color:#E84855; font-size:13px">No .h5 files selected.</p>'; return; }
  list.innerHTML = valid.map(f =>
    `<div class="file-row">
       <div class="file-name"><i class="ti ti-file-database" aria-hidden="true"></i> ${f.name}</div>
       <div class="file-size">${(f.size/1024).toFixed(1)} KB</div>
       <span class="badge badge-done">ready</span>
     </div>`
  ).join('');
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.borderColor = '';
  handleFiles(e.dataTransfer.files);
}

function validateShape() {
  const val = document.getElementById('validate-input').value.trim();
  const res = document.getElementById('validate-result');
  if (!val) { res.innerHTML = ''; return; }
  const ok = val.endsWith('.h5') && (val.includes('/good/') || val.includes('/bad/'));
  res.innerHTML = ok
    ? `<span style="color:#2a9d5c"><i class="ti ti-check" aria-hidden="true"></i> Path looks valid. Expected dataset shape: <code>(1024, 3)</code> float32.</span>`
    : `<span style="color:#E84855"><i class="ti ti-x" aria-hidden="true"></i> Path should end in <code>.h5</code> and be inside a <code>good/</code> or <code>bad/</code> subfolder.</span>`;
}
</script>
{% endif %}

</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    logged_in = "factory_id" in session
    factory_id = session.get("factory_id", "")
    db = load_factories()
    next_client_num = len(db) + 1
    return render_template_string(
        HTML,
        logged_in=logged_in,
        factory_id=factory_id,
        next_client_num=next_client_num,
        mode="login",
        flash_msg=None,
        flash_ok=False,
    )


@app.route("/login", methods=["POST"])
def login():
    fid = request.form.get("factory_id", "").strip().upper()
    pwd = request.form.get("password", "")
    db  = load_factories()
    if fid in db and db[fid] == pwd:
        session["factory_id"] = fid
        return redirect(url_for("index"))
    return render_template_string(
        HTML, logged_in=False, factory_id="", next_client_num=len(db)+1,
        mode="login", flash_msg="Invalid factory ID or password.", flash_ok=False,
    )


@app.route("/signup", methods=["POST"])
def signup():
    fid  = request.form.get("factory_id", "").strip().upper()
    pwd  = request.form.get("password", "")
    pwd2 = request.form.get("password2", "")
    db   = load_factories()

    if not fid:
        msg = "Factory ID cannot be empty."
    elif fid in db:
        msg = f"Factory ID '{fid}' is already registered."
    elif pwd != pwd2:
        msg = "Passwords do not match."
    elif len(pwd) < 4:
        msg = "Password must be at least 4 characters."
    else:
        db[fid] = pwd
        save_factories(db)
        session["factory_id"] = fid
        return redirect(url_for("index"))

    return render_template_string(
        HTML, logged_in=False, factory_id="", next_client_num=len(db)+1,
        mode="signup", flash_msg=msg, flash_ok=False,
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("=" * 55)
    print("  SplitLearn Factory Portal")
    print("  http://127.0.0.1:5051")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5051, debug=False)