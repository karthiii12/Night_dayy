import sys
import os
import time
import threading
import subprocess
import queue
import uuid
import sqlite3
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCER_DIR = os.path.join(BASE_DIR, "enhancer", "EnlightenGAN-inference")
WEIGHTS_PATH = os.path.join(BASE_DIR, "detector", "best.pt")
UPLOAD_DIR   = os.path.join(BASE_DIR, "webapp", "static", "uploads")
OUTPUT_DIR   = os.path.join(BASE_DIR, "webapp", "static", "outputs")
DB_PATH      = os.path.join(BASE_DIR, "webapp", "adas_logs.db")

if ENHANCER_DIR not in sys.path:
    sys.path.append(ENHANCER_DIR)

from enlighten_inference import EnlightenOnnxModel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# ── Load models once at startup ───────────────────────────────────────────────
print("🔄 Loading models...")
enhancer = EnlightenOnnxModel()
detector = YOLO(WEIGHTS_PATH)
print("✅ Models ready.")

# ── Job state ─────────────────────────────────────────────────────────────────
jobs = {}   # job_id → { status, progress, stats, output_filename }

PROC_W, PROC_H = 256, 192
DARK_THRESHOLD = 30
GAMMA_VALUE    = 2.2

# ── Webcam process state ──────────────────────────────────────────────────────
webcam_process = None


# ── SQLite setup ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT,
            timestamp   TEXT,
            frame_no    INTEGER,
            object_cls  TEXT,
            confidence  REAL,
            severity    TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            job_id           TEXT PRIMARY KEY,
            filename         TEXT,
            started_at       TEXT,
            completed_at     TEXT,
            total_frames     INTEGER,
            total_detections INTEGER,
            danger_frames    INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_incident(job_id, frame_no, object_cls, confidence, severity):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            INSERT INTO incidents (job_id, timestamp, frame_no, object_cls, confidence, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            frame_no,
            object_cls,
            round(confidence, 3),
            severity
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠ DB log error: {e}")


def log_session(job_id, filename, started_at, total_frames, total_detections, danger_frames):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO sessions
            (job_id, filename, started_at, completed_at, total_frames, total_detections, danger_frames)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            filename,
            started_at,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            total_frames,
            total_detections,
            danger_frames
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠ DB session error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def gamma_boost(frame, gamma=GAMMA_VALUE):
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def smart_enhance(frame):
    mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if mean_brightness < DARK_THRESHOLD:
        frame = gamma_boost(frame)
    return enhancer.predict(frame)


def draw_hud_brackets(img, x1, y1, x2, y2, color, thickness=2, length=15):
    for (px, py), (dx, dy) in [
        ((x1,y1),( length,0)), ((x1,y1),(0, length)),
        ((x2,y1),(-length,0)), ((x2,y1),(0, length)),
        ((x1,y2),( length,0)), ((x1,y2),(0,-length)),
        ((x2,y2),(-length,0)), ((x2,y2),(0,-length)),
    ]:
        cv2.line(img, (px,py), (px+dx,py+dy), color, thickness)


# ── Video processing job (runs in background thread) ─────────────────────────
def process_video_job(job_id, input_path, output_filename):
    jobs[job_id]["status"] = "processing"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    cap          = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0

    temp_filename = output_filename.replace(".mp4", "_tmp.mp4")
    temp_path     = os.path.join(OUTPUT_DIR, temp_filename)
    out_path      = os.path.join(OUTPUT_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps_in, (PROC_W * 2, PROC_H))

    stats = {
        "total_frames":     0,
        "total_detections": 0,
        "danger_frames":    0,
        "class_counts":     {},
        "fps_avg":          0.0,
    }
    fps_samples = []
    frame_idx   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0    = time.time()
        small = cv2.resize(frame, (PROC_W, PROC_H))

        enhanced = smart_enhance(small)
        enhanced = np.ascontiguousarray(enhanced)
        results  = detector(enhanced, imgsz=PROC_W, verbose=False)

        h, w   = enhanced.shape[:2]
        hud    = enhanced.copy()
        danger = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ratio      = (y2 - y1) / max(h, 1)
            conf_score = float(box.conf[0])

            if ratio > 0.4:
                color  = (0, 0, 255)
                danger = True
            elif ratio > 0.25:
                color  = (0, 255, 255)
            else:
                color  = (0, 255, 0)

            draw_hud_brackets(hud, x1, y1, x2, y2, color)

            cls_id   = int(box.cls[0])
            cls_name = detector.names.get(cls_id, str(cls_id))
            stats["class_counts"][cls_name] = \
                stats["class_counts"].get(cls_name, 0) + 1

            # ── Log danger incidents to SQLite ────────────────────────────────
            if ratio > 0.4:
                log_incident(
                    job_id     = job_id,
                    frame_no   = frame_idx,
                    object_cls = cls_name,
                    confidence = conf_score,
                    severity   = "CRITICAL" if ratio > 0.55 else "HIGH"
                )

        stats["total_detections"] += len(results[0].boxes)
        if danger:
            stats["danger_frames"] += 1

        side = np.hstack([small, hud])
        writer.write(side)

        fps_samples.append(1.0 / max(time.time() - t0, 0.001))
        stats["total_frames"]    = frame_idx
        jobs[job_id]["progress"] = int((frame_idx / max(total_frames, 1)) * 100)

    cap.release()
    writer.release()

    # ── Re-encode to H.264 for browser ───────────────────────────────────────
    jobs[job_id]["progress"] = 99
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i",       temp_path,
            "-vcodec",  "libx264",
            "-pix_fmt", "yuv420p",
            "-preset",  "fast",
            "-crf",     "23",
            out_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"⚠ ffmpeg re-encode failed:\n{result.stderr}")
            os.replace(temp_path, out_path)
        else:
            os.remove(temp_path)
            print(f"✅ Re-encoded to H.264 → {out_path}")

    except FileNotFoundError:
        print("⚠ ffmpeg not found — serving raw mp4v.")
        os.replace(temp_path, out_path)

    # ── Log completed session to SQLite ──────────────────────────────────────
    log_session(
        job_id           = job_id,
        filename         = os.path.basename(input_path),
        started_at       = started_at,
        total_frames     = stats["total_frames"],
        total_detections = stats["total_detections"],
        danger_frames    = stats["danger_frames"]
    )

    stats["fps_avg"]                = round(sum(fps_samples) / max(len(fps_samples), 1), 1)
    jobs[job_id]["status"]          = "done"
    jobs[job_id]["stats"]           = stats
    jobs[job_id]["output_filename"] = output_filename


# ── Comparison route — GAN vs Gamma vs Raw ────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400

    SAMPLE_COUNT = 10

    f        = request.files["video"]
    tmp_id   = str(uuid.uuid4())[:8]
    tmp_name = f"cmp_{tmp_id}.mp4"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    f.save(tmp_path)

    cap          = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        return jsonify({"error": "Could not read video"}), 400

    margin  = max(1, int(total_frames * 0.05))
    indices = [
        margin + int(i * (total_frames - 2 * margin) / (SAMPLE_COUNT - 1))
        for i in range(SAMPLE_COUNT)
    ]

    results_raw   = []
    results_gamma = []
    results_gan   = []
    frame_labels  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (PROC_W, PROC_H))
        frame_labels.append(idx)

        # 1. Raw
        r_raw    = detector(np.ascontiguousarray(small), imgsz=PROC_W, verbose=False, conf=0.25)
        raw_det  = len(r_raw[0].boxes)
        raw_conf = float(r_raw[0].boxes.conf.mean()) if raw_det > 0 else 0.0
        results_raw.append({"det": raw_det, "conf": round(raw_conf, 3)})

        # 2. Gamma
        gamma_frame  = gamma_boost(small, gamma=GAMMA_VALUE)
        r_gamma      = detector(np.ascontiguousarray(gamma_frame), imgsz=PROC_W, verbose=False, conf=0.25)
        gamma_det    = len(r_gamma[0].boxes)
        gamma_conf   = float(r_gamma[0].boxes.conf.mean()) if gamma_det > 0 else 0.0
        results_gamma.append({"det": gamma_det, "conf": round(gamma_conf, 3)})

        # 3. EnlightenGAN
        gan_frame = enhancer.predict(small)
        gan_frame = np.ascontiguousarray(gan_frame)
        r_gan     = detector(gan_frame, imgsz=PROC_W, verbose=False, conf=0.15)
        gan_det   = len(r_gan[0].boxes)
        gan_conf  = float(r_gan[0].boxes.conf.mean()) if gan_det > 0 else 0.0
        results_gan.append({"det": gan_det, "conf": round(gan_conf, 3)})

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    total_raw   = sum(r["det"]  for r in results_raw)
    total_gamma = sum(r["det"]  for r in results_gamma)
    total_gan   = sum(r["det"]  for r in results_gan)

    avg_conf_raw   = round(sum(r["conf"] for r in results_raw)   / max(len(results_raw),   1), 3)
    avg_conf_gamma = round(sum(r["conf"] for r in results_gamma) / max(len(results_gamma), 1), 3)
    avg_conf_gan   = round(sum(r["conf"] for r in results_gan)   / max(len(results_gan),   1), 3)

    def improvement(baseline, enhanced):
        if baseline == 0:
            return 100 if enhanced > 0 else 0
        return round((enhanced - baseline) / baseline * 100, 1)

    return jsonify({
        "frame_labels":  frame_labels,
        "results_raw":   [r["det"]  for r in results_raw],
        "results_gamma": [r["det"]  for r in results_gamma],
        "results_gan":   [r["det"]  for r in results_gan],
        "conf_raw":      [r["conf"] for r in results_raw],
        "conf_gamma":    [r["conf"] for r in results_gamma],
        "conf_gan":      [r["conf"] for r in results_gan],
        "totals": {
            "raw":   total_raw,
            "gamma": total_gamma,
            "gan":   total_gan,
        },
        "avg_conf": {
            "raw":   avg_conf_raw,
            "gamma": avg_conf_gamma,
            "gan":   avg_conf_gan,
        },
        "improvement": {
            "gamma_vs_raw":      improvement(total_raw,    total_gamma),
            "gan_vs_raw":        improvement(total_raw,    total_gan),
            "gan_vs_gamma":      improvement(total_gamma,  total_gan),
            "conf_gan_vs_raw":   improvement(avg_conf_raw, avg_conf_gan),
            "conf_gan_vs_gamma": improvement(avg_conf_gamma, avg_conf_gan),
        }
    })


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400

    f        = request.files["video"]
    job_id   = str(uuid.uuid4())[:8]
    ext      = os.path.splitext(f.filename)[1] or ".mp4"
    in_name  = f"input_{job_id}{ext}"
    out_name = f"output_{job_id}.mp4"

    in_path  = os.path.join(UPLOAD_DIR, in_name)
    f.save(in_path)

    jobs[job_id] = {
        "status":          "queued",
        "progress":        0,
        "stats":           None,
        "output_filename": out_name,
        "input_filename":  in_name,
    }

    t = threading.Thread(
        target=process_video_job,
        args=(job_id, in_path, out_name),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/static/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ── Incident log routes ───────────────────────────────────────────────────────
@app.route("/incidents")
def get_incidents():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 50")
        incidents = [dict(r) for r in c.fetchall()]

        c.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 20")
        sessions = [dict(r) for r in c.fetchall()]

        conn.close()
        return jsonify({"incidents": incidents, "sessions": sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/incidents/clear", methods=["POST"])
def clear_incidents():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM incidents")
        conn.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── Webcam routes ─────────────────────────────────────────────────────────────
@app.route("/launch_webcam", methods=["POST"])
def launch_webcam():
    global webcam_process

    if webcam_process and webcam_process.poll() is None:
        webcam_process.terminate()
        webcam_process.wait()

    try:
        launcher   = os.path.join(BASE_DIR, "detector", "launch_webcam.py")
        python_exe = sys.executable

        webcam_process = subprocess.Popen(
    [python_exe, launcher],
    cwd=BASE_DIR,
    **({'creationflags': subprocess.CREATE_NEW_CONSOLE} if os.name == 'nt' else {})
)
        return jsonify({"ok": True, "pid": webcam_process.pid})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/webcam_status")
def webcam_status():
    global webcam_process
    if webcam_process is None:
        return jsonify({"state": "idle"})
    poll = webcam_process.poll()
    if poll is None:
        return jsonify({"state": "running", "pid": webcam_process.pid})
    else:
        return jsonify({"state": "stopped", "exit_code": poll})


@app.route("/stop_webcam", methods=["POST"])
def stop_webcam_route():
    global webcam_process
    if webcam_process and webcam_process.poll() is None:
        webcam_process.terminate()
        webcam_process.wait()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "No active webcam process"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()
    print("✅ Database initialized.")
    app.run(debug=False, host="0.0.0.0", port=5000)