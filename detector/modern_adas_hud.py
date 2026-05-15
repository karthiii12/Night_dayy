import sys
import os
import time
import threading
import queue
import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCER_DIR = os.path.join(BASE_DIR, "enhancer", "EnlightenGAN-inference")
WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "heavy_night_model2", "weights", "best.pt")
DEFAULT_VIDEO = os.path.join(BASE_DIR, "enhancer", "Test3.mp4")

# ── Settings ──────────────────────────────────────────────────────────────────
PROC_W, PROC_H   = 256, 192
OUT_W,  OUT_H    = 1280, 720
FRAME_STRIDE     = 4          # Detect every 4th frame — smoother than 2
FPS_WINDOW       = 30
DARK_THRESHOLD   = 30         # Mean pixel brightness below this → use gamma boost
GAMMA_VALUE      = 2.2        # Gamma correction for near-black frames

if ENHANCER_DIR not in sys.path:
    sys.path.append(ENHANCER_DIR)

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print(f"❌ Cannot import EnlightenGAN from: {ENHANCER_DIR}")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_hud_brackets(img, x1, y1, x2, y2, color, thickness=2, length=18):
    for (px, py), (dx, dy) in [
        ((x1,y1),( length,0)), ((x1,y1),(0, length)),
        ((x2,y1),(-length,0)), ((x2,y1),(0, length)),
        ((x1,y2),( length,0)), ((x1,y2),(0,-length)),
        ((x2,y2),(-length,0)), ((x2,y2),(0,-length)),
    ]:
        cv2.line(img, (px,py), (px+dx,py+dy), color, thickness)


def draw_text_bg(img, text, pos, scale=0.7, color=(255,255,255), thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x-6, y-th-6), (x+tw+6, y+6), (0,0,0), -1)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def gamma_boost(frame, gamma=GAMMA_VALUE):
    """Apply gamma correction to lift near-black frames before GAN."""
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def smart_enhance(enhancer, frame):
    """
    Brightness-aware enhancement:
    - Near-black frames → gamma boost first, then GAN
    - Sufficiently lit frames → GAN directly
    """
    mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if mean_brightness < DARK_THRESHOLD:
        frame = gamma_boost(frame)
    return enhancer.predict(frame)


# ── Pipeline worker thread ────────────────────────────────────────────────────
class PipelineWorker(threading.Thread):
    def __init__(self, enhancer, detector, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.enhancer     = enhancer
        self.detector     = detector
        self.frame_queue  = frame_queue
        self.result_queue = result_queue
        self.stop_event   = threading.Event()

    def run(self):
        frame_idx   = 0
        last_boxes  = []
        last_danger = False
        last_enhanced = None   # ← KEY: always keep last good enhanced frame

        while not self.stop_event.is_set():
            try:
                raw = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_idx += 1
            small = cv2.resize(raw, (PROC_W, PROC_H))

            if frame_idx % FRAME_STRIDE == 0 or last_enhanced is None:
                # Full pipeline: enhance + detect
                enhanced      = smart_enhance(self.enhancer, small)
                enhanced      = np.ascontiguousarray(enhanced)
                last_enhanced = enhanced.copy()

                results = self.detector(enhanced, imgsz=PROC_W, verbose=False)
                h, w    = enhanced.shape[:2]
                boxes   = []
                danger  = False
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ratio = (y2 - y1) / max(h, 1)
                    if ratio > 0.4:
                        color = (0,0,255); danger = True
                    elif ratio > 0.25:
                        color = (0,255,255)
                    else:
                        color = (0,255,0)
                    boxes.append((x1,y1,x2,y2,color))
                last_boxes  = boxes
                last_danger = danger
            else:
                # Non-detect frame: reuse last ENHANCED frame (no flicker)
                enhanced = last_enhanced.copy()

            # Draw boxes on enhanced frame
            hud = enhanced.copy()
            for (x1,y1,x2,y2,color) in last_boxes:
                draw_hud_brackets(hud, x1,y1,x2,y2,color)

            # Scale up to display resolution
            final = cv2.resize(hud, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

            # PiP: raw input thumbnail (top-right)
            pip_w, pip_h = 350, 200
            pip = cv2.resize(raw, (pip_w, pip_h))
            cv2.rectangle(pip, (0,0), (pip_w-1,pip_h-1), (255,255,255), 2)
            cv2.putText(pip, "RAW INPUT", (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            yo, xo = 24, OUT_W - pip_w - 24
            final[yo:yo+pip_h, xo:xo+pip_w] = pip

            # Push result (overwrite old if display can't keep up)
            try:
                self.result_queue.put_nowait((final, last_danger))
            except queue.Full:
                try:
                    self.result_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self.result_queue.put_nowait((final, last_danger))

    def stop(self):
        self.stop_event.set()


# ── Main display loop ─────────────────────────────────────────────────────────
def run_adas(video_source=DEFAULT_VIDEO, save_output=False):
    print("🚀 ASIET ADAS — Night Vision System")
    print(f"   Weights  : {WEIGHTS_PATH}")
    print(f"   Source   : {video_source}")
    print(f"   Dark gate: mean brightness < {DARK_THRESHOLD} → gamma boost first")

    enhancer = EnlightenOnnxModel()
    detector = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_source}"); return

    # Hint the capture buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    frame_queue  = queue.Queue(maxsize=6)
    result_queue = queue.Queue(maxsize=4)

    worker = PipelineWorker(enhancer, detector, frame_queue, result_queue)
    worker.start()

    writer    = None
    if save_output:
        out_path = os.path.join(BASE_DIR, "output_adas.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, 20.0, (OUT_W, OUT_H))
        print(f"💾 Recording → {out_path}")

    fps_times    = deque(maxlen=FPS_WINDOW)
    prev_time    = time.time()
    last_display = None     # ← holds last rendered frame, shown when queue empty

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            break

        try:
            frame_queue.put_nowait(raw)
        except queue.Full:
            pass

        # ── Get latest result — NEVER show blank, always fallback to last ──
        try:
            last_display, danger = result_queue.get_nowait()
        except queue.Empty:
            if last_display is None:
                continue          # first frames still loading
            danger = False        # reuse last danger state implicitly

        final = last_display.copy()

        # ── Rolling FPS ───────────────────────────────────────────────────
        now = time.time()
        fps_times.append(now - prev_time)
        prev_time = now
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

        # ── Overlay ───────────────────────────────────────────────────────
        draw_text_bg(final, "ASIET ADAS: NIGHT VISION", (40,60),  scale=1.0, color=(0,255,0),    thickness=2)
        draw_text_bg(final, f"FPS: {fps:.1f}",           (40,105), scale=0.8, color=(0,255,255),  thickness=2)
        draw_text_bg(final, "Q quit  |  S screenshot",   (40,145), scale=0.6, color=(200,200,200),thickness=1)

        # ── Danger banner ─────────────────────────────────────────────────
        if danger:
            overlay = final.copy()
            h_out   = final.shape[0]
            cv2.rectangle(overlay, (0, int(h_out*0.82)), (OUT_W,h_out), (0,0,255), -1)
            final = cv2.addWeighted(overlay, 0.45, final, 0.55, 0)
            cv2.putText(final, "WARNING: VEHICLE TOO CLOSE",
                        (60, h_out-35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)

        cv2.imshow("ADAS Night Vision HUD", final)
        if writer:
            writer.write(final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = int(time.time())
            spath = os.path.join(BASE_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(spath, final)
            print(f"📸 Screenshot → {spath}")

    worker.stop()
    cap.release()
    if writer:
        writer.release()
        print("✅ Output saved.")
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASIET ADAS Night Vision")
    parser.add_argument("--source", default="",          help="Video path or 0 for webcam")
    parser.add_argument("--save",   action="store_true", help="Save output as MP4")
    args = parser.parse_args()

    if args.source == "0":
        source = 0
    elif args.source:
        source = args.source
    else:
        print("\n=== ASIET ADAS Night Vision ===")
        print("ENTER → default video  |  0 → webcam  |  or type a path")
        u = input("Source: ").strip()
        source = 0 if u == "0" else (u if u else DEFAULT_VIDEO)

    run_adas(source, save_output=args.save)