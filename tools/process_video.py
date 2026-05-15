import argparse
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCER_DIR = os.path.join(BASE_DIR, "enhancer", "EnlightenGAN-inference")

if ENHANCER_DIR not in sys.path:
    sys.path.append(ENHANCER_DIR)

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError as e:
    raise SystemExit(
        f"Cannot import EnlightenGAN from {ENHANCER_DIR}. "
        "Make sure enhancer/EnlightenGAN-inference exists."
    ) from e


def _safe_makedirs(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _bracket_box(img, x1, y1, x2, y2, color, thickness=2, length=18):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def process_video(
    source: str,
    weights: str,
    out_path: str,
    process_w: int,
    process_h: int,
    out_w: int,
    out_h: int,
    frame_stride: int,
    conf: float,
    show: bool,
) -> None:
    enhancer = EnlightenOnnxModel()
    detector = YOLO(weights)

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    # Output video writer
    _safe_makedirs(out_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in / max(frame_stride, 1) if fps_in and fps_in > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(
            f"Could not open video writer for: {out_path}. "
            "If you're in Docker, ensure ffmpeg is installed in the image."
        )

    t0 = time.time()
    frames = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1

            if frame_stride > 1 and (frames % frame_stride) != 0:
                continue

            small = cv2.resize(frame, (process_w, process_h))
            enhanced = enhancer.predict(small)
            enhanced = np.ascontiguousarray(enhanced)

            results = detector(enhanced, imgsz=process_w, conf=conf, verbose=False)

            hud = enhanced.copy()
            h, _ = hud.shape[:2]
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height_ratio = (y2 - y1) / max(h, 1)
                if height_ratio > 0.35:
                    color = (0, 0, 255)
                elif height_ratio > 0.2:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                _bracket_box(hud, x1, y1, x2, y2, color)

            out_frame = cv2.resize(hud, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            # PIP raw (top-right)
            pip_h, pip_w = int(out_h * 0.27), int(out_w * 0.27)
            pip = cv2.resize(frame, (pip_w, pip_h))
            cv2.rectangle(pip, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)
            cv2.putText(
                pip, "RAW INPUT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            y_off, x_off = 20, out_w - pip_w - 20
            out_frame[y_off : y_off + pip_h, x_off : x_off + pip_w] = pip

            processed += 1
            elapsed = time.time() - t0
            fps = processed / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                out_frame,
                f"FPS: {fps:0.1f} (stride={frame_stride})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            writer.write(out_frame)

            if show:
                cv2.imshow("Night-to-Day (Docker/headless-safe)", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="enhancer/test1.mp4", help="Video path or 0 for webcam")
    parser.add_argument(
        "--weights",
        default="runs/detect/heavy_night_model2/weights/best.pt",
        help="Path to YOLOv8 weights",
    )
    parser.add_argument("--out", default="outputs/out.mp4", help="Output MP4 path")
    parser.add_argument("--process-w", type=int, default=256)
    parser.add_argument("--process-h", type=int, default=192)
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=720)
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Show window (not recommended in Docker)")
    args = parser.parse_args()

    weights = os.path.join(BASE_DIR, args.weights)
    if not os.path.exists(weights):
        raise SystemExit(f"Weights not found: {weights}")

    source = args.source
    if source != "0":
        source = os.path.join(BASE_DIR, source) if not os.path.isabs(source) else source

    out_path = args.out
    out_path = os.path.join(BASE_DIR, out_path) if not os.path.isabs(out_path) else out_path

    process_video(
        source=source,
        weights=weights,
        out_path=out_path,
        process_w=args.process_w,
        process_h=args.process_h,
        out_w=args.out_w,
        out_h=args.out_h,
        frame_stride=max(args.stride, 1),
        conf=args.conf,
        show=args.show,
    )


if __name__ == "__main__":
    main()

