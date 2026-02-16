import os
import tempfile
import math
from collections import deque
from itertools import combinations

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CAR_CLASS_ID = 2
MATCH_DIST_PX = 60
CLOSE_DIST_PX = 80
MAX_TRACK_AGE = 20
NEAR_MISS_TIME_S = 4.0

# =========================
# SPEED CALCULATION
# =========================
def compute_speed_mph(history, fps, meters_per_pixel):
    if len(history) < 2:
        return 0.0

    f0, x0, y0 = history[-2]
    f1, x1, y1 = history[-1]

    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0

    dist_m = math.hypot(x1 - x0, y1 - y0) * meters_per_pixel
    speed_mps = dist_m / dt
    return speed_mps * 2.23694  # mph

# =========================
# SIMPLE TRACKER
# =========================
class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        updated = {}

        for cx, cy, bbox in detections:
            matched = None

            for tid, tr in self.tracks.items():
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist < MATCH_DIST_PX:
                    matched = tid
                    break

            if matched is None:
                matched = self.next_id
                self.next_id += 1
                updated[matched] = {
                    "cx": cx,
                    "cy": cy,
                    "bbox": bbox,
                    "last": frame_idx,
                    "hist": deque(maxlen=20),
                }
            else:
                tr = self.tracks[matched]
                tr["cx"] = cx
                tr["cy"] = cy
                tr["bbox"] = bbox
                tr["last"] = frame_idx
                updated[matched] = tr

            updated[matched]["hist"].append((frame_idx, cx, cy))

        self.tracks = {
            tid: tr for tid, tr in updated.items()
            if frame_idx - tr["last"] <= MAX_TRACK_AGE
        }

        return self.tracks

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Traffic Near Miss Detection")

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

st.subheader("Calibration")
meters_per_pixel = st.number_input(
    "Meters per pixel (estimate using lane width)",
    value=0.05,
    step=0.01
)

run = st.button("Run Analysis", type="primary")

# =========================
# MAIN LOGIC
# =========================
if run and uploaded:

    with st.spinner("Processing video..."):

        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, uploaded.name)

        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(tmp_dir, "output.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        model = YOLO("yolov8n.pt")
        tracker = SimpleTracker()

        progress = st.progress(0)
        preview = st.empty()
        popup = st.empty()

        events = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.3, verbose=False)[0]

            detections = []
            if results.boxes is not None:
                for box, cls in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.cls.cpu().numpy().astype(int)
                ):
                    if cls == CAR_CLASS_ID:
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        detections.append((cx, cy, box))

            tracks = tracker.update(detections, frame_idx)

            danger_ids = set()
            popup_lines = []

            track_list = list(tracks.items())

            for (id1, t1), (id2, t2) in combinations(track_list, 2):

                dist = math.hypot(t1["cx"] - t2["cx"], t1["cy"] - t2["cy"])

                if dist < CLOSE_DIST_PX:

                    danger_ids.update([id1, id2])

                    v1 = compute_speed_mph(t1["hist"], fps, meters_per_pixel)
                    v2 = compute_speed_mph(t2["hist"], fps, meters_per_pixel)

                    event_time = round(frame_idx / fps, 2)

                    popup_lines.append(
                        f"Cars {id1} & {id2} | Avg Speed {round((v1+v2)/2,1)} mph | Time {event_time}s"
                    )

                    events.append({
                        "time_s": event_time,
                        "car_1": id1,
                        "car_2": id2,
                        "avg_speed_mph": round((v1+v2)/2,2)
                    })

            popup.info("\n".join(popup_lines) if popup_lines else "No active conflict")

            for tid, tr in tracks.items():
                x1, y1, x2, y2 = map(int, tr["bbox"])
                color = (0, 0, 255) if tid in danger_ids else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            if frame_idx % 5 == 0:
                preview.image(frame, channels="BGR", use_container_width=True)

            writer.write(frame)

            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1

        cap.release()
        writer.release()

    df = pd.DataFrame(events)

    st.subheader("Analysis Summary")

    if df.empty:
        st.write("No near-miss events detected.")
    else:
        st.write(f"Total Near Miss Events: {len(df)}")
        st.write(f"Average Speed During Events: {round(df['avg_speed_mph'].mean(),2)} mph")

    st.subheader("Event Table")
    st.dataframe(df, use_container_width=True)

    csv_path = os.path.join(tmp_dir, "events.csv")
    df.to_csv(csv_path, index=False)

    st.video(out_path)
    st.download_button("Download CSV", open(csv_path, "rb"), "events.csv")
