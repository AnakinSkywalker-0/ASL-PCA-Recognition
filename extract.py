"""
STEP 1 — Extract hand landmarks from dataset images → gestures.csv
===================================================================
Run this ONCE on your Figshare dataset folder.

Expected folder layout:
    dataset/
        hello/   image1.jpg  image2.jpg ...
        bye/     image1.jpg  ...
        yes/     ...
        no/      ...
        help/    ...
        (any other gesture folders)

Output: gestures.csv  (label + 42 normalised floats per row)

Usage:
    python step1_extract.py --dataset ./dataset --out gestures.csv
"""

import os
import csv
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── download model if missing ──────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"

def download_model():
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print("Downloading MediaPipe hand model (~25 MB)...")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# ── normalise vector so position/scale don't matter ───────────────────────────
def normalise(landmarks):
    """
    Subtract wrist (landmark 0) so the hand is position-independent.
    Divide by the distance from wrist to middle-finger-MCP (landmark 9)
    so scale is consistent regardless of how close the hand is to camera.
    Returns a flat 42-float numpy array.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks])   # (21, 2)
    origin = pts[0].copy()
    pts -= origin                                          # centre on wrist
    scale = np.linalg.norm(pts[9]) + 1e-6                 # distance to mid-MCP
    pts /= scale                                           # normalise scale
    return pts.flatten()                                   # (42,)

# ── main extraction loop ───────────────────────────────────────────────────────
def extract(dataset_dir, out_csv):
    if not os.path.exists(MODEL_PATH):
        download_model()

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )

    detected = skipped = 0
    rows = []

    with HandLandmarker.create_from_options(options) as detector:
        gesture_folders = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        print(f"Found gesture folders: {gesture_folders}\n")

        for label in gesture_folders:
            folder = os.path.join(dataset_dir, label)
            images = [
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            label_count = 0

            for img_file in images:
                path = os.path.join(folder, img_file)
                try:
                    # 1. Load image via OpenCV instead of MediaPipe's direct loader
                    bgr_img = cv2.imread(path)
                    if bgr_img is None:
                        print(f"  Could not read: {img_file}")
                        skipped += 1
                        continue
                    
                    # 2. Convert BGR to RGB (MediaPipe requirement)
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                    
                    result = detector.detect(mp_img)

                    if result.hand_landmarks:
                        vec = normalise(result.hand_landmarks[0])
                        rows.append([label] + vec.tolist())
                        label_count += 1
                        detected += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"  Error on {img_file}: {e}")
                    skipped += 1
    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y')]
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nDone. {detected} vectors saved → {out_csv}")
    print(f"Skipped (no hand detected): {skipped}")
    print(f"Tip: if many images are skipped, your dataset images may already have")
    print(f"     landmarks drawn on them. See README for how to handle that case.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset",
                    help="Path to dataset root folder (subfolders = gesture labels)")
    ap.add_argument("--out", default="gestures.csv",
                    help="Output CSV filename")
    args = ap.parse_args()
    extract(args.dataset, args.out)
