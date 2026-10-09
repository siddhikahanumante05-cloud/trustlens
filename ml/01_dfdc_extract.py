"""DFDC Face Extraction pipeline for Kaggle and Colab.
Extracts 16-frame 160x160 face crops per 3-second window, strictly partitioned by source identity.
"""
import os
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def extract_faces_from_dfdc(
    dfdc_root: str,
    output_dir: str,
    parts: List[int] = [0, 1, 2],
    frames_per_window: int = 16,
    crop_size: int = 160,
    fps_target: int = 15,
):
    """
    Extract aligned face crops from DFDC video archives.
    Rule 1: Split data by source video ('original' field) so train and test NEVER share a person.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    metadata_all = {}
    video_paths = []

    for part in parts:
        part_dir = os.path.join(dfdc_root, f"dfdc_train_part_{part}")
        meta_file = os.path.join(part_dir, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
                for v_name, v_info in meta.items():
                    v_path = os.path.join(part_dir, v_name)
                    if os.path.exists(v_path):
                        metadata_all[v_name] = v_info
                        video_paths.append(v_path)

    print(f"Discovered {len(video_paths)} videos across parts {parts}.")

    # Identify unique original video identities
    # 'original' field points to the real source video for deepfakes; for reals, it's itself
    originals = set()
    for v_name, v_info in metadata_all.items():
        orig = v_info.get("original", v_name) or v_name
        originals.add(orig)

    sorted_originals = sorted(list(originals))
    # 80/20 train/test split on unique source identities
    split_idx = int(0.8 * len(sorted_originals))
    train_origs = set(sorted_originals[:split_idx])
    test_origs = set(sorted_originals[split_idx:])

    print(f"Partitioned into {len(train_origs)} train identities and {len(test_origs)} test identities.")

    # Face cascade detector for extraction
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    processed_count = 0
    for v_path in video_paths:
        v_name = os.path.basename(v_path)
        info = metadata_all.get(v_name, {})
        label = 1 if info.get("label") == "FAKE" else 0
        orig = info.get("original", v_name) or v_name

        dest_folder = train_dir if orig in train_origs else test_dir
        out_name = f"{os.path.splitext(v_name)[0]}_label_{label}.npy"
        out_path = os.path.join(dest_folder, out_name)

        if os.path.exists(out_path):
            continue

        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 45:
            cap.release()
            continue

        # Extract 16 frames evenly from a 3-second (approx 45 frames) segment
        step = max(1, int(fps / fps_target))
        frame_indices = np.linspace(0, min(total_frames - 1, int(3.0 * fps)), frames_per_window, dtype=int)

        crops = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))

            if len(faces) > 0:
                # Largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_crop = rgb[y:y + h, x:x + w]
                crop_resized = cv2.resize(face_crop, (crop_size, crop_size))
            else:
                # Central fallback crop
                h, w, _ = rgb.shape
                cx, cy = w // 2, h // 2
                half = min(cx, cy, 100)
                crop_resized = cv2.resize(rgb[cy - half:cy + half, cx - half:cx + half], (crop_size, crop_size))

            crops.append(crop_resized)

        cap.release()

        if len(crops) == frames_per_window:
            arr = np.array(crops, dtype=np.uint8)  # (16, 160, 160, 3)
            np.save(out_path, arr)
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Extracted {processed_count} clips...")

    print(f"Extraction complete: {processed_count} clips cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfdc_root", default="./data/dfdc", help="Root DFDC directory")
    parser.add_argument("--output_dir", default="./data/cached_faces", help="Output directory for crops")
    args = parser.parse_args()
    extract_faces_from_dfdc(args.dfdc_root, args.output_dir)
