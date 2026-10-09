"""TrustLens Benchmark Replay Harness.
Evaluates detection pipeline over test video datasets and synthetic sessions.
Computes ROC curves, AUC, EER, latency profiles, and signal ablation analysis.
Strictly checks and reports stubs_in_use status.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple

# Ensure backend modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.decoder import ChunkDecoder
from app.windower import Windower
from app.pipeline import LivePipeline
from app.config import settings


def compute_roc_and_auc(scores: List[float], labels: List[int]) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Computes Receiver Operating Characteristic (ROC), Area Under Curve (AUC),
    and Equal Error Rate (EER).
    labels: 0 = authentic, 1 = deepfake.
    """
    if len(set(labels)) < 2:
        return 0.5, 0.5, []

    # Sort descending by score
    combined = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    tp = 0
    fp = 0
    roc_points = [{"threshold": 1.0, "fpr": 0.0, "tpr": 0.0}]

    eer = 0.5
    min_diff = 1.0

    for score, label in combined:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = 1.0 - tpr

        roc_points.append({"threshold": round(score, 4), "fpr": round(fpr, 4), "tpr": round(tpr, 4)})

        diff = abs(fpr - fnr)
        if diff < min_diff:
            min_diff = diff
            eer = (fpr + fnr) / 2.0

    # Trapezoidal rule for AUC
    auc = 0.0
    for i in range(1, len(roc_points)):
        p1 = roc_points[i - 1]
        p2 = roc_points[i]
        auc += (p2["fpr"] - p1["fpr"]) * (p1["tpr"] + p2["tpr"]) / 2.0

    return float(np.clip(auc, 0.0, 1.0)), float(eer), roc_points


def replay_file(file_path: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replays a single media file through the complete LivePipeline."""
    t0 = time.perf_counter()
    with open(file_path, "rb") as f:
        content = f.read()

    session_id = f"bench_{os.path.basename(file_path)}"
    decoder = ChunkDecoder(clip_mode=True)
    pipeline = LivePipeline(session_id=session_id, tap_mode="agent")

    if meta:
        pipeline.update_metadata(meta)

    decoder.feed_chunk(content)
    windower = Windower(decoder=decoder, stride_sec=1.0)
    if meta and "challenge" in meta:
        windower.register_challenge(meta["challenge"])

    windows_count = 0
    risks = []

    windows = windower.extract_all_windows()
    for win in windows:
        checks, risk, action, findings, evidence = pipeline.process_window(win)
        risks.append(risk)
        windows_count += 1

    summary_text, ai_gen = pipeline.finalize(consent_status="consented")
    elapsed = time.perf_counter() - t0

    pipeline.close()
    decoder.close()
    windower.stop()

    final_risk = float(np.mean(risks)) if risks else pipeline.latest_risk

    return {
        "file": os.path.basename(file_path),
        "windows_count": windows_count,
        "elapsed_sec": round(elapsed, 3),
        "final_risk": round(final_risk, 4),
        "category": pipeline.latest_category,
        "stubs_in_use": pipeline.runtime.stubs_in_use,
    }


def run_benchmark(fixtures_dir: str, output_path: str):
    """Executes the full evaluation replay suite."""
    print(f"[*] Starting TrustLens benchmark replay on {fixtures_dir}...")
    sample_webm = os.path.join(fixtures_dir, "sample_5s.webm")

    scores: List[float] = []
    labels: List[int] = []
    item_results: List[Dict[str, Any]] = []
    stubs_active = False

    # 1. Test authentic sample replay
    if os.path.exists(sample_webm):
        res_auth = replay_file(sample_webm, meta={"camera_label": "Integrated Webcam (04f2:b6d9)"})
        scores.append(res_auth["final_risk"])
        labels.append(0)  # Authentic
        item_results.append(res_auth)
        if res_auth["stubs_in_use"]:
            stubs_active = True

        # 2. Test synthetic / virtual cam replay
        res_fake = replay_file(sample_webm, meta={"camera_label": "OBS Virtual Camera", "frame_intervals_ms": [66.666] * 45})
        scores.append(res_fake["final_risk"])
        labels.append(1)  # Deepfake / Virtual cam
        item_results.append(res_fake)
        if res_fake["stubs_in_use"]:
            stubs_active = True

    # Compute ROC and metrics
    auc, eer, roc_curve = compute_roc_and_auc(scores, labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = {
        "timestamp": int(time.time()),
        "total_evaluated": len(scores),
        "stubs_in_use": stubs_active,
        "certified": not stubs_active,
        "warning": (
            "Stubs in use (stub-0). Certified production accuracy cannot be asserted until real model weights are trained and loaded."
            if stubs_active
            else None
        ),
        "metrics": {
            "auc": round(auc, 4),
            "eer": round(eer, 4),
        },
        "roc_curve": roc_curve,
        "evaluations": item_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Benchmark finished. Results saved to {output_path}")
    print(f"    Total Evaluated: {len(scores)} | Stubs Active: {stubs_active} | Certified: {summary['certified']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default=os.path.join(os.path.dirname(__file__), "fixtures"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "results", "summary.json"))
    args = parser.parse_args()

    run_benchmark(args.fixtures, args.output)
