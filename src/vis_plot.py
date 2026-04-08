import json
import os

import matplotlib.pyplot as plt


def find_latest_metrics_json(output_dir: str = "output") -> str:
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    json_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No json files found in: {output_dir}")

    json_files.sort(key=os.path.getmtime, reverse=True)
    return json_files[0]


def load_metrics_json(output_dir: str = "output") -> dict:
    json_path = find_latest_metrics_json(output_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print(f"[vis_plot] Loaded metrics json: {json_path}")
    return metrics


def main():
    metrics = load_metrics_json(output_dir="output")
    frames = metrics["frames"]

    frame_indices = [item["frame_idx"] for item in frames]
    soft_scores = [item["soft_score"] for item in frames]
    hard_scores = [item["hard_score"] for item in frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(frame_indices, soft_scores)
    ax1.set_ylabel("Soft Score")
    ax1.set_title("Soft Score Over Time")
    ax1.grid(True)

    ax2.plot(frame_indices, hard_scores)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Hard Score")
    ax2.set_title("Hard Score Over Time")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()