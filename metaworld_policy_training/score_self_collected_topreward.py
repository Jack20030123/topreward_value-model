"""
Score self-collected videos using TOPReward (vLLM server with Qwen3-VL-8B).

Reads existing mp4/gif videos from self_collected_videos/, scores each prefix
with the VLM, and generates 1x3 visualisation videos:
  left: original video | middle: raw progress curve | right: diff progress curve

Usage (from metaworld_policy_training/):
    python score_self_collected_topreward.py \
        --video_root ../self_collected_videos \
        --api_url http://<vllm-host>:8000 \
        --output_dir score_output/self_collected_topreward
"""

import os
import sys
import io
import base64
import argparse
import numpy as np
import requests
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

environment_to_instruction = {
    "assembly-v2": "assembly",
    "basketball-v2": "play basketball",
    "bin-picking-v2": "pick bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "disassemble-v2": "disassemble",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "drawer-close-v2": "Close the drawer",
    "drawer-open-v2": "open drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "hammer-v2": "hammer nail",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "lever-pull-v2": "pull lever",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-wall-v2": "Pick up the block and placing it to the goal position",
    "pick-out-of-hole-v2": "pick bin",
    "reach-v2": "Reach the goal",
    "push-back-v2": "Push the block back to the goal",
    "push-v2": "Push the block to the goal",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-side-v2": "Slide the plate into the gate from the side",
    "plate-slide-back-v2": "Slide the plate out of the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "peg-unplug-side-v2": "unplug peg",
    "soccer-v2": "Slide the ball into the gate",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "push-wall-v2": "push bin",
    "reach-wall-v2": "Reach the goal",
    "shelf-place-v2": "place bin to shelf",
    "sweep-into-v2": "Sweep the block into the hole",
    "sweep-v2": "sweep block",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
}


def dir_name_to_env_id(name):
    if name.endswith("-v2"):
        return name
    return name.replace("_", "-") + "-v2"


def read_video_frames(video_path):
    frames = iio.imread(video_path)
    result = []
    for frame in frames:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        result.append(frame.astype(np.uint8))
    return result


def frames_to_base64(frames):
    b64_list = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_list


def query_vlm_reward(api_url, model_name, frames_b64, instruction):
    """Query vLLM server for log P("True") as reward."""
    prompt_text = (
        "The above video shows a robot manipulation trajectory "
        "that completes the following task: "
    )
    instruction_suffix = (
        f"{instruction} Decide whether the above statement is True or not. "
        "The answer is:"
    )

    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": f"{prompt_text}{instruction_suffix}"})

    messages = [{"role": "user", "content": content}]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 5,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }

    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()

        logprobs_content = result["choices"][0]["logprobs"]["content"]
        if logprobs_content:
            first_token = logprobs_content[0]
            for entry in first_token.get("top_logprobs", []):
                if entry["token"].strip().lower() == "true":
                    return entry["logprob"]
            if first_token["token"].strip().lower() == "true":
                return first_token["logprob"]

        return -10.0

    except Exception as e:
        print(f"  [TOPReward] VLM query failed: {e}")
        return -10.0


def score_trajectory(api_url, model_name, raw_images, instruction, num_prefix_samples=15):
    """Score a trajectory using TOPReward prefix rewards."""
    num_frames = len(raw_images)
    num_samples = min(num_prefix_samples, num_frames)

    if num_frames > 2:
        prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
        prefix_lengths = sorted(set(int(x) for x in prefix_lengths))
    else:
        prefix_lengths = [num_frames]

    prefix_rewards = []
    for length in prefix_lengths:
        prefix = raw_images[:length]
        if len(prefix) > num_prefix_samples:
            indices = np.linspace(0, len(prefix) - 1, num_prefix_samples, dtype=int)
            prefix = [prefix[i] for i in indices]
        b64 = frames_to_base64(prefix)
        r = query_vlm_reward(api_url, model_name, b64, instruction)
        prefix_rewards.append(r)

    # Interpolate to get per-step rewards
    all_steps = np.arange(1, num_frames + 1)
    per_step_rewards = np.interp(all_steps, prefix_lengths, prefix_rewards)

    return per_step_rewards


def generate_video(images, progress_raw, progress_diff, video_path, title, fps=10):
    """Generate 1x3 MP4: left=video, middle=raw progress, right=diff progress."""
    diff_padded = np.concatenate([[0.0], progress_diff])
    num_frames = len(images)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: video frames
    ax_img = axes[0]
    im = ax_img.imshow(images[0])
    ax_img.set_title("Video", fontsize=12)
    ax_img.axis("off")
    step_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Middle: raw progress (log P("True"))
    ax_raw = axes[1]
    (line_raw,) = ax_raw.plot([], [], "b-", linewidth=2)
    ax_raw.set_xlim(0, num_frames)
    margin = max(0.05, (np.max(progress_raw) - np.min(progress_raw)) * 0.1)
    ax_raw.set_ylim(np.min(progress_raw) - margin, np.max(progress_raw) + margin)
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("log P(True)")
    ax_raw.set_title("Raw Progress (TOPReward)", fontsize=12)
    ax_raw.grid(True, alpha=0.3)
    (dot_raw,) = ax_raw.plot([], [], "bo", markersize=5)

    # Right: diff progress
    ax_diff = axes[2]
    (line_diff,) = ax_diff.plot([], [], "m-", linewidth=2)
    ax_diff.set_xlim(0, num_frames)
    d_margin = max(0.01, (np.max(diff_padded) - np.min(diff_padded)) * 0.15)
    ax_diff.set_ylim(np.min(diff_padded) - d_margin, np.max(diff_padded) + d_margin)
    ax_diff.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_diff.set_xlabel("Step")
    ax_diff.set_ylabel("Diff")
    ax_diff.set_title("Diff Progress (TOPReward)", fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    (dot_diff,) = ax_diff.plot([], [], "mo", markersize=5)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    def init():
        line_raw.set_data([], [])
        line_diff.set_data([], [])
        dot_raw.set_data([], [])
        dot_diff.set_data([], [])
        step_text.set_text("")
        return line_raw, line_diff, dot_raw, dot_diff, step_text, im

    def animate(frame):
        im.set_array(images[frame])
        step_text.set_text(f"Step: {frame}/{num_frames - 1}")
        x = np.arange(frame + 1)
        line_raw.set_data(x, progress_raw[: frame + 1])
        line_diff.set_data(x, diff_padded[: frame + 1])
        dot_raw.set_data([frame], [progress_raw[frame]])
        dot_diff.set_data([frame], [diff_padded[frame]])
        return line_raw, line_diff, dot_raw, dot_diff, step_text, im

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(video_path, writer=writer)
    plt.close(fig)


def collect_videos(video_root):
    SKIP_DIRS = {"eval_tasks", "train_tasks"}
    for entry in sorted(os.listdir(video_root)):
        entry_path = os.path.join(video_root, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry in SKIP_DIRS:
            for env_dir in sorted(os.listdir(entry_path)):
                env_dir_path = os.path.join(entry_path, env_dir)
                if not os.path.isdir(env_dir_path):
                    continue
                env_id = dir_name_to_env_id(env_dir)
                for cat in sorted(os.listdir(env_dir_path)):
                    cat_path = os.path.join(env_dir_path, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for vf in sorted(os.listdir(cat_path)):
                        if vf.lower().endswith((".mp4", ".gif")):
                            yield os.path.join(cat_path, vf), env_id, f"{entry}/{env_dir}/{cat}"
        else:
            env_id = dir_name_to_env_id(entry)
            for cat in sorted(os.listdir(entry_path)):
                cat_path = os.path.join(entry_path, cat)
                if not os.path.isdir(cat_path):
                    continue
                for vf in sorted(os.listdir(cat_path)):
                    if vf.lower().endswith((".mp4", ".gif")):
                        yield os.path.join(cat_path, vf), env_id, f"{entry}/{cat}"


def main():
    parser = argparse.ArgumentParser(description="Score self-collected videos with TOPReward")
    parser.add_argument("--video_root", type=str, default="../self_collected_videos",
                        help="Root directory containing self-collected videos")
    parser.add_argument("--api_url", type=str, required=True,
                        help="vLLM server URL (e.g. http://b18-17.hpc.usc.edu:8000)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_prefix_samples", type=int, default=15,
                        help="Number of prefix lengths to sample for scoring")
    parser.add_argument("--output_dir", type=str, default="score_output/self_collected_topreward")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    print("=== TOPReward Video Scoring ===")
    print(f"vLLM server: {args.api_url}")
    print(f"Model: {args.model_name}")

    # Health check
    try:
        resp = requests.get(f"{args.api_url}/health", timeout=10)
        resp.raise_for_status()
        print("vLLM server is healthy!\n")
    except Exception as e:
        print(f"ERROR: vLLM server health check failed: {e}")
        sys.exit(1)

    videos = list(collect_videos(args.video_root))
    print(f"Found {len(videos)} videos to score.\n")

    for idx, (video_path, env_id, rel_category) in enumerate(videos):
        if env_id not in environment_to_instruction:
            print(f"[{idx+1}/{len(videos)}] SKIP {video_path} — unknown env_id '{env_id}'")
            continue

        text_instruction = environment_to_instruction[env_id]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx+1}/{len(videos)}] {rel_category}/{video_name}  env={env_id}  inst=\"{text_instruction}\"")

        try:
            raw_images = read_video_frames(video_path)
        except Exception as e:
            print(f"  ERROR reading video: {e}")
            continue

        if len(raw_images) < 2:
            print(f"  SKIP — only {len(raw_images)} frame(s)")
            continue

        # Score with TOPReward
        progress_raw = score_trajectory(
            args.api_url, args.model_name, raw_images,
            text_instruction, num_prefix_samples=args.num_prefix_samples,
        )
        progress_diff = np.diff(progress_raw)

        # Output
        out_dir = os.path.join(args.output_dir, rel_category)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{video_name}_scored.mp4")

        title = f"{rel_category}/{video_name} — {env_id} (TOPReward)"
        generate_video(raw_images, progress_raw, progress_diff, out_path, title, fps=args.fps)
        print(f"  -> {out_path}  ({len(raw_images)} frames, progress [{progress_raw.min():.3f}, {progress_raw.max():.3f}])")

    print("\nDone!")


if __name__ == "__main__":
    main()
