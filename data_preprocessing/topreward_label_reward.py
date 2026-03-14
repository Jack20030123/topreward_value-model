"""
Label offline trajectories using TOPReward (vLLM server with Qwen3-VL-8B).

Replaces ReWiND's learned reward model with TOPReward's token-probability reward.
The output H5 file has the same schema as metaworld_labeled.h5 so the rest of
the offline-to-online pipeline works unchanged.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import io
import base64
import argparse
import h5py
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from data_preprocessing.generate_dino_embeddings import DINO_BATCH_SIZE
from utils.processing_utils import dino_load_image


DINO_BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO for image embeddings (still needed for policy observation features in the H5)
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
dinov2_vits14 = dinov2_model.to(device)

# MiniLM for text embeddings (still needed for policy language features in the H5)
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_dino_embeddings(imgs_list):
    episode_images_dino = [dino_load_image(img) for img in imgs_list]
    episode_images_dino = [
        torch.concatenate(episode_images_dino[i : i + DINO_BATCH_SIZE])
        for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
    ]
    embedding_list = []
    for batch in episode_images_dino:
        episode_image_embeddings = (
            dinov2_vits14(batch.to(device)).squeeze().detach().cpu().numpy()
        )
        if len(episode_image_embeddings.shape) == 1:
            episode_image_embeddings = np.expand_dims(episode_image_embeddings, 0)
        embedding_list.append(episode_image_embeddings)
    return np.concatenate(embedding_list)


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
        print(f"[TOPReward] VLM query failed: {e}")
        return -10.0


def compute_prefix_rewards(api_url, model_name, video_frames, instruction, num_samples=15):
    """Compute TOPReward for trajectory prefixes of varying lengths."""
    num_frames = len(video_frames)
    num_samples = min(num_samples, num_frames)

    if num_frames > 2:
        prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
        prefix_lengths = sorted(set(int(x) for x in prefix_lengths))
    else:
        prefix_lengths = [num_frames]

    prefix_rewards = []
    for length in prefix_lengths:
        prefix = video_frames[:length]
        # Subsample frames within each prefix if too many
        if len(prefix) > num_samples:
            indices = np.linspace(0, len(prefix) - 1, num_samples, dtype=int)
            prefix = [prefix[i] for i in indices]
        b64 = frames_to_base64(prefix)
        r = query_vlm_reward(api_url, model_name, b64, instruction)
        prefix_rewards.append(r)

    # Interpolate to get per-step rewards
    # prefix_lengths maps to prefix_rewards; interpolate for all steps 1..num_frames
    all_steps = np.arange(1, num_frames + 1)
    per_step_rewards = np.interp(all_steps, prefix_lengths, prefix_rewards)

    return per_step_rewards


def label_trajectories(args, traj_h5, embedding_h5):
    training_keys = list(embedding_h5.keys())

    # Compute total timesteps
    total_timesteps = 0
    for key in training_keys:
        for traj_id in traj_h5[key].keys():
            total_timesteps += len(traj_h5[key][traj_id]["reward"])
    total_timesteps = int(total_timesteps * 5)  # 5 annotations per trajectory

    labeled_dataset = h5py.File(args.output_path, "w")
    labeled_dataset.create_dataset("action", (total_timesteps, 4), dtype="float32")
    labeled_dataset.create_dataset("rewards", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset("done", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset("policy_lang_embedding", (total_timesteps, 384), dtype="float32")
    labeled_dataset.create_dataset("img_embedding", (total_timesteps, 768), dtype="float32")
    labeled_dataset.create_dataset("env_id", (total_timesteps,), dtype="S20")

    # Task instruction mapping (same as in metaworld pipeline)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "metaworld_policy_training"))
    from envs.metaworld import environment_to_instruction

    current_timestep = 0

    for key in tqdm(training_keys):
        instruction = environment_to_instruction.get(key, key)

        for traj_id in traj_h5[key].keys():
            traj_data = traj_h5[key][traj_id]
            num_steps = len(traj_data["done"])
            video_frames = np.array(traj_data["img"])
            video_frames_list = [img for img in video_frames]

            save_actions = np.array(traj_data["action"])
            save_dones = np.array(traj_data["done"])

            # Get DINO embeddings (still needed for policy features)
            video_frame_embeddings = get_dino_embeddings(video_frames_list)
            save_video_slices = video_frame_embeddings

            # Compute TOPReward for each prefix
            per_step_rewards = compute_prefix_rewards(
                args.api_url, args.model_name,
                video_frames_list, instruction,
                num_samples=args.num_prefix_samples,
            )

            # per_step_rewards has length num_frames; we need num_steps rewards
            # (num_steps = num_frames - 1 typically, matching save_actions)
            if args.use_progress_diff:
                save_reward_outputs = per_step_rewards[1:] - per_step_rewards[:-1]
            else:
                save_reward_outputs = per_step_rewards[1:]

            # Language embeddings from the embedding H5
            lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])

            for i in range(len(lang_embeddings)):
                lang_embedding = lang_embeddings[i]

                labeled_dataset["action"][current_timestep:current_timestep + num_steps] = save_actions
                labeled_dataset["done"][current_timestep:current_timestep + num_steps] = save_dones
                labeled_dataset["rewards"][current_timestep:current_timestep + num_steps] = save_reward_outputs
                labeled_dataset["policy_lang_embedding"][current_timestep:current_timestep + num_steps] = np.tile(lang_embedding, (num_steps, 1))
                labeled_dataset["img_embedding"][current_timestep:current_timestep + num_steps] = save_video_slices[:-1]
                labeled_dataset["env_id"][current_timestep:current_timestep + num_steps] = key

                current_timestep += num_steps

    print(f"Successfully processed and saved {current_timestep} timesteps.")
    labeled_dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Label rewards using TOPReward (vLLM).")
    parser.add_argument("--h5_video_path", default="datasets/metaworld_generation.h5")
    parser.add_argument("--h5_embedding_path", default="datasets/metaworld_embeddings_train.h5")
    parser.add_argument("--output_path", type=str, default="datasets/metaworld_topreward_labeled.h5")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                        help="vLLM server URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_prefix_samples", type=int, default=15,
                        help="Number of prefix lengths to sample for reward computation")
    parser.add_argument("--use_progress_diff", action="store_true",
                        help="Use progress diff (R(s') - R(s)) instead of R(s) as reward.")

    args = parser.parse_args()

    h5_video_file = h5py.File(args.h5_video_path, "r")
    h5_embedding_file = h5py.File(args.h5_embedding_path, "r")

    label_trajectories(args, h5_video_file, h5_embedding_file)


if __name__ == "__main__":
    main()
