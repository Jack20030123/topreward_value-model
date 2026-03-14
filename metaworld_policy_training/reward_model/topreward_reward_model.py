"""
TOPReward reward model that calls a vLLM server running Qwen3-VL-8B.

Instead of using a locally-loaded learned reward model (like ReWiND),
this computes rewards by querying a VLM with video frames and a task
instruction, extracting token log-probabilities for "True" to measure
task completion progress.
"""

import os
import base64
import fcntl
import io
import requests
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Union
from PIL import Image

from reward_model.base_reward_model import BaseRewardModel
from reward_model.reward_utils import dino_load_image, mean_pooling
from transformers import AutoTokenizer, AutoModel


class TOPRewardModel(BaseRewardModel):
    def __init__(
        self,
        api_url: str,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        camera_names: List[str] = None,
        device: str = "cuda",
        batch_size: int = 64,
        reward_at_every_step: bool = False,
        success_bonus: float = 10.0,
        num_prefix_samples: int = 15,
        fps: float = 2.0,
    ):
        super().__init__(device, batch_size, success_bonus=success_bonus)
        self.reward_at_every_step = reward_at_every_step
        self.camera_names = camera_names or ["image"]
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.num_prefix_samples = num_prefix_samples
        self.fps = fps

        # Track raw rewards for min-max normalization across episode
        self._episode_raw_rewards = []

        # MiniLM for text encoding (same as ReWiND, used for policy features)
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(device)

        # DINO for image encoding (same as ReWiND, used for policy features)
        self.dino_vits14 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14"
        ).to(device)
        self.dino_batch_size = 64

        # Store the task instruction text (set via set_instruction)
        self._instruction = None

        # File lock to prevent concurrent VLM requests from multiple training jobs
        self._lock_path = "/project2/biyik_1165/haobaizh/rewind_topreward/logs/vllm_request.lock"

    def set_instruction(self, instruction: str):
        """Set the task instruction for reward computation."""
        self._instruction = instruction

    # ── Text encoding (for reward model internal use) ──

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        with torch.no_grad():
            encoded_input = self.minilm_tokenizer(
                text, padding=False, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.minilm_model(**encoded_input)
            text_embeddings = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .numpy()
            )
        return text_embeddings

    # ── Image encoding (DINO, same as ReWiND) ──

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[0] == 1, "TOPReward doesn't support batch > 1"
        images = images.squeeze(0)
        with torch.inference_mode():
            episode_images_dino = [
                dino_load_image(
                    img.to("cpu").numpy().transpose(1, 2, 0).astype(np.uint8)
                )
                for img in images
            ]
            episode_images_dino = [
                torch.concatenate(episode_images_dino[i : i + self.dino_batch_size])
                for i in range(0, len(episode_images_dino), self.dino_batch_size)
            ]
            embedding_list = []
            for batch in episode_images_dino:
                episode_image_embeddings = (
                    self.dino_vits14(batch.to(self.device)).squeeze().detach().cpu()
                )
                embedding_list.append(episode_image_embeddings)
            episode_image_embeddings = torch.concat(embedding_list)
        return episode_image_embeddings.unsqueeze(0)

    # ── Core TOPReward: call vLLM server ──

    def _frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert numpy frames (H, W, C) to base64-encoded PNG strings."""
        b64_list = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return b64_list

    def _query_vlm_reward(self, frames_b64: List[str], instruction: str) -> float:
        """
        Query the vLLM server to compute instruction reward.

        Constructs the TOPReward prompt:
          [video frames] "The above video shows a robot manipulation trajectory
          that completes the following task: {instruction}
          Decide whether the above statement is True or not. The answer is: True"

        Then extracts log P("True") as the reward.
        """
        prompt_text = (
            "The above video shows a robot manipulation trajectory "
            "that completes the following task: "
        )
        instruction_suffix = (
            f"{instruction} Decide whether the above statement is True or not. "
            "The answer is:"
        )

        # Build multimodal content
        content = []
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": f"{prompt_text}{instruction_suffix}"})

        messages = [{"role": "user", "content": content}]

        # Call vLLM with logprobs to extract P("True")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }

        try:
            # File lock: only one training job queries the server at a time
            lock_file = open(self._lock_path, "w")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                resp = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

            # Extract logprob of "True" from the first generated token
            logprobs_content = result["choices"][0]["logprobs"]["content"]
            if logprobs_content:
                first_token = logprobs_content[0]
                # Check if "True" is in top_logprobs
                for entry in first_token.get("top_logprobs", []):
                    if entry["token"].strip().lower() == "true":
                        return entry["logprob"]
                # If "True" not in top logprobs, check if it's the generated token
                if first_token["token"].strip().lower() == "true":
                    return first_token["logprob"]

            # Fallback: return a very low reward
            return -10.0

        except Exception as e:
            print(f"[TOPReward] VLM query failed: {e}")
            return -10.0

    # ── Reward calculation (called by LearnedRewardWrapper) ──

    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
        camera_name: str = None,
    ) -> np.ndarray:
        """
        Calculate rewards using TOPReward.

        During online RL, this is called with DINO-encoded video embeddings.
        However, TOPReward needs raw frames (not embeddings) to send to the VLM.
        We store raw frames in _raw_frames_buffer (set by the wrapper).
        If raw frames are available, use VLM; otherwise fall back to a default.
        """
        if hasattr(self, '_raw_frames_buffer') and self._raw_frames_buffer is not None and len(self._raw_frames_buffer) > 0:
            return self._compute_vlm_reward()
        else:
            # During offline phase, rewards come from pre-labeled H5 file
            # This path should not normally be reached during online RL
            print("[TOPReward] Warning: No raw frames available, returning 0 reward")
            return np.array([0.0])

    def _compute_vlm_reward(self) -> np.ndarray:
        """Compute reward from raw frames using VLM."""
        frames = self._raw_frames_buffer
        instruction = self._instruction

        if instruction is None:
            print("[TOPReward] Warning: No instruction set, returning 0 reward")
            return np.array([0.0])

        # Subsample frames if too many
        if len(frames) > self.num_prefix_samples:
            indices = np.linspace(0, len(frames) - 1, self.num_prefix_samples, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        frames_b64 = self._frames_to_base64(sampled_frames)
        raw_reward = self._query_vlm_reward(frames_b64, instruction)

        return np.array([raw_reward])

    def _calculate_reward_batch(
        self,
        encoded_texts: torch.Tensor,
        encoded_videos: torch.Tensor,
    ) -> np.ndarray:
        """Fallback batch reward calculation. Not used in TOPReward online path."""
        return np.array([0.0])

    def store_raw_frame(self, frame: np.ndarray):
        """Store a raw frame for VLM reward computation."""
        if not hasattr(self, '_raw_frames_buffer'):
            self._raw_frames_buffer = []
        self._raw_frames_buffer.append(frame.copy())

    def clear_raw_frames(self):
        """Clear the raw frames buffer (called on episode reset)."""
        self._raw_frames_buffer = []

    @property
    def img_output_dim(self) -> int:
        return 768

    @property
    def text_output_dim(self) -> int:
        return 384

    @property
    def name(self) -> str:
        return "topreward"
