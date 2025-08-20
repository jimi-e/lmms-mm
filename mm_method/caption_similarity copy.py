import os
import hashlib
import json
from typing import List

import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None


def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _uniform_sample(total: int, num: int) -> List[int]:
    if total <= 0 or num <= 0:
        return []
    num = min(total, num)
    return np.linspace(0, total - 1, num, dtype=int).tolist()


class _QwenVLCaptioner:
    """A lightweight wrapper around Qwen2-VL for single-image captioning.

    Requires transformers>=4.41 and the model weights to be available locally or via HF hub.
    Model: Qwen/Qwen2-VL-7B-Instruct
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda") -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM  # lazy import

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device

    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 64) -> List[str]:
        captions: List[str] = []
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe the image briefly in one sentence."},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            captions.append(caption.strip())
        return captions


class _BLIPCaptioner:
    """Fallback captioner using BLIP if Qwen2-VL is not available."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = "cuda") -> None:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # lazy import

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 32) -> List[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return [c.strip() for c in self.processor.batch_decode(output_ids, skip_special_tokens=True)]


class _TextEmbedder:
    """Sentence-transformers based embedder with L2-normalized outputs."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


def _get_captioner(prefer_qwen: bool, device: str):
    if prefer_qwen:
        try:
            return _QwenVLCaptioner(device=device)
        except Exception:
            # Fallback to BLIP if Qwen2-VL is unavailable
            return _BLIPCaptioner(device=device)
    else:
        return _BLIPCaptioner(device=device)


def select_frames_by_caption_similarity(
    video_path: str,
    question: str,
    *,
    cache_root: str = "mm_method_cache",
    device: str = "cuda",
    prefer_qwen_captioner: bool = True,
    max_candidates: int = 256,
    top_n: int = 32,
    save_selected_dir: bool = True,
) -> List[str]:
    """Return a list of selected frame file paths by caption-text similarity.

    Steps:
      1) Uniformly sample up to `max_candidates` frames from `video_path`.
      2) Generate a short caption per frame (Qwen2-VL preferred; BLIP fallback).
      3) Embed captions and the question, compute cosine similarity, and pick Top-N.
      4) Save selected frames to cache directory and return their file paths.

    Note: This function intentionally does not change the caller's workflow; it only
          returns a list of image paths so the main pipeline can stack them as frames.
    """

    if VideoReader is None:
        raise RuntimeError("decord is required for frame sampling but is not available.")

    # 0) Prepare cache
    vid_stem = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(cache_root, vid_stem)
    _ensure_dir(cache_dir)

    # 1) Sample candidate frames
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idxs = _uniform_sample(total, max_candidates)
    frames = vr.get_batch(idxs).asnumpy()  # (K, H, W, C)

    # Convert to PIL list
    pil_frames = [Image.fromarray(arr).convert("RGB") for arr in frames]

    # 2) Captioning (with caching)
    cap_key = _hash_key(f"cap:{len(idxs)}:{question}")
    cap_path = os.path.join(cache_dir, f"captions_{cap_key}.json")
    if os.path.exists(cap_path):
        captions = json.load(open(cap_path, "r"))
    else:
        captioner = _get_captioner(prefer_qwen=prefer_qwen_captioner, device=device)
        captions = captioner.caption_batch(pil_frames)
        json.dump(captions, open(cap_path, "w"), ensure_ascii=False)

    # 3) Embedding + similarity (with caching)
    emb_key = _hash_key(f"emb:{len(idxs)}:{question}")
    emb_caps_path = os.path.join(cache_dir, f"cap_emb_{emb_key}.npy")
    emb_q_path = os.path.join(cache_dir, f"q_emb_{emb_key}.npy")

    if os.path.exists(emb_caps_path) and os.path.exists(emb_q_path):
        cap_embs = np.load(emb_caps_path)
        q_emb = np.load(emb_q_path)
    else:
        embedder = _TextEmbedder(device=device)
        cap_embs = embedder.encode(captions)
        q_emb = embedder.encode([question])[0]
        np.save(emb_caps_path, cap_embs)
        np.save(emb_q_path, q_emb)

    sims = cap_embs @ q_emb  # cosine similarity since normalized
    order = np.argsort(-sims)
    sel = order[: min(top_n, len(order))].tolist()

    # 4) Save selected frames to disk and return their paths
    selected_dir = os.path.join(cache_dir, f"selected_top{len(sel)}")
    if save_selected_dir:
        _ensure_dir(selected_dir)
        paths: List[str] = []
        for rank, i in enumerate(sel):
            out_path = os.path.join(selected_dir, f"{rank:04d}_idx{i}.jpg")
            pil_frames[i].save(out_path, format="JPEG", quality=90)
            paths.append(out_path)
        return paths
    else:
        # Save to temp files in cache_dir root
        paths: List[str] = []
        for rank, i in enumerate(sel):
            out_path = os.path.join(cache_dir, f"tmp_{rank:04d}_idx{i}.jpg")
            pil_frames[i].save(out_path, format="JPEG", quality=90)
            paths.append(out_path)
        return paths
