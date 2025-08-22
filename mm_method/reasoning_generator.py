#!/usr/bin/env python3
"""
Stage-1 Reasoning Generator (non-overlapping windows)
=====================================================

Reads caption JSONs and produces per-frame reasoning using an LLM
(Transformers model: Mistral-7B-Instruct). Multi-process parallel processing similar
to the caption generator. Each worker processes a subset of files.

Output JSON mirrors the caption JSON structure, adding a
frames[<id>].reasoning field and a top-level reasoning_stage1
summary with known_info_all and window parameters.

Requirements:
- Hugging Face Transformers model e.g. mistralai/Mistral-7B-Instruct-v0.3
- Input directory: caption JSONs as produced earlier.

Usage example:
    python3 mm_method/reasoning_stage1_ollama.py \
        --input_dir /home/syh/work-d/test/video/lmms-eval/qwen_videomme_captions \
        --output_dir /home/syh/work-d/test/video/lmms-eval/videomme_reasoning_mistral \
        --num_gpus 4 \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --split test
"""

import os
import json
import argparse
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Callable
import time
import re
from dataclasses import dataclass
from tqdm import tqdm
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a video‑QA evidence writer. You ONLY see:\n"
    "- the question (Q),\n"
    "- an optional short “known info” summary from previous windows,\n"
    "- the captions of frames in the current non‑overlapping window,\n"
    "- and the video/window metadata provided below.\n\n"
    "For EVERY frame in this window, write 1–2 sentences of logical reasoning about HOW this frame could help answer the question (not mere description).\n"
    "If the question is temporal, explicitly reason about time (e.g., earlier/later/before/after/stage) and refer to timestamps when helpful.\n"
    "Within this window, group near‑duplicate frames by caption semantics:\n"
    "- If a cluster has ≥3 almost‑identical frames (no substantial new facts), keep 1–2 representatives WITHOUT any tag, and append or prepend “[LOW]” to the reasoning of ALL OTHER frames in that cluster.\n"
    "Do NOT invent details beyond captions.\n"
    "Finally, provide a concise time‑anchored “known info” update (≤ 3 bullet items) that summarizes NEW, non‑redundant facts discovered IN THIS WINDOW to help subsequent windows. Each bullet MUST include a time anchor like “[t=xx.xx s]” or a coarse phase “[phase=k]”.\n"
    "Use the SAME language as Q for all reasoning and bullets.\n\n"
    "STRICT FORMAT RULES:\n"
    "- Fill ONLY the 'reasoning:' lines for each id in the provided skeleton; write ≤2 sentences per frame.\n"
    "- If a frame is low-importance/near-duplicate, PREFIX the reasoning with '[LOW] ' and place the time anchor like '[t=xx.xx s]' at the END of the line.\n"
    "- DO NOT merge frames or use ranges/commas (no “id: 20-21”, “id: 28, 29”). Each id must have EXACTLY ONE 'reasoning:' line.\n"
    "- DO NOT add/remove ids or lines; DO NOT output anything outside the sentinel tags; DO NOT repeat or paraphrase the instructions."
)

USER_TEMPLATE = (
    "Question Q: {Q}\n\n"
    "Video metadata:\n"
    "- VIDEO_DURATION_SEC: {VIDEO_DURATION_SEC}\n"
    "- WINDOW_INDEX: {WINDOW_INDEX} / {TOTAL_WINDOWS}\n"
    "- WINDOW_SPAN_SEC: [{WINDOW_START_SEC} → {WINDOW_END_SEC}]\n\n"
    "Known info from previous windows (may be empty):\n"
    "{KNOWN_INFO_PREV}\n\n"
    "Frames in this window (id | t_sec | caption) — TOTAL_FRAMES_IN_WINDOW = {FRAME_COUNT}:\n"
    "{FRAME_LIST}\n\n"
    "Output using the sentinel-tagged skeleton provided below. Fill ONLY the 'reasoning:' lines for each frame, write ≤2 sentences per frame; then write ≤3 time‑anchored bullets and the STATS. Do NOT add or remove any ids/lines or extra text outside the tags."
)


@dataclass
class WindowParams:
    W: int = 16
    stride: Optional[int] = None  # default to W for non-overlapping
    retries: int = 2


def build_frame_lines(frames_window: List[Dict]) -> str:
    lines = []
    for f in frames_window:
        t = float(f["t_sec"]) if isinstance(f["t_sec"], (int, float, str)) else 0.0
        cap = str(f["caption"]).strip()
        lines.append(f'{f["id"]} | {t:.2f} | {cap} [t={t:.2f} s]')
    return "\n".join(lines)


def build_frame_skeleton(frames_window: List[Dict]) -> str:
    """Produce a strict frames section skeleton with ids in order and empty reasoning fields."""
    lines = ["<<<BEGIN_FRAMES>>>", "# FRAMES"]
    for f in frames_window:
        fid = int(f["id"])
        lines.append(f"id: {fid}")
        lines.append("reasoning: ")
        lines.append("")
    lines.append("<<<END_FRAMES>>>")
    lines.append("")
    lines.append("<<<BEGIN_KNOWN_INFO_UPDATE>>>")
    lines.append("# KNOWN_INFO_UPDATE")
    lines.append("- ")
    lines.append("- ")
    lines.append("- ")
    lines.append("<<<END_KNOWN_INFO_UPDATE>>>")
    lines.append("")
    lines.append("<<<BEGIN_STATS>>>")
    lines.append("# STATS")
    lines.append(f"FRAME_COUNT_EMITTED: {len(frames_window)}")
    # Window span must be filled exactly by the model; keep as the exact value to echo
    lines.append("WINDOW_SPAN_ECHO: [{WINDOW_SPAN}]")
    lines.append("<<<END_STATS>>>")
    return "\n".join(lines)


def _expand_ids_token(token: str) -> List[int]:
    """Expand tokens like '20-21', '28, 29', '97..111', '[12]', '(13)', 'ids 1 to 16' into a flat id list."""
    s = token.strip()
    # normalize brackets and words
    s = re.sub(r"[\[\]\(\)]", " ", s)
    s = re.sub(r"(?i)\bids?\b", " ", s)
    s = re.sub(r"(?i)\bframes?\b", " ", s)
    s = re.sub(r"(?i)\bfid\b", " ", s)
    s = re.sub(r"(?i)\bto\b", "-", s)
    s = s.replace("–", "-").replace("—", "-").replace("~", "-")
    s = s.replace("..", "-")
    # split by comma/Chinese comma
    parts = re.split(r"[,\uFF0C]+", s)
    out: List[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*(-?\d+)\s*\-\s*(-?\d+)\s*$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(b, a + 1)))
        else:
            m2 = re.match(r"^\s*(-?\d+)\s*$", part)
            if m2:
                out.append(int(m2.group(1)))
    # deduplicate while preserving order
    seen: set = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _extract_id_reason_pairs(text: str) -> Dict[int, str]:
    """Extract (id, reasoning) pairs; accepts messy variants and id lists/ranges.
    Returns an insertion‑ordered dict; if an id appears multiple times, the first wins.
    """
    block_re = re.compile(
        r"(?mis)^\s*(?:id|ids?|frame|frames?|fid)\s*:\s*([^\n\r]+?)\s*" \
        r"(?:\n+|\s+)(?:reason(?:ing)?|rationale)?\s*:\s*(.*?)(?=^\s*(?:id|ids?|frame|frames?|fid)\s*:|\Z)"
    )
    out: Dict[int, str] = {}
    for m in block_re.finditer(text):
        ids_token = m.group(1)
        reason = m.group(2).strip()
        ids = _expand_ids_token(ids_token)
        if not reason:
            continue
        for fid in ids:
            if fid not in out:
                out[fid] = reason
    return out


def _normalize_frame_ids_by_shift(frames_map: Dict[int, str], expected_ids_in_order: List[int]) -> Tuple[Dict[int, str], Optional[int]]:
    """Map model ids to expected ids by a global constant shift if possible.
    Returns (normalized_map, shift). shift is the amount ADDED to model ids to match expected.
    If not possible, returns ({}, None).
    """
    expected_set = set(expected_ids_in_order)
    keys = list(frames_map.keys())
    if set(keys) == expected_set:
        return dict(frames_map), 0
    if not keys:
        return {}, None
    min_k, max_k = min(keys), max(keys)
    # If model ids form a contiguous block of same length, align starts
    if len(keys) == len(expected_ids_in_order) and max_k - min_k + 1 == len(keys):
        shift = expected_ids_in_order[0] - min_k
        shifted = {k + shift: v for k, v in frames_map.items()}
        if set(shifted.keys()) == expected_set:
            return shifted, shift
    # Fallback: explicit off-by-one
    if {k - 1 for k in keys} == expected_set:
        return ({k - 1: v for k, v in frames_map.items()}, -1)
    if {k + 1 for k in keys} == expected_set:
        return ({k + 1: v for k, v in frames_map.items()}, +1)
    return {}, None


def _remap_in_order(frames_map: Dict[int, str], expected_ids_in_order: List[int]) -> Optional[Dict[int, str]]:
    """Ignore provided ids and map by appearance order to expected ids (e.g., model wrote 1..W per window).
    Only used when counts match.
    """
    if len(frames_map) != len(expected_ids_in_order):
        return None
    vals = list(frames_map.values())
    if len(vals) != len(expected_ids_in_order):
        return None
    return {expected_ids_in_order[i]: vals[i] for i in range(len(expected_ids_in_order))}


def parse_sections(text: str) -> Tuple[str, str, str]:
    """Parse output into frames/known-info/stats blocks.
    Supports both new sentinel tags and legacy "# SECTION" anchors. Tolerates code fences.
    """
    # Strip code fences if present
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "\n", text.strip(), flags=re.MULTILINE)
    # Prefer sentinel tags
    b1 = text.find("<<<BEGIN_FRAMES>>>")
    e1 = text.find("<<<END_FRAMES>>>")
    b2 = text.find("<<<BEGIN_KNOWN_INFO_UPDATE>>>")
    e2 = text.find("<<<END_KNOWN_INFO_UPDATE>>>")
    b3 = text.find("<<<BEGIN_STATS>>>")
    e3 = text.find("<<<END_STATS>>>")
    if all(x >= 0 for x in [b1, e1, b2, e2, b3, e3]):
        s1 = text[b1 + len("<<<BEGIN_FRAMES>>>"):e1].strip()
        s2 = text[b2 + len("<<<BEGIN_KNOWN_INFO_UPDATE>>>"):e2].strip()
        s3 = text[b3 + len("<<<BEGIN_STATS>>>"):e3].strip()
        return s1, s2, s3
    # Legacy fallback
    m1 = re.search(r"#\s*FRAMES", text, re.IGNORECASE)
    m2 = re.search(r"#\s*KNOWN_INFO_UPDATE", text, re.IGNORECASE)
    m3 = re.search(r"#\s*STATS", text, re.IGNORECASE)
    if not (m1 and m2 and m3):
        raise ValueError("Missing required sections (# FRAMES / # KNOWN_INFO_UPDATE / # STATS)")
    s1 = text[m1.end():m2.start()].strip()
    s2 = text[m2.end():m3.start()].strip()
    s3 = text[m3.end():].strip()
    return s1, s2, s3


def parse_frames_block(frames_block: str, expected_ids_in_order: List[int], expected_count: int) -> Dict[int, str]:
    """Strict parser for the frames block; expects pairs of id/reasoning lines.
    Raises on count/order mismatches. Use salvage_parse_frames_anywhere for recovery.
    """
    lines = [ln.strip() for ln in frames_block.splitlines() if ln.strip()]
    out: Dict[int, str] = {}
    i = 0
    parsed_pairs = 0
    while i < len(lines):
        if not lines[i].lower().startswith("id:"):
            raise ValueError(f"Line {i+1} does not start with 'id:'")
        try:
            fid = int(lines[i].split(":", 1)[1].strip().strip("<>").strip())
        except Exception:
            m = re.search(r"(-?\d+)", lines[i])
            if not m:
                raise
            fid = int(m.group(1))
        i += 1
        if i >= len(lines) or not lines[i].lower().startswith("reasoning:"):
            raise ValueError("Missing 'reasoning:' after id")
        reasoning = lines[i].split(":", 1)[1].strip()
        out[fid] = reasoning
        parsed_pairs += 1
        i += 1
    if parsed_pairs != expected_count:
        raise ValueError(f"FRAME_COUNT mismatch: got {parsed_pairs}, expect {expected_count}")
    for fid in expected_ids_in_order:
        if fid not in out:
            raise ValueError(f"Missing frame id {fid}")
    return out


def salvage_parse_frames_anywhere(text: str, expected_ids_in_order: List[int]) -> Dict[int, str]:
    """Best-effort extraction of (id, reasoning) pairs anywhere in text.
    Accepts messy outputs; preserves expected order using expected_ids_in_order.
    - Accepts id synonyms and ranges/commas; optional reasoning prefix.
    """
    # Broadly find id blocks including ranges/commas
    id_line = re.compile(r'(?mi)^(?:\s*(?:id|ids?|frame|frames?|fid)\s*[:#]?\s*|[\[\(]?\s*)([^\n\r]+?)\s*[\]\)]?\s*(?:[:\-]|$)')
    matches = list(id_line.finditer(text))
    temp: Dict[int, str] = {}
    if not matches:
        return {}

    for i, m in enumerate(matches):
        token = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        # optional reasoning prefix
        chunk = re.sub(r'(?is)^\s*(reason(?:ing)?|rationale)\s*[:\-]\s*', '', chunk).strip()
        # keep first line if verbose
        if '\n' in chunk:
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            if lines:
                chunk = lines[0]
        ids = _expand_ids_token(token)
        for fid in ids:
            if fid not in temp and chunk:
                temp[fid] = chunk

    out: Dict[int, str] = {}
    for fid in expected_ids_in_order:
        if fid in temp:
            out[fid] = temp[fid]
    return out


# ===== Normalization: enforce [LOW] at prefix and keep time-anchor at end =====
_TIME_ANCHOR_RE = re.compile(r"\[\s*t\s*=\s*\d+(?:\.\d+)?(?:\s*→\s*\d+(?:\.\d+)?)?\s*s\s*\]", re.IGNORECASE)
_LOW_BRACKET_RE = re.compile(r"\[\s*low\b[^\]]*\]", re.IGNORECASE)


def normalize_low_tag(reasoning: str) -> str:
    """Normalize [LOW] tag to prefix form; keep time anchor at the end.
    - Accepts variants like '[LOW]', '[low]', '[LOW, t=49.40 s]', '... [LOW]' anywhere.
    - Moves [LOW] to prefix '[LOW] ' once; extracts time anchor '[t=.. s]' and puts it at the end.
    Idempotent for already-normalized lines.
    """
    if not isinstance(reasoning, str):
        return reasoning
    s = reasoning.strip()
    if not s:
        return s
    # 1) Extract the last time anchor (if any), remove from body
    anchors = list(_TIME_ANCHOR_RE.finditer(s))
    time_anchor = anchors[-1].group(0) if anchors else ""
    if time_anchor:
        s = _TIME_ANCHOR_RE.sub("", s).strip()
    # 2) Detect and remove any [LOW ...] brackets anywhere
    low_found = False
    if _LOW_BRACKET_RE.search(s):
        low_found = True
        s = _LOW_BRACKET_RE.sub("", s).strip()
    # 3) Also handle prefix '[LOW]' already present to avoid duplication
    if s.startswith("[LOW]"):
        low_found = True
        s = s[5:].lstrip()
    # 4) Compose
    head = "[LOW] " if low_found else ""
    out = f"{head}{s}".strip()
    if time_anchor and not out.endswith(time_anchor):
        out = f"{out} {time_anchor}"
    # 5) Compress whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _finalize_frames_reasoning(frames_out: Dict[int, str]) -> Dict[int, str]:
    for k in list(frames_out.keys()):
        try:
            frames_out[k] = normalize_low_tag(frames_out[k])
        except Exception:
            pass
    return frames_out


def parse_known_info_block(ki_block: str) -> List[str]:
    bullets = []
    for ln in ki_block.splitlines():
        ln = ln.strip()
        if ln.startswith("- ") or ln.startswith("* "):
            bullets.append(ln[2:].strip())
        else:
            # Allow "1. ..." enumeration
            m = re.match(r"\d+\.\s+(.*)", ln)
            if m:
                bullets.append(m.group(1).strip())
    return bullets


def parse_stats_block(stats_block: str) -> Tuple[int, str]:
    fc_match = re.search(r"FRAME_COUNT_EMITTED:\s*(\d+)", stats_block)
    span_match = re.search(r"WINDOW_SPAN_ECHO:\s*\[(.*?)\]", stats_block)
    if not fc_match or not span_match:
        raise ValueError("Invalid # STATS block")
    return int(fc_match.group(1)), span_match.group(1).strip()


def ensure_time_anchor(bullets: List[str], default_t: float) -> List[str]:
    out: List[str] = []
    for b in bullets[:3]:
        if ("[t=" in b and "s]" in b) or ("[phase=" in b):
            out.append(b)
        else:
            out.append(f"{b} [t={default_t:.2f} s]")
    return out[:3]


# Prefer interval-style anchoring: if a bullet lacks any anchor, add [t=start→end s]
def ensure_time_anchor_interval(bullets: List[str], start_t: float, end_t: float, limit: int = 3) -> List[str]:
    out: List[str] = []
    for b in bullets[:limit]:
        if ("[t=" in b and "s]" in b) or ("[phase=" in b):
            out.append(b)
        else:
            out.append(f"{b} [t={start_t:.2f}→{end_t:.2f} s]")
    return out


# Helpers for aggregating known info into interval summaries
_time_range_re = re.compile(r"\[t\s*=\s*([0-9]*\.?[0-9]+)\s*→\s*([0-9]*\.?[0-9]+)\s*s\]", re.IGNORECASE)
_time_point_re = re.compile(r"\[t\s*=\s*([0-9]*\.?[0-9]+)\s*s\]", re.IGNORECASE)
_phase_range_re = re.compile(r"\[phase\s*=\s*(\d+)\s*→\s*(\d+)\s*\]", re.IGNORECASE)
_phase_point_re = re.compile(r"\[phase\s*=\s*(\d+)\s*\]", re.IGNORECASE)
_anchor_strip_re = re.compile(r"\s*\[(?:t\s*=[^\]]+|phase\s*=[^\]]+)\]\s*", re.IGNORECASE)


def _extract_anchors(text: str, fallback_start: float, fallback_end: float) -> Tuple[float, float, Optional[int], Optional[int]]:
    """Extract time/phase anchors as (t_start, t_end, phase_min, phase_max)."""
    m = _time_range_re.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a if a <= b else b), (b if b >= a else a), None, None
    m = _time_point_re.search(text)
    if m:
        x = float(m.group(1))
        return x, x, None, None
    m = _phase_range_re.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return fallback_start, fallback_end, (a if a <= b else b), (b if b >= a else a)
    m = _phase_point_re.search(text)
    if m:
        k = int(m.group(1))
        return fallback_start, fallback_end, k, k
    # No anchors: fallback to window interval
    return fallback_start, fallback_end, None, None


def _topic_key(text: str) -> str:
    """Normalize bullet text by stripping anchors and lowercasing to form a topic key."""
    base = _anchor_strip_re.sub(" ", text).strip()
    return re.sub(r"\s+", " ", base).lower()


@dataclass
class _KnownRec:
    base_text: str
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    p_min: Optional[int] = None
    p_max: Optional[int] = None
    count: int = 0
    last_seen_window: int = -1


def _update_known_store(store: Dict[str, _KnownRec], bullets: List[str],
                        w_idx: int, start_t: float, end_t: float) -> None:
    for b in bullets:
        key = _topic_key(b)
        if not key:
            continue
        t0, t1, p0, p1 = _extract_anchors(b, start_t, end_t)
        rec = store.get(key)
        if rec is None:
            rec = _KnownRec(base_text=_anchor_strip_re.sub(" ", b).strip())
            store[key] = rec
        # Merge time intervals
        if t0 is not None and t1 is not None:
            rec.t_start = t0 if rec.t_start is None else min(rec.t_start, t0)
            rec.t_end = t1 if rec.t_end is None else max(rec.t_end, t1)
        # Merge phases
        if p0 is not None and p1 is not None:
            rec.p_min = p0 if rec.p_min is None else min(rec.p_min, p0)
            rec.p_max = p1 if rec.p_max is None else max(rec.p_max, p1)
        rec.count += 1
        rec.last_seen_window = max(rec.last_seen_window, w_idx)


def _render_known_summary(store: Dict[str, _KnownRec], max_items: int = 6) -> List[str]:
    # Prefer recently updated and frequently seen
    items = sorted(store.values(), key=lambda r: (r.last_seen_window, r.count), reverse=True)
    out: List[str] = []
    for r in items[:max_items]:
        anchor = ""
        if r.t_start is not None and r.t_end is not None:
            if abs(r.t_end - r.t_start) < 1e-3:
                anchor = f"[t={r.t_start:.2f} s]"
            else:
                anchor = f"[t={r.t_start:.2f}→{r.t_end:.2f} s]"
        elif r.p_min is not None and r.p_max is not None:
            anchor = f"[phase={r.p_min}→{r.p_max}]" if r.p_min != r.p_max else f"[phase={r.p_min}]"
        base_clean = _anchor_strip_re.sub(" ", r.base_text).strip()
        out.append(f"{base_clean} {anchor}".strip())
    return out


def windows(frames_all: List[Dict], W: int, stride: int) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for i in range(0, len(frames_all), stride):
        out.append(frames_all[i:i + W])
    return out


class MistralGenerator:
    def __init__(self, model_name: str, device: str = "cuda:0", hf_token: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tok_kwargs = {}
        mdl_kwargs = {"torch_dtype": torch_dtype}
        if self.hf_token:
            tok_kwargs["token"] = self.hf_token
            mdl_kwargs["token"] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)
        if torch.cuda.is_available():
            self.model.to(device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, *, max_new_tokens: int = 1024,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 1234) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            torch.manual_seed(seed)
            do_sample = temperature is not None and float(temperature) > 0.0
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs.update({
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                })
            output_ids = self.model.generate(**gen_kwargs)
        gen_ids = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


class ReasoningEngine:
    def __init__(self, generator: MistralGenerator, params: WindowParams) -> None:
        self.generator = generator
        self.params = params

    @staticmethod
    def _build_user_prompt(Q: str, video_duration: float, idx: int, total: int,
                           start: float, end: float, known_prev: List[str],
                           frames_window: List[Dict]) -> str:
        ki_prev = "\n".join([f"- {s}" for s in known_prev]) if known_prev else "(empty)"
        frame_lines = build_frame_lines(frames_window)
        return USER_TEMPLATE.format(
            Q=Q,
            VIDEO_DURATION_SEC=f"{video_duration:.2f}",
            WINDOW_INDEX=idx + 1,
            TOTAL_WINDOWS=total,
            WINDOW_START_SEC=f"{start:.2f}",
            WINDOW_END_SEC=f"{end:.2f}",
            KNOWN_INFO_PREV=ki_prev,
            FRAME_COUNT=len(frames_window),
            FRAME_LIST=frame_lines,
        )

    def run(self, Q: str, video_duration: float, frames_all: List[Dict],
            max_new_tokens_hint: int = 1024, seed: int = 1234,
            on_update: Optional[Callable[[Dict[int, str], List[str], int, int], None]] = None,
            start_window: int = 0,
            initial_known: Optional[List[str]] = None,
            initial_reasoning: Optional[Dict[int, str]] = None,
            on_parse_failure: Optional[Callable[[int, int], None]] = None,
            log_context: Optional[str] = None,
            on_raw_response: Optional[Callable[[int, int, str, str], None]] = None,
            ) -> Tuple[Dict[str, str], List[str]]:
        frames_all_sorted = sorted(frames_all, key=lambda x: (float(x["t_sec"]), int(x["id"])) )
        W = self.params.W
        stride = self.params.stride if self.params.stride is not None else W
        wins = windows(frames_all_sorted, W, stride)
        total_windows = len(wins) if wins else 0
        # Aggregated rolling summary store and rendered list
        known_store: Dict[str, _KnownRec] = {}
        # Seed with initial known info if provided (fallback interval: whole video)
        if initial_known:
            try:
                _update_known_store(known_store, list(initial_known), -1, 0.0, float(video_duration or 0.0))
            except Exception:
                pass
        known_prev: List[str] = _render_known_summary(known_store, max_items=6)
        # Prefill reasoning map if resuming
        reasoning_map: Dict[int, str] = {}
        if initial_reasoning:
            try:
                reasoning_map.update({int(k): v for k, v in initial_reasoning.items() if isinstance(v, str) and v.strip() != ""})
            except Exception:
                pass

        for w_idx, w_frames in enumerate(wins):
            if w_idx < int(start_window):
                continue
            if not w_frames:
                continue
            start_t = float(w_frames[0]["t_sec"]) if w_frames else 0.0
            end_t = float(w_frames[-1]["t_sec"]) if w_frames else start_t
            expected_ids_order = [int(f["id"]) for f in w_frames]
            expected_count = len(w_frames)

            user_prompt = self._build_user_prompt(
                Q=Q,
                video_duration=video_duration,
                idx=w_idx,
                total=total_windows,
                start=start_t,
                end=end_t,
                known_prev=known_prev,
                frames_window=w_frames,
            )

            # heuristic token budget
            max_new = max(max_new_tokens_hint, expected_count * 38 + 128)

            frames_out: Dict[int, str] = {}
            ki_update: List[str] = []
            parsed_ok = False
            last_resp: Optional[str] = None
            for attempt in range(self.params.retries + 1):
                # From attempt 0 use skeleton; attempt 1 adds a reminder; attempt >=2 repeats skeleton
                stage = "skeleton" if attempt == 0 else ("skeleton+reminder" if attempt == 1 else "skeleton")
                skeleton = build_frame_skeleton(w_frames).replace("{WINDOW_SPAN}", f"{start_t:.2f} → {end_t:.2f}")
                prompt_now = (
                    user_prompt
                    + ("\n\n[STRICT REMINDER] Keep EXACTLY the provided skeleton: fill only 'reasoning:' lines, 3 bullets, and STATS; no extra text or ids." if attempt == 1 else "")
                    + "\n\n[FORMAT SKELETON — FILL ONLY]"\
                    + "\n" + skeleton
                )
                # Print attempt stage
                if log_context:
                    print(f"[{log_context}] window {w_idx+1}/{total_windows}: attempt {attempt} ({stage})", flush=True)
                resp = self.generator.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt_now,
                    max_new_tokens=max_new,
                    temperature=0.0,
                    top_p=1.0,
                    seed=seed,
                )
                # record raw response per attempt
                if on_raw_response is not None:
                    try:
                        on_raw_response(w_idx, attempt, stage, resp)
                    except Exception:
                        pass
                last_resp = resp
                try:
                    fb, kib, sb = parse_sections(resp)
                    # First try strict parsing
                    frames_out = parse_frames_block(fb, expected_ids_order, expected_count)
                    # If parsed strictly, also accept as-is
                    fc_emit, span_echo = parse_stats_block(sb)
                    if fc_emit != expected_count:
                        raise ValueError(f"FRAME_COUNT_EMITTED {fc_emit} != {expected_count}")
                    span_expect = f"{start_t:.2f} → {end_t:.2f}"
                    if span_echo.replace(" ", "") != span_expect.replace(" ", ""):
                        raise ValueError(f"WINDOW_SPAN_ECHO '{span_echo}' != '{span_expect}'")
                    ki_update = parse_known_info_block(kib)
                    # Prefer interval-style anchors within the current window
                    ki_update = ensure_time_anchor_interval(ki_update, start_t, end_t)
                    parsed_ok = True
                    break
                except Exception:
                    # Lenient path: extract pairs and normalize id shift (0..31 vs 1..32)
                    try:
                        fb2, kib2, sb2 = parse_sections(resp)
                        raw_pairs = _extract_id_reason_pairs(fb2)
                        norm_map, shift = _normalize_frame_ids_by_shift(raw_pairs, expected_ids_order)
                        if shift is not None and len(norm_map) == expected_count:
                            fc_emit2, span_echo2 = parse_stats_block(sb2)
                            if fc_emit2 == expected_count:
                                span_expect2 = f"{start_t:.2f} → {end_t:.2f}"
                                if span_echo2.replace(" ", "") == span_expect2.replace(" ", ""):
                                    frames_out = {fid: norm_map[fid] for fid in expected_ids_order}
                                    ki_update = ensure_time_anchor_interval(parse_known_info_block(kib2), start_t, end_t)
                                    parsed_ok = True
                                    break
                    except Exception:
                        pass
                    continue

            if not parsed_ok:
                # Try salvage parse from the last response (if any)
                if last_resp is not None:
                    # First, try raw extraction + id-shift normalization
                    raw_pairs_all = _extract_id_reason_pairs(last_resp)
                    norm_map, shift = _normalize_frame_ids_by_shift(raw_pairs_all, expected_ids_order)
                    if shift is not None:
                        if log_context:
                            print(f"[{log_context}] window {w_idx+1}/{total_windows}: parse failed, salvage via id-shift (shift={shift}, recovered={len(norm_map)}/{expected_count})", flush=True)
                        for fid in expected_ids_order:
                            if fid in norm_map and str(norm_map[fid]).strip():
                                frames_out[fid] = norm_map[fid]
                    # If still incomplete and counts match, try order-based remap (1..W → expected ids)
                    if len(frames_out) < expected_count and len(raw_pairs_all) == expected_count:
                        remap = _remap_in_order(raw_pairs_all, expected_ids_order)
                        if remap:
                            if log_context:
                                print(f"[{log_context}] window {w_idx+1}/{total_windows}: salvage via order-remap (recovered={len(remap)}/{expected_count})", flush=True)
                            frames_out = remap
                    # Finally, general lenient salvage
                    if len(frames_out) < expected_count:
                        frames_salv = salvage_parse_frames_anywhere(last_resp, expected_ids_order)
                        if log_context:
                            print(f"[{log_context}] window {w_idx+1}/{total_windows}: parse failed, trying salvage (found {len(frames_salv)} frames)", flush=True)
                        for fid in expected_ids_order:
                            if fid in frames_salv and frames_salv[fid].strip():
                                frames_out[fid] = frames_salv[fid]
                # minimal repair: fill missing with [LOW]
                for fid in expected_ids_order:
                    frames_out[fid] = frames_out.get(fid, "[LOW] Minimal fallback reasoning due to formatting failure.")
                if log_context:
                    missing_cnt = sum(1 for fid in expected_ids_order if frames_out.get(fid, "").strip().startswith("[LOW] Minimal fallback"))
                    print(f"[{log_context}] window {w_idx+1}/{total_windows}: auto-filled {missing_cnt} frames with [LOW] fallback", flush=True)
                if not ki_update:
                    # Attempt to salvage bullets from anywhere between known-info sentinels or header
                    try:
                        fb2, kib2, sb2 = parse_sections(last_resp or "")
                        ki_update = ensure_time_anchor_interval(parse_known_info_block(kib2), start_t, end_t)
                    except Exception:
                        ki_update = [f"Window {w_idx+1} parsed with fallback [t={start_t:.2f}→{end_t:.2f} s]"]
                # notify caller about parse failure for logging
                if on_parse_failure is not None:
                    try:
                        # Only log as failure if we actually had to auto-fill any frames
                        missing_cnt = sum(1 for fid in expected_ids_order if frames_out.get(fid, "").strip().startswith("[LOW] Minimal fallback"))
                        if missing_cnt > 0:
                            on_parse_failure(w_idx, total_windows)
                    except Exception:
                        pass

            # Normalize LOW tags before merging
            frames_out = _finalize_frames_reasoning(frames_out)

            # merge
            for fid in expected_ids_order:
                reasoning_map[fid] = frames_out[fid]

            # Update aggregated store and render rolling summary (≤ 6 items)
            _update_known_store(known_store, ki_update, w_idx, start_t, end_t)
            known_prev = _render_known_summary(known_store, max_items=6)

            # incremental callback after each window
            if on_update is not None:
                try:
                    on_update(dict(reasoning_map), list(known_prev), w_idx, total_windows)
                except Exception:
                    pass

        reasoning_dict = {str(k): v for k, v in sorted(reasoning_map.items(), key=lambda x: x[0])}
        return reasoning_dict, known_prev


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def list_input_files(input_dir: str) -> List[str]:
    return [f for f in os.listdir(input_dir) if f.endswith('.json')]


def read_caption_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_frames_meta_from_caption(caption_obj: Dict) -> Tuple[List[Dict], float]:
    frames_meta: List[Dict] = []
    frames = caption_obj.get('frames', {})
    for k, v in frames.items():
        try:
            idx = int(v.get('frame_index', int(k)))
        except Exception:
            idx = int(k)
        frames_meta.append({
            'id': idx,
            't_sec': float(v.get('ts', 0.0)),
            'caption': str(v.get('caption', '')),
        })
    frames_meta.sort(key=lambda x: (x['t_sec'], x['id']))
    video_duration = float(caption_obj.get('metadata', {}).get('video_duration', 0.0))
    return frames_meta, video_duration


def _canonical_video_keys(name: str) -> List[str]:
    base = os.path.basename(name)
    no_json = base[:-5] if base.endswith('.json') else base
    no_ext = os.path.splitext(no_json)[0]
    return [no_json, no_ext, f"{no_ext}.mp4", f"{no_ext}.MP4"]


def load_videomme_questions(split: str = "test") -> Dict[str, List[Dict[str, str]]]:
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    mapping: Dict[str, List[Dict[str, str]]] = {}
    for ex in ds:
        # Prefer official Video-MME schema fields
        vid = ex.get("videoID") or ex.get("video_id") or ex.get("video") or ex.get("video_name")
        q = ex.get("question") or ex.get("Question")
        qid = ex.get("question_id")  # expected like '001-1'
        if vid is None or q is None or qid is None:
            continue
        keys = _canonical_video_keys(vid)
        for k in keys:
            mapping.setdefault(k, []).append({"question_id": str(qid), "question": str(q)})
    return mapping


def process_files_worker(worker_id: int, files: List[str], input_dir: str, output_dir: str,
                         params: WindowParams, model_name: str, device_id: int,
                         hf_token: Optional[str],
                         questions_map: Dict[str, List[Dict[str, str]]],
                         progress_queue: Optional[mp.Queue] = None, overwrite: bool = False,
                         max_new_tokens_hint: int = 1024):
    # Bind a physical GPU per worker (avoid post-import CUDA_VISIBLE_DEVICES)
    if torch.cuda.is_available() and device_id is not None and device_id >= 0 and device_id < torch.cuda.device_count():
        try:
            torch.cuda.set_device(device_id)
        except Exception as e:
            print(f"[W{worker_id}] Warning: torch.cuda.set_device({device_id}) failed: {e}", flush=True)
        device = f"cuda:{device_id}"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        device = "cpu"

    print(f"[W{worker_id}] Initializing model '{model_name}' on {device}…", flush=True)
    generator = MistralGenerator(model_name=model_name, device=device, hf_token=hf_token)
    engine = ReasoningEngine(generator=generator, params=params)
    print(f"[W{worker_id}] Model ready.", flush=True)

    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        base_keys = _canonical_video_keys(fname)
        qs: List[Dict[str, str]] = []
        for k in base_keys:
            if k in questions_map:
                qs = questions_map[k]
                break
        if not qs:
            # If no questions found, skip gracefully
            if progress_queue:
                progress_queue.put({'worker_id': worker_id, 'file': fname, 'status': 'skipped'})
            continue

        t0 = time.time()
        try:
            cap = read_caption_json(in_path)
            frames_meta, video_duration = build_frames_meta_from_caption(cap)

            print(f"[W{worker_id}] ▶️ Video {fname} | frames={len(frames_meta)} | questions={len(qs)}", flush=True)

            # For each question associated with this video, run reasoning and write separate file
            for q_obj in qs:
                question = q_obj['question']
                question_id = q_obj['question_id']  # keep as-is, e.g., '001-1'

                # Output path: output_dir/<video_base_without_ext>/<question_id>.json
                base_no_json = os.path.splitext(os.path.basename(fname))[0]  # e.g., N1cdUjctpG8.mp4
                video_base = os.path.splitext(base_no_json)[0]               # -> N1cdUjctpG8
                out_dir_video = os.path.join(output_dir, video_base)
                os.makedirs(out_dir_video, exist_ok=True)
                out_path = os.path.join(out_dir_video, f"{question_id}.json")

                # Resume/skip logic: if output exists and not overwrite, check completeness
                resume_known: Optional[List[str]] = None
                resume_map: Optional[Dict[int, str]] = None
                resume_start_window: int = 0
                if os.path.exists(out_path):
                    try:
                        with open(out_path, 'r', encoding='utf-8') as fr:
                            prev = json.load(fr)
                        # known info to seed
                        resume_known = prev.get('reasoning_stage1', {}).get('known_info_all')
                        # prefill existing per-frame reasoning
                        frames_prev = prev.get('frames', {})
                        resume_map = {}
                        for fk, fv in frames_prev.items():
                            try:
                                fi = int(fv.get('frame_index', int(fk)))
                            except Exception:
                                continue
                            rv = fv.get('reasoning')
                            if isinstance(rv, str) and rv.strip() != "":
                                resume_map[fi] = rv
                        # determine total windows and first incomplete window
                        progress_w = prev.get('metadata', {}).get('progress_window_index')
                        # recompute windows from current frames_meta
                        wins_local = windows(frames_meta, params.W, params.stride if params.stride is not None else params.W)
                        total_w_calc = len(wins_local)
                        # heuristic: find first window that has any missing reasoning
                        resume_start_window = 0
                        for wi, wf in enumerate(wins_local):
                            missing = False
                            for f in wf:
                                if int(f['id']) not in resume_map:
                                    missing = True
                                    break
                            if missing:
                                resume_start_window = wi
                                break
                        else:
                            # all windows appear complete
                            if (not overwrite):
                                # skip fully completed
                                continue
                            resume_start_window = max(0, (progress_w or total_w_calc or 1) - 1)
                    except Exception:
                        pass

                print(f"[W{worker_id}]   🧠 Q {question_id}: starting reasoning…", flush=True)

                def write_partial(reasoning_map: Dict[int, str], known_info_now: List[str], w_idx: int, total_w: int):
                    # Build frames output with current reasoning
                    frames_out_local = {}
                    # ensure normalization on write
                    normalized_map = {int(k): normalize_low_tag(v) for k, v in reasoning_map.items()}
                    for fk, fv in cap.get('frames', {}).items():
                        fidx = fv.get('frame_index', int(fk))
                        frames_out_local[fk] = {
                            'ts': fv.get('ts'),
                            'ts_formatted': fv.get('ts_formatted', format_timestamp(float(fv.get('ts', 0.0)))) if fv.get('ts') is not None else None,
                            'caption': fv.get('caption', ''),
                            'frame_index': fidx,
                            'reasoning': normalized_map.get(int(fidx), "")
                        }

                    metadata_local = cap.get('metadata', {}).copy()
                    metadata_local.update({
                        'reasoning_model': model_name,
                        'reasoning_engine': 'transformers',
                        'window_size': params.W,
                        'stride': params.stride if params.stride is not None else params.W,
                        'retries': params.retries,
                        'question': question,
                        'question_id': question_id,
                        'processed_reasoning_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'progress_window_index': w_idx + 1,
                        'progress_total_windows': total_w,
                    })

                    final_obj_local = {
                        'metadata': metadata_local,
                        'frames': frames_out_local,
                        'reasoning_stage1': {
                            'known_info_all': known_info_now
                        }
                    }

                    tmp_path = out_path + ".tmp"
                    with open(tmp_path, 'w', encoding='utf-8') as ftmp:
                        json.dump(final_obj_local, ftmp, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, out_path)  # atomic
                    print(f"[W{worker_id}]   💾 Q {question_id}: saved window {w_idx+1}/{total_w}", flush=True)

                # Failure logger: writes a line when a window couldn't be parsed cleanly
                os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)
                failure_log_path = os.path.join(os.getcwd(), 'logs', 'reasoning_parse_failures.log')
                raw_log_dir = os.path.join(os.getcwd(), 'logs', 'raw')
                os.makedirs(raw_log_dir, exist_ok=True)
                raw_log_path = os.path.join(raw_log_dir, f"{video_base}__{question_id}.raw.txt")

                def log_parse_failure(w_idx: int, total_w: int):
                    try:
                        with open(failure_log_path, 'a', encoding='utf-8') as flog:
                            video_base = os.path.splitext(os.path.basename(fname))[0]
                            flog.write(f"video={video_base}, question_id={question_id}, window={w_idx+1}/{total_w}, time={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                    except Exception:
                        pass

                def append_raw_response(w_idx: int, attempt: int, stage: str, resp_text: str):
                    try:
                        with open(raw_log_path, 'a', encoding='utf-8') as fraw:
                            fraw.write(f"===== window {w_idx+1} attempt {attempt} ({stage}) @ {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
                            fraw.write(resp_text)
                            fraw.write("\n\n")
                    except Exception:
                        pass

                # run reasoning with incremental saving
                reasoning_dict, known_info = engine.run(
                    Q=question,
                    video_duration=video_duration,
                    frames_all=frames_meta,
                    max_new_tokens_hint=max_new_tokens_hint,
                    on_update=write_partial,
                    start_window=resume_start_window,
                    initial_known=resume_known,
                    initial_reasoning=resume_map,
                    on_parse_failure=log_parse_failure,
                    log_context=f"W{worker_id}|video={os.path.splitext(os.path.basename(fname))[0]}|qid={question_id}",
                    on_raw_response=append_raw_response,
                )

                # Ensure a final save (covers cases with no windows or last state)
                total_w = len(windows(frames_meta, params.W, params.stride if params.stride is not None else params.W))
                last_idx = max(0, total_w - 1)
                final_map = {int(k): v for k, v in reasoning_dict.items()}
                write_partial(final_map, known_info, w_idx=last_idx, total_w=total_w)
                print(f"[W{worker_id}]   ✅ Q {question_id}: done.", flush=True)

            dt = time.time() - t0
            if progress_queue:
                progress_queue.put({'worker_id': worker_id, 'file': fname, 'status': 'completed', 'time': dt})

        except Exception as e:
            if progress_queue:
                progress_queue.put({'worker_id': worker_id, 'file': fname, 'status': 'error', 'error': str(e)})

        # Light memory hygiene between videos
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if progress_queue:
        progress_queue.put({'worker_id': worker_id, 'status': 'worker_completed'})


def progress_monitor(total_files: int, num_workers: int, queue: mp.Queue):
    completed = skipped = errors = workers_done = 0
    times = []
    pbar = tqdm(total=total_files, desc='Reasoning', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    while workers_done < num_workers:
        try:
            msg = queue.get(timeout=1)
        except Exception:
            continue
        status = msg.get('status')
        if status == 'completed':
            completed += 1
            t = msg.get('time')
            if isinstance(t, (int, float)):
                times.append(t)
            pbar.update(1)
        elif status == 'skipped':
            skipped += 1
            pbar.update(1)
        elif status == 'error':
            errors += 1
            tqdm.write(f"❌ Worker {msg.get('worker_id')} error on {msg.get('file')}: {msg.get('error')}")
            pbar.update(1)
        elif status == 'worker_completed':
            workers_done += 1
            tqdm.write(f"🎉 Worker {msg.get('worker_id')} completed")
    pbar.close()
    if times:
        avg = sum(times)/len(times)
        tqdm.write(f"✅ Done. Completed {completed}, skipped {skipped}, errors {errors}. Avg {avg:.1f}s/file")
    else:
        tqdm.write(f"✅ Done. Completed {completed}, skipped {skipped}, errors {errors}.")


def main():
    parser = argparse.ArgumentParser(description='Stage-1 reasoning generator (Transformers Mistral-7B-Instruct)')
    parser.add_argument('--input_dir', type=str, default='/home/syh/work-d/test/video/lmms-eval/qwen_videomme_captions', help='Directory of caption JSONs')
    parser.add_argument('--output_dir', type=str, default='/home/syh/work-d/test/video/lmms-eval/videomme_reasoning_mistral', help='Where to write reasoning JSONs')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of parallel workers (logical GPUs)')
    parser.add_argument('--window_size', type=int, default=16, help='Window size W')
    parser.add_argument('--stride', type=int, default=None, help='Stride (default=W, non-overlapping)')
    parser.add_argument('--retries', type=int, default=2, help='Retries per window on parse/validation failure')
    parser.add_argument('--max_new_tokens_hint', type=int, default=1024, help='Base max token hint per window')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='HF model name')
    parser.add_argument('--split', type=str, default='test', help='Video-MME split to load (e.g., test)')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for gated/private models (optional). If not set, will fallback to env HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = list_input_files(args.input_dir)
    print(f"📄 Found {len(files)} caption JSONs in {args.input_dir}")
    if not files:
        print("No input files found.")
        return

    # Load questions mapping from Video-MME
    print(f"🔎 Loading Video-MME questions (split={args.split})…")
    questions_map = load_videomme_questions(args.split)
    print(f"🧩 Loaded questions for {len(questions_map)} video keys")

    # Decide worker count based on requested GPUs and availability
    if torch.cuda.is_available():
        avail_gpus = torch.cuda.device_count()
    else:
        avail_gpus = 0
    if int(args.num_gpus) > 0 and avail_gpus == 0:
        print("⚠️  CUDA 不可用，将在 CPU 上顺序运行。", flush=True)
    num_workers = max(1, int(args.num_gpus))
    if avail_gpus > 0:
        num_workers = min(num_workers, avail_gpus, len(files) if files else 1)
        gpu_ids = list(range(num_workers))
        print(f"🖥️  Using {num_workers} GPU(s): {gpu_ids}", flush=True)
    else:
        num_workers = 1
        gpu_ids = [-1]
    chunk = len(files) // num_workers
    rem = len(files) % num_workers
    chunks: List[List[str]] = []
    start = 0
    for i in range(num_workers):
        sz = chunk + (1 if i < rem else 0)
        end = start + sz
        chunks.append(files[start:end])
        print(f"Worker {i}: {len(chunks[i])} files")
        start = end

    params = WindowParams(W=args.window_size, stride=(args.stride if args.stride is not None else args.window_size), retries=args.retries)

    # resolve HF token
    hf_token = args.hf_token or os.environ.get('HUGGINGFACEHUB_API_TOKEN') or os.environ.get('HF_TOKEN')

    queue: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    for i in range(num_workers):
        if not chunks[i]:
            continue
        device_id = gpu_ids[i] if i < len(gpu_ids) else (-1 if avail_gpus == 0 else (i % avail_gpus))
        p = mp.Process(target=process_files_worker, args=(
            i, chunks[i], args.input_dir, args.output_dir,
            params, args.model, device_id,
            hf_token,
            questions_map,
            queue, args.overwrite, args.max_new_tokens_hint
        ))
        p.start()
        procs.append(p)

    monitor = mp.Process(target=progress_monitor, args=(len(files), len(procs), queue))
    monitor.start()

    for p in procs:
        p.join()
    monitor.join()
    print("🏁 All workers completed!")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
