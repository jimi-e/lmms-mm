"""
Standalone caption-only runner for FrameSearch.

Goals:
- Do NOT load LLaVA; only run the caption stage using the same logic, logs, and JSON schema as mm_method.frame_search.
- Write caches under mm_method_cache/<vid>/<vid>_caption.json identical to frame_search.
- Designed to be reused later by reasoning/scoring stages without modification.

Usage examples:
  source ./eval/bin/activate
    python -m mm_method.fs_caption_only \
    --videos /home/syh/.cache/huggingface/videomme/data/TGom0uiW130.mp4 \
    --cache-root mm_method_cache \
    --device cuda \
    --caption-backend hf \
        --caption-model Qwen/Qwen2.5-VL-7B-Instruct

  # Batch via glob or scanning a root directory
  python -m mm_method.fs_caption_only --videos '/path/to/videos/*.mp4' --cache-root mm_method_cache
  python -m mm_method.fs_caption_only --scan-root /home/syh/.cache/huggingface/videomme/data
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

from .frame_search import (
    select_frames_by_framesearch,
    CaptionOnlyCompleted,
    CONFIG,
)


def _gather_videos(
    patterns: List[str] | None,
    from_file: str | None,
    scan_root: str | None,
    exts: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".webm"),
) -> List[str]:
    vids: List[str] = []
    # from patterns/globs
    if patterns:
        for p in patterns:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        if fn.lower().endswith(exts):
                            vids.append(os.path.join(root, fn))
            else:
                vids.extend(glob.glob(p))
    # from file list (one path per line)
    if from_file and os.path.exists(from_file):
        with open(from_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                vids.append(p)
    # scan root dir recursively
    if scan_root and os.path.isdir(scan_root):
        for root, _, files in os.walk(scan_root):
            for fn in files:
                if fn.lower().endswith(exts):
                    vids.append(os.path.join(root, fn))

    # normalize + dedup + keep only files
    out: List[str] = []
    seen = set()
    for v in vids:
        vp = os.path.abspath(v)
        if vp in seen:
            continue
        if os.path.isfile(vp):
            out.append(vp)
            seen.add(vp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Caption-only runner for FrameSearch caches.")
    ap.add_argument("--videos", nargs="*", help="Video paths or globs; directories are recursively scanned.")
    ap.add_argument("--videos-from-file", dest="videos_from_file", help="Text file with one video path per line.")
    ap.add_argument("--scan-root", help="Scan this directory recursively for video files.")

    ap.add_argument("--cache-root", default="mm_method_cache", help="Cache root directory.")
    ap.add_argument("--device", default="cuda", help="Device for caption model (e.g., cuda, cuda:0, cpu).")
    ap.add_argument("--duration", default="long", choices=["short", "medium", "long"], help="Frame candidate density preset.")
    ap.add_argument("--k-top", type=int, default=32, help="Top-K frames (kept for schema compatibility; not used in caption-only).")
    ap.add_argument("--max-candidates", type=int, default=None, help="Override candidate frames (default 256).")

    ap.add_argument("--caption-backend", choices=["hf", "ollama"], default="hf", help="Caption backend.")
    ap.add_argument("--caption-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF caption model id (when backend=hf).")
    # Optional HF performance knobs
    ap.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Torch dtype for HF model (env FS_TORCH_DTYPE).")
    ap.add_argument("--attn", choices=["auto", "flash_attention_2", "eager"], default="auto", help="Attention implementation (env FS_ATTN_IMPL).")
    ap.add_argument("--cuda-alloc-expandable", action="store_true", help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for lower frag.")

    args = ap.parse_args()

    # Build video list
    videos = _gather_videos(args.videos, args.videos_from_file, args.scan_root)
    if not videos:
        print("[ERROR] No videos provided. Use --videos/--scan-root/--videos-from-file.")
        raise SystemExit(2)

    # Configure FrameSearch to caption-only, matching frame_search behavior
    CONFIG["PIPELINE_MODE"] = "caption_only"
    CONFIG["CAPTION_BACKEND"] = args.caption_backend
    # The builder computes backend from CAPTION_BACKEND; USE_OLLAMA is not strictly required but keep consistent.
    CONFIG["USE_OLLAMA"] = bool(args.caption_backend == "ollama")
    CONFIG["USE_OPENAI"] = False

    # Optional low-frag allocator to reduce OOM from fragmentation
    if args.cuda_alloc_expandable:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Pass HF dtype/attention to loaders in frame_search via env
    os.environ["FS_TORCH_DTYPE"] = args.torch_dtype
    os.environ["FS_ATTN_IMPL"] = args.attn

    print(f"[INFO] Caption-only run | backend={args.caption_backend} | model={args.caption_model} | cache_root={args.cache_root}")
    print(f"[INFO] Videos: {len(videos)}")

    # Use a benign placeholder question (unused in caption stage, but kept for schema consistency)
    placeholder_q = "__framesearch_caption_only__"

    ok, fail = 0, 0
    for i, vp in enumerate(videos, 1):
        print(f"[INFO] ({i}/{len(videos)}) caption-only => {vp}")
        try:
            # This will raise CaptionOnlyCompleted at the end of caption stage; treat as success
            select_frames_by_framesearch(
                video_path=vp,
                question=placeholder_q,
                duration_label=args.duration,
                k_top=int(args.k_top),
                cache_root=args.cache_root,
                device=str(args.device),
                caption_model=str(args.caption_model),
                max_candidates=args.max_candidates,
            )
        except CaptionOnlyCompleted:
            print("[OK] Caption cache completed.")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {vp}: {e}")
            fail += 1

    print(f"[INFO] Done. success={ok}, failed={fail}, total={len(videos)}")


if __name__ == "__main__":
    main()
