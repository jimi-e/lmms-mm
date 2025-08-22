#!/usr/bin/env python3
import json
import sys
import re
from pathlib import Path

_TIME_ANCHOR_RE = re.compile(r"\[\s*t\s*=\s*\d+(?:\.\d+)?(?:\s*→\s*\d+(?:\.\d+)?)?\s*s\s*\]", re.IGNORECASE)
_LOW_BRACKET_RE = re.compile(r"\[\s*low\b[^\]]*\]", re.IGNORECASE)

def normalize_low_tag(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return s
    s0 = s.strip()
    anchors = list(_TIME_ANCHOR_RE.finditer(s0))
    time_anchor = anchors[-1].group(0) if anchors else ""
    if time_anchor:
        s0 = _TIME_ANCHOR_RE.sub("", s0).strip()
    low_found = bool(_LOW_BRACKET_RE.search(s0)) or s0.startswith("[LOW]")
    if low_found:
        s0 = _LOW_BRACKET_RE.sub("", s0).strip()
        if s0.startswith("[LOW]"):
            s0 = s0[5:].lstrip()
    out = ("[LOW] " + s0).strip() if low_found else s0
    if time_anchor and not out.endswith(time_anchor):
        out = f"{out} {time_anchor}"
    return re.sub(r"\s+", " ", out).strip()


def process_file(p: Path, dry: bool=False) -> bool:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"skip (json load failed): {p} ({e})")
        return False
    frames = data.get("frames", {})
    changed = False
    for k, v in list(frames.items()):
        if isinstance(v, dict) and "reasoning" in v:
            old = v.get("reasoning", "")
            new = normalize_low_tag(old)
            if new != old:
                v["reasoning"] = new
                changed = True
    if changed and not dry:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main():
    if len(sys.argv) < 2:
        print("usage: normalize_low_tags.py <output_dir> [--dry-run]")
        sys.exit(1)
    root = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv[2:]
    files = list(root.rglob("*.json"))
    total, changed = 0, 0
    for f in files:
        if f.name.endswith(".json"):
            total += 1
            if process_file(f, dry=dry):
                changed += 1
    print(f"done. files={total}, changed={changed}, dry={dry}")


if __name__ == "__main__":
    main()
