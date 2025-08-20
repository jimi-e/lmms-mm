import os
import json
import math
import hashlib
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
from PIL import Image
import base64
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None


# ---------- helpers ----------

# Global config (edit here to switch backends without environment variables)
CONFIG: Dict[str, object] = {
    'PIPELINE_MODE': 'caption_only',     # full | caption_only | from_caption
    'USE_OLLAMA': False,          # True -> caption/reason 默认用 Ollama
    'USE_OPENAI': False,          # True -> score 默认用 OpenAI
    'CAPTION_BACKEND': 'hf',      # 'hf' | 'ollama' | None (None=跟随 USE_OLLAMA)
    'REASON_BACKEND': 'ollama',       # 'hf' | 'ollama' | None
    'SCORE_BACKEND': 'ollama',        # 'hf' | 'ollama' | 'openai' | None
    # Ollama settings
    'OLLAMA_HOST': 'http://127.0.0.1:11434',
    'CAPTION_OLLAMA_MODEL': 'qwen2.5vl:7b',
    'REASON_OLLAMA_MODEL': 'qwen3:14b',
    'SCORE_OLLAMA_MODEL': 'qwen2.5vl:7b',
    # OpenAI settings
    'OPENAI_BASE_URL': 'https://api.openai.com/v1',
    'OPENAI_MODEL': 'gpt-4o',
    'OPENAI_API_KEY': '',
}

def _hash(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _check_cache_completeness(cache_root: str, vid_stem: str, question: str, k_top: int) -> Dict[str, bool]:
    """检查缓存文件的完整性
    
    Returns:
        Dict with keys: 'caption', 'reasoning', 'scoring', 'selected'
        Values indicate whether each stage is complete
    """
    print(f"[DEBUG] Checking cache completeness for {vid_stem}")
    
    qh = _hash(question)
    vdir = os.path.join(cache_root, vid_stem)
    
    # 检查各个阶段的文件
    cap_json_path = os.path.join(vdir, f"{vid_stem}_caption.json")
    rsn_json_path = os.path.join(vdir, f"{vid_stem}_reasoning_{qh}.json")
    scr_json_path = os.path.join(vdir, f"{vid_stem}_scoring_{qh}.json")
    sel_json_path = os.path.join(vdir, f"{vid_stem}_selected_{qh}.json")
    
    result = {
        'caption': False,
        'reasoning': False,
        'scoring': False,
        'selected': False
    }
    
    # 检查 caption 文件
    if os.path.exists(cap_json_path):
        try:
            with open(cap_json_path, 'r', encoding='utf-8') as f:
                cap_data = json.load(f)
            captions = cap_data.get('captions', {})
            if captions and len(captions) > 0:
                result['caption'] = True
                print(f"[DEBUG] Caption cache found: {len(captions)} captions")
        except Exception as e:
            print(f"[DEBUG] Caption cache error: {e}")
    
    # 检查 reasoning 文件
    if os.path.exists(rsn_json_path):
        try:
            with open(rsn_json_path, 'r', encoding='utf-8') as f:
                rsn_data = json.load(f)
            items = rsn_data.get('items', [])
            if items and len(items) > 0:
                # 检查 reasoning 内容是否非空
                valid_items = [item for item in items if item.get('reasoning', '').strip()]
                if len(valid_items) == len(items):
                    result['reasoning'] = True
                    print(f"[DEBUG] Reasoning cache found: {len(items)} items")
                else:
                    print(f"[DEBUG] Reasoning cache incomplete: {len(valid_items)}/{len(items)} valid")
        except Exception as e:
            print(f"[DEBUG] Reasoning cache error: {e}")
    
    # 检查 scoring 文件
    if os.path.exists(scr_json_path):
        try:
            with open(scr_json_path, 'r', encoding='utf-8') as f:
                scr_data = json.load(f)
            items = scr_data.get('items', [])
            if items and len(items) > 0:
                result['scoring'] = True
                print(f"[DEBUG] Scoring cache found: {len(items)} items")
        except Exception as e:
            print(f"[DEBUG] Scoring cache error: {e}")
    
    # 检查 selected 文件
    if os.path.exists(sel_json_path):
        try:
            with open(sel_json_path, 'r', encoding='utf-8') as f:
                sel_data = json.load(f)
            indices = sel_data.get('indices', [])
            k = sel_data.get('k', 0)
            if indices and len(indices) > 0 and k == k_top:
                result['selected'] = True
                print(f"[DEBUG] Selection cache found: {len(indices)} selected frames (k={k})")
            else:
                print(f"[DEBUG] Selection cache mismatch: expected k={k_top}, got k={k}")
        except Exception as e:
            print(f"[DEBUG] Selection cache error: {e}")
    
    print(f"[DEBUG] Cache completeness: {result}")
    return result


def _load_selected_from_cache(cache_root: str, vid_stem: str, question: str) -> Optional[Tuple[np.ndarray, str, float]]:
    """从缓存中加载已选择的帧
    
    Returns:
        (video_frames, frame_time, video_time) if successful, None otherwise
    """
    print("[DEBUG] Loading selected frames from cache...")
    
    qh = _hash(question)
    vdir = os.path.join(cache_root, vid_stem)
    sel_json_path = os.path.join(vdir, f"{vid_stem}_selected_{qh}.json")
    
    if not os.path.exists(sel_json_path):
        print("[DEBUG] Selection cache file not found")
        return None
    
    try:
        # 加载选择信息
        with open(sel_json_path, 'r', encoding='utf-8') as f:
            sel_data = json.load(f)
        
        indices = sel_data.get('indices', [])
        frame_time = sel_data.get('frame_time', '')
        video_time = sel_data.get('video_time', 0.0)
        
        if not indices:
            print("[DEBUG] No indices in selection cache")
            return None
        
        # 需要重新读取视频文件来获取帧数据
        # 从 caption cache 中获取视频路径
        cap_json_path = os.path.join(vdir, f"{vid_stem}_caption.json")
        if not os.path.exists(cap_json_path):
            print("[DEBUG] Caption cache not found, cannot get video path")
            return None
        
        with open(cap_json_path, 'r', encoding='utf-8') as f:
            cap_data = json.load(f)
        
        video_path = cap_data.get('video_path')
        if not video_path or not os.path.exists(video_path):
            print(f"[DEBUG] Video path not found: {video_path}")
            return None
        
        # 重新读取视频帧
        if VideoReader is None:
            print("[DEBUG] VideoReader not available")
            return None
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        out_frames = vr.get_batch(indices).asnumpy()
        
        print(f"[DEBUG] Successfully loaded {out_frames.shape} frames from cache")
        return out_frames, frame_time, float(video_time)
        
    except Exception as e:
        print(f"[DEBUG] Error loading from selection cache: {e}")
        return None


def _duration_to_candidates(duration: str) -> int:
    d = (duration or '').lower()
    if d == 'short':
        return 256
    if d == 'medium':
        return 256
    if d == 'long':
        return 256
    # fallback if unknown
    return 256


def _uniform_indices(total: int, num: int) -> List[int]:
    if total <= 0 or num <= 0:
        return []
    num = min(total, num)
    return np.linspace(0, total - 1, num, dtype=int).tolist()


def _read_frames(vr, idxs: List[int]) -> np.ndarray:
    if not idxs:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return vr.get_batch(idxs).asnumpy()


def _to_pils(arr: np.ndarray) -> List[Image.Image]:
    return [Image.fromarray(x).convert('RGB') for x in arr]


def _img_to_base64(img: Image.Image, fmt: str = 'PNG') -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _http_post_json(url: str, data: Dict, headers: Optional[Dict] = None, timeout: float = 60.0) -> Dict:
    payload = json.dumps(data).encode('utf-8')
    hdrs = {'Content-Type': 'application/json'}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url=url, data=payload, headers=hdrs, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        try:
            return json.loads(body.decode('utf-8'))
        except Exception:
            return {}


# ---------- caption JSON helpers (per-video, incremental) ----------

def _cap_json_path(cache_root: str, vid_stem: str) -> str:
    """Caption cache path: <cache_root>/<vid_stem>/<vid_stem>_caption.json"""
    vdir = os.path.join(cache_root, vid_stem)
    _ensure_dir(vdir)
    return os.path.join(vdir, f"{vid_stem}_caption.json")


def _write_json_atomic(path: str, data: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _now_iso() -> str:
    try:
        return datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    except Exception:
        return datetime.utcnow().isoformat() + 'Z'


def _load_caps_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _init_caps_json(video_path: str, avg_fps: float, total: int) -> Dict:
    created = _now_iso()
    return {
        'video_path': video_path,
        'avg_fps': float(avg_fps),
        'total_frames': int(total),
        'captions': {},  # frame_idx(str) -> { ts: float, caption: str }
        'meta': {
            'created_at': created,
            'last_updated_at': created,
            'total_time_sec': 0.0,
        },
    }


def _migrate_old_caption_cache(vdir: str, cap_json_path: str, video_path: str, avg_fps: float, total: int) -> Optional[Dict]:
    """If legacy captions_*.json exists under vdir, convert to new single-file cache.
    Return loaded dict if migrated, else None.
    """
    try:
        for name in os.listdir(vdir):
            if not name.startswith('captions_') or not name.endswith('.json'):
                continue
            legacy_path = os.path.join(vdir, name)
            with open(legacy_path, 'r', encoding='utf-8') as f:
                legacy = json.load(f)
            caps_map: Dict[str, Dict] = {}
            for item in legacy:
                idx = int(item.get('frame_idx', -1))
                if idx < 0:
                    continue
                ts = item.get('ts')
                if ts is None and avg_fps > 0:
                    ts = round(idx / avg_fps, 2)
                caps_map[str(idx)] = {
                    'ts': float(ts if ts is not None else 0.0),
                    'caption': str(item.get('caption', ''))
                }
            data = _init_caps_json(video_path, avg_fps, total)
            data['captions'] = caps_map
            _write_json_atomic(cap_json_path, data)
            return data
    except Exception:
        return None
    return None


# ---------- captioner / llm / vlm (lightweight wrappers) ----------

# Custom control-flow exceptions for pipeline modes
class CaptionOnlyCompleted(Exception):
    """Raised after caption cache is written when PIPELINE_MODE='caption_only'."""
    pass

class CaptionsMissing(Exception):
    """Raised when PIPELINE_MODE='from_caption' but caption cache is missing."""
    pass

class _QwenVLCaptioner:
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct', device: str = 'cuda') -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        # Optional dtype/attention controls via env
        torch_dtype = None
        attn_impl = os.getenv('FS_ATTN_IMPL', 'auto').lower()
        try:
            import torch  # type: ignore
            dtype_env = os.getenv('FS_TORCH_DTYPE', 'bf16').lower()
            if dtype_env == 'fp16':
                torch_dtype = torch.float16
            elif dtype_env == 'fp32':
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.bfloat16
        except Exception:
            torch_dtype = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            kwargs = dict(trust_remote_code=True, use_safetensors=True, low_cpu_mem_usage=True)
            if torch_dtype is not None:
                kwargs['torch_dtype'] = torch_dtype  # type: ignore
            if attn_impl in ('flash_attention_2', 'eager', 'sdpa'):
                kwargs['attn_implementation'] = attn_impl  # type: ignore
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **kwargs,
            ).to(device)
        except Exception:
            # Fallback to a more widely supported model
            fb = os.getenv('FS_CAPTION_MODEL_FALLBACK', 'Qwen/Qwen-VL-Chat')
            self.processor = AutoProcessor.from_pretrained(fb, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                fb,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device)
        self.device = device

    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 64) -> List[str]:
        out = []
        for img in images:
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img},
                        {'type': 'text', 'text': '请用一句话描述画面，严格输出JSON：{"caption":"..."}。只输出JSON。'},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], return_tensors='pt').to(self.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            resp = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            # try extract JSON
            try:
                j = json.loads(resp.strip())
                out.append(str(j.get('caption', '')).strip())
            except Exception:
                out.append(resp.strip())
        return out


class _TextReasoner:
    """Use Qwen2.5-VL as text-only chat LLM to produce reasoning lines in JSONL style."""
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct', device: str = 'cuda') -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        torch_dtype = None
        attn_impl = os.getenv('FS_ATTN_IMPL', 'auto').lower()
        try:
            import torch  # type: ignore
            dtype_env = os.getenv('FS_TORCH_DTYPE', 'bf16').lower()
            if dtype_env == 'fp16':
                torch_dtype = torch.float16
            elif dtype_env == 'fp32':
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.bfloat16
        except Exception:
            torch_dtype = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            kwargs = dict(trust_remote_code=True, use_safetensors=True, low_cpu_mem_usage=True)
            if torch_dtype is not None:
                kwargs['torch_dtype'] = torch_dtype  # type: ignore
            if attn_impl in ('flash_attention_2', 'eager', 'sdpa'):
                kwargs['attn_implementation'] = attn_impl  # type: ignore
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **kwargs,
            ).to(device)
        except Exception:
            fb = os.getenv('FS_REASON_MODEL_FALLBACK', os.getenv('FS_CAPTION_MODEL_FALLBACK', 'Qwen/Qwen-VL-Chat'))
            self.processor = AutoProcessor.from_pretrained(fb, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                fb,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device)
        self.device = device

    def reason_chunk(self, items: List[Dict], question: str, max_new_tokens: int = 1024) -> Dict[int, str]:
        # items: [{id:int, caption:str}]
        # Build a strict instruction
        header = '你将看到若干行JSON，每行包含一个id与该帧的caption。针对每行，与给定问题一起，生成对应的reasoning，并严格逐行输出JSON：{"id":id,"reasoning":"..."}。不要输出多余文字。\n'
        lines = '\n'.join([json.dumps({'id': it['id'], 'caption': it['caption'], 'question': question}, ensure_ascii=False) for it in items])
        prompt_text = header + lines
        messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=None, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        resp = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # parse JSON per line
        results: Dict[int, str] = {}
        for line in resp.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if isinstance(j, dict) and 'id' in j:
                    results[int(j['id'])] = str(j.get('reasoning', '')).strip()
            except Exception:
                continue
        return results


class _VLMScorer:
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct', device: str = 'cuda') -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        torch_dtype = None
        attn_impl = os.getenv('FS_ATTN_IMPL', 'auto').lower()
        try:
            import torch  # type: ignore
            dtype_env = os.getenv('FS_TORCH_DTYPE', 'bf16').lower()
            if dtype_env == 'fp16':
                torch_dtype = torch.float16
            elif dtype_env == 'fp32':
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.bfloat16
        except Exception:
            torch_dtype = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            kwargs = dict(trust_remote_code=True, use_safetensors=True, low_cpu_mem_usage=True)
            if torch_dtype is not None:
                kwargs['torch_dtype'] = torch_dtype  # type: ignore
            if attn_impl in ('flash_attention_2', 'eager', 'sdpa'):
                kwargs['attn_implementation'] = attn_impl  # type: ignore
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **kwargs,
            ).to(device)
        except Exception:
            fb = os.getenv('FS_SCORE_MODEL_FALLBACK', os.getenv('FS_CAPTION_MODEL_FALLBACK', 'Qwen/Qwen-VL-Chat'))
            self.processor = AutoProcessor.from_pretrained(fb, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                fb,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device)
        self.device = device

    def score(self, image: Image.Image, question: str, reasoning: str, max_new_tokens: int = 64) -> float:
        instr = '请基于问题与推理判断该截图是否有助于回答问题，输出0到1之间的小数，严格JSON：{"score":0.xx}。只输出JSON。'
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': f'问题：{question}\n推理：{reasoning}\n{instr}'}
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt').to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        resp = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            j = json.loads(resp.strip())
            score = float(j.get('score', 0.0))
            if math.isnan(score) or math.isinf(score):
                return 0.0
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0


# ---------- Ollama / OpenAI backends ----------

class _OllamaCaptioner:
    """Caption images via Ollama vision model.

    Env:
      - FS_OLLAMA_HOST (default: http://127.0.0.1:11434)
      - FS_CAPTION_OLLAMA_MODEL (default: qwen2.5vl:7b)
    """
    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None) -> None:
        self.host = (host or CONFIG.get('OLLAMA_HOST') or 'http://127.0.0.1:11434')  # type: ignore
        self.model = model_name or (CONFIG.get('CAPTION_OLLAMA_MODEL') or 'qwen2.5vl:7b')  # type: ignore

    def caption_batch(self, images: List[Image.Image], max_new_tokens: int = 1000) -> List[str]:
        out: List[str] = []
        url = urllib.parse.urljoin(self.host, '/api/generate')
        prompt = 'You are an image captioning assistant for a Video Question Answering system. Your task is to produce a detailed description for the image between 30 and 100 tokens. Your description cannot exceed 100 tokens. You must capture all important visual information without omission. Describe concrete elements such as objects, people, actions, background, colors, symbols, logos, or visible text with their positions. If some categories are absent (e.g., no text, no people), omit them without inventing details. Do not mention layout, alignment, design style, or atmosphere. The description must be factual, comprehensive, and cover every essential element that is visibly present. Output only the caption text.'
        for img in images:
            b64 = _img_to_base64(img)
            data = {
            'model': self.model,
            'prompt': prompt,
            'images': [b64],
            'stream': False,
                # token control if supported by model
                'options': {'num_predict': max_new_tokens}
            }
            try:
                resp = _http_post_json(url, data, timeout=180.0)
                text = str(resp.get('response', '')).strip()
                try:
                    j = json.loads(text)
                    out.append(str(j.get('caption', '')).strip())
                except Exception:
                    out.append(text)
            except Exception:
                out.append('')
        return out


class _OllamaReasoner:
    """Reason over caption lines with a text model on Ollama.

    Env:
      - FS_OLLAMA_HOST (default: http://127.0.0.1:11434)
      - FS_REASON_OLLAMA_MODEL (default: qwen3:14b)
    """
    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None) -> None:
        self.host = (host or CONFIG.get('OLLAMA_HOST') or 'http://127.0.0.1:11434')  # type: ignore
        self.model = model_name or (CONFIG.get('REASON_OLLAMA_MODEL') or 'qwen3:14b')  # type: ignore

    def reason_chunk(self, items: List[Dict], question: str, max_new_tokens: int = 1024) -> Dict[int, str]:
        """Ask Ollama to return strict JSON (single object with items array) and parse it."""
        url = urllib.parse.urljoin(self.host, '/api/generate')
        # 组织为更稳定的 JSON 目标：要求只输出 JSON 对象：{"items":[{"id":..,"reasoning":"..."}, ...]}
        guide = (
            "你会收到若干条输入，每条包含 id、caption 与相同的 question。"\
            "请逐条生成简短的 reasoning，并严格只输出一个 JSON 对象："\
            "{\"items\":[{\"id\":<与输入相同的id>,\"reasoning\":\"...\"}, ...]}。"\
            "不要输出任何其他文本，不要输出数组之外的内容。"
        )
        payload = {
            'question': question,
            'entries': [
                {'id': int(it['id']), 'caption': str(it['caption'])}
                for it in items
            ]
        }
        prompt = guide + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
        data = {
            'model': self.model,
            'prompt': prompt,
            'format': 'json',  # 强制返回合法 JSON
            'stream': False,
            'options': {'num_predict': max_new_tokens, 'temperature': 0, 'top_p': 0}
        }
        results: Dict[int, str] = {}
        try:
            resp = _http_post_json(url, data, timeout=300.0)
            text = str(resp.get('response', '')).strip()
            if text:
                try:
                    j = json.loads(text)
                except Exception:
                    j = resp.get('response') if isinstance(resp.get('response'), dict) else {}
                if isinstance(j, dict):
                    arr = j.get('items')
                    if isinstance(arr, list):
                        for obj in arr:
                            if isinstance(obj, dict) and 'id' in obj:
                                try:
                                    idx = int(obj['id'])
                                except Exception:
                                    continue
                                results[idx] = str(obj.get('reasoning', '')).strip()
        except Exception:
            pass
        return results

    def reason_single(self, item: Dict, question: str, max_new_tokens: int = 256) -> Tuple[str, str]:
        """Reason one caption via Ollama, return (reasoning, raw_response)."""
        url = urllib.parse.urljoin(self.host, '/api/generate')
        
        # Simplified prompt that's more likely to work with different models
        caption = str(item.get('caption', ''))
        prompt = f"""基于以下图像描述和问题，提供简短的推理分析：

图像描述：{caption}

问题：{question}

请分析这个图像描述是否能帮助回答问题，并说明原因。只输出推理内容，不要包含其他格式。"""
        
        def clean_response(text: str) -> str:
            """Clean <think> tags and other unwanted formatting from response."""
            if not text:
                return text
            
            # Remove <think>...</think> tags and their content
            import re
            # Match <think> followed by any content (including newlines) followed by </think>
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            # Clean up excessive whitespace and newlines
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Multiple newlines to single
            cleaned = cleaned.strip()
            
            return cleaned
        
        # small retry loop to mitigate occasional empty/invalid responses
        for attempt in range(3):
            data = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': max_new_tokens,
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_ctx': 4096,
                }
            }
            try:
                resp = _http_post_json(url, data, timeout=180.0)
                raw = resp.get('response', '') if isinstance(resp, dict) else ''
                raw_str = str(raw).strip()
                
                # Clean the response before using it
                cleaned_response = clean_response(raw_str)
                
                # If we get a non-empty cleaned response, use it
                if cleaned_response and len(cleaned_response) > 5:  # Ensure it's not just punctuation
                    return cleaned_response, raw_str  # Return cleaned for use, raw for logging
                    
            except Exception:
                raw_str = ''
            
            # backoff only if not last attempt
            if attempt < 2:
                try:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                except Exception:
                    pass
        
        # Fallback reasoning if all attempts fail
        fallback = f"基于给定的图像描述，无法确定是否能回答关于'{question}'的问题。"
        return fallback, fallback


class _OllamaImageScorer:
    """Score image relevance via Ollama vision model.

    Env:
      - FS_OLLAMA_HOST (default: http://127.0.0.1:11434)
      - FS_SCORE_OLLAMA_MODEL (default: qwen2.5vl:7b)
    """
    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None) -> None:
        self.host = (host or CONFIG.get('OLLAMA_HOST') or 'http://127.0.0.1:11434')  # type: ignore
        self.model = model_name or (CONFIG.get('SCORE_OLLAMA_MODEL') or 'qwen2.5vl:7b')  # type: ignore

    def score(self, image: Image.Image, question: str, reasoning: str, max_new_tokens: int = 64) -> float:
        url = urllib.parse.urljoin(self.host, '/api/generate')
        b64 = _img_to_base64(image)
        prompt = (
            '请基于问题与推理判断该截图是否有助于回答问题，输出0到1之间的小数，严格JSON：{"score":0.xx}。只输出JSON。\n'
            f'问题：{question}\n推理：{reasoning}'
        )
        data = {
            'model': self.model,
            'prompt': prompt,
            'images': [b64],
            'stream': False,
            'options': {'num_predict': max_new_tokens}
        }
        try:
            resp = _http_post_json(url, data, timeout=180.0)
            text = str(resp.get('response', '')).strip()
            try:
                j = json.loads(text)
                score = float(j.get('score', 0.0))
                if math.isnan(score) or math.isinf(score):
                    return 0.0
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0
        except Exception:
            return 0.0


class _OpenAIImageScorer:
    """Score image relevance via OpenAI API (e.g., gpt-4o).

    Env:
      - OPENAI_API_KEY (required)
      - FS_OPENAI_BASE_URL (optional, default https://api.openai.com/v1)
      - FS_SCORE_OPENAI_MODEL (default gpt-4o)
    """
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model = model_name or (CONFIG.get('OPENAI_MODEL') or 'gpt-4o')  # type: ignore
        base = (base_url or CONFIG.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1')  # type: ignore
        self.base_url = str(base).rstrip('/')
        self.api_key = str(CONFIG.get('OPENAI_API_KEY') or '')

    def score(self, image: Image.Image, question: str, reasoning: str, max_new_tokens: int = 64) -> float:
        if not self.api_key:
            return 0.0
        url = self.base_url + '/chat/completions'
        b64 = _img_to_base64(image)
        system_msg = {
            'role': 'system',
            'content': 'You score how helpful an image is for answering a question. Output only JSON with a float "score" between 0 and 1.'
        }
        # Some APIs expect different content schema; try OpenAI's vision format
        user_msg = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'Question: {question}\nReasoning: {reasoning}\nRespond strictly with JSON: {{"score":0.xx}}.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
            ]
        }
        data = {
            'model': self.model,
            'messages': [system_msg, user_msg],
            'max_tokens': max_new_tokens,
            'temperature': 0.0,
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            resp = _http_post_json(url, data, headers=headers, timeout=180.0)
            text = ''
            if isinstance(resp, dict):
                choices = resp.get('choices') or []
                if choices:
                    message = choices[0].get('message') or {}
                    text = str(message.get('content', '')).strip()
            if not text:
                return 0.0
            try:
                j = json.loads(text)
                score = float(j.get('score', 0.0))
                if math.isnan(score) or math.isinf(score):
                    return 0.0
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0
        except Exception:
            return 0.0


def _build_captioner(backend: str, device: str, hf_model: str) -> object:
    if backend == 'ollama':
        return _OllamaCaptioner(model_name=str(CONFIG.get('CAPTION_OLLAMA_MODEL') or 'qwen2.5vl:7b'))
    # default HF
    return _QwenVLCaptioner(model_name=hf_model, device=device)


def _build_reasoner(backend: str, device: str, hf_model: str) -> object:
    if backend == 'ollama':
        return _OllamaReasoner(model_name=str(CONFIG.get('REASON_OLLAMA_MODEL') or 'qwen3:14b'))
    return _TextReasoner(model_name=hf_model, device=device)


def _build_scorer(backend: str, device: str, hf_model: str) -> object:
    if backend == 'openai':
        return _OpenAIImageScorer(model_name=str(CONFIG.get('OPENAI_MODEL') or 'gpt-4o'))
    elif backend == 'ollama':
        return _OllamaImageScorer(model_name=str(CONFIG.get('SCORE_OLLAMA_MODEL') or 'qwen2.5vl:7b'))
    else:
        return _VLMScorer(model_name=hf_model, device=device)


# ---------- main API ----------

def select_frames_by_framesearch(
    video_path: str,
    question: str,
    *,
    duration_label: str = 'medium',
    k_top: int = 32,
    cache_root: str = 'mm_method_cache',
    device: str = 'cuda',
    chunk_size: int = 128,
    caption_model: str = 'Qwen/Qwen2-VL-7B-Instruct',
    reason_model: str = 'Qwen/Qwen2-VL-7B-Instruct',
    score_model: str = 'Qwen/Qwen2-VL-7B-Instruct',
    max_candidates: int | None = None,
) -> Tuple[np.ndarray, str, float]:
    """Return (video_np[K,H,W,C], frame_time, video_time) using FrameSearch pipeline.

    Caches intermediate artifacts (captions/reasoning/scores) as JSON under cache_root/<video_id>/.
    """
    print("[DEBUG] ===== FrameSearch Pipeline Start =====")
    print(f"[DEBUG] video_path: {video_path}")
    print(f"[DEBUG] question: {question[:100]}...")
    print(f"[DEBUG] k_top: {k_top}")
    print(f"[DEBUG] cache_root: {cache_root}")
    
    if VideoReader is None:
        raise RuntimeError('decord is required for frame sampling but is not available.')

    vid_stem = os.path.splitext(os.path.basename(video_path))[0]
    vdir = os.path.join(cache_root, vid_stem)
    _ensure_dir(vdir)
    print(f"[DEBUG] vid_stem: {vid_stem}")

    # 首先检查缓存完整性
    cache_status = _check_cache_completeness(cache_root, vid_stem, question, k_top)
    
    # 如果所有缓存都完整，直接从缓存加载
    if all(cache_status.values()):
        print("[DEBUG] All caches complete, loading from cache...")
        cached_result = _load_selected_from_cache(cache_root, vid_stem, question)
        if cached_result is not None:
            print("[DEBUG] Successfully loaded complete result from cache")
            return cached_result
        else:
            print("[DEBUG] Failed to load from cache, proceeding with pipeline")

    # open video & sample candidates by duration
    print("[DEBUG] Opening video file...")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    avg_fps = float(vr.get_avg_fps()) if hasattr(vr, 'get_avg_fps') else 0.0
    print(f"[DEBUG] Video stats: {total} frames, {avg_fps:.2f} fps")
    
    M = _duration_to_candidates(duration_label)
    if max_candidates is not None:
        try:
            M = max(1, min(int(max_candidates), M))
        except Exception:
            pass
    cand_idxs = _uniform_indices(total, M)
    print(f"[DEBUG] Candidate frames: {len(cand_idxs)} frames")
    
    if not cand_idxs:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8), '', 0.0

    # Decide backends via CONFIG (global toggles in this file)
    use_ollama = bool(CONFIG.get('USE_OLLAMA'))
    use_openai = bool(CONFIG.get('USE_OPENAI'))
    caption_backend = str(CONFIG.get('CAPTION_BACKEND') or ('ollama' if use_ollama else 'hf')).lower()
    reason_backend = str(CONFIG.get('REASON_BACKEND') or ('ollama' if use_ollama else 'hf')).lower()
    # score preference: openai > ollama > hf
    score_backend = str(CONFIG.get('SCORE_BACKEND') or ('openai' if use_openai else ('ollama' if use_ollama else 'hf'))).lower()
    print(f"[DEBUG] Backends - caption: {caption_backend}, reason: {reason_backend}, score: {score_backend}")

    # Stage 1: captions (cache, per-video JSON with incremental writes)
    print("[DEBUG] ===== Stage 1: Captions =====")
    if cache_status['caption']:
        print("[DEBUG] Caption cache complete, skipping caption generation")
    else:
        print("[DEBUG] Caption cache incomplete, will generate captions")
    
    mode = str(CONFIG.get('PIPELINE_MODE', 'full')).lower()
    print(f"[DEBUG] Pipeline mode: {mode}")
    
    cap_json_path = _cap_json_path(cache_root, vid_stem)
    caps_json = _load_caps_json(cap_json_path)
    if caps_json is None:
        # Try migrate from legacy per-run captions_*.json
        caps_json = _migrate_old_caption_cache(vdir, cap_json_path, video_path, avg_fps, total)
    # Also migrate from older single-file root cache: <cache_root>/<vid>.json
    if caps_json is None:
        old_single = os.path.join(cache_root, f"{vid_stem}.json")
        if os.path.exists(old_single):
            try:
                with open(old_single, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                # Expect same structure as new caption JSON
                if isinstance(old, dict) and 'captions' in old:
                    caps_json = old
                    _write_json_atomic(cap_json_path, caps_json)
            except Exception:
                caps_json = None
    if caps_json is None:
        caps_json = _init_caps_json(video_path, avg_fps, total)
        _write_json_atomic(cap_json_path, caps_json)

    # ensure metadata fresh
    changed_meta = False
    if abs(float(caps_json.get('avg_fps', 0.0)) - avg_fps) > 1e-6:
        caps_json['avg_fps'] = float(avg_fps)
        changed_meta = True
    if int(caps_json.get('total_frames', 0)) != int(total):
        caps_json['total_frames'] = int(total)
        changed_meta = True
    if changed_meta:
        _write_json_atomic(cap_json_path, caps_json)

    # Determine which indices still need captions
    cap_map: Dict[str, Dict] = caps_json.get('captions', {}) or {}
    needed = []
    for idx in cand_idxs:
        key = str(idx)
        entry = cap_map.get(key)
        if not entry or not str(entry.get('caption', '')).strip():
            needed.append(idx)

    print(f"[DEBUG] Caption status: {len(cap_map)} existing, {len(needed)} needed")

    if mode == 'from_caption' and needed:
        # Strict: do not compute in from_caption mode
        print("[DEBUG] from_caption mode but missing captions, raising exception")
        raise CaptionsMissing(cap_json_path)

    cap_stage_start_ts = time.time()
    if needed and not cache_status['caption']:
        print(f"[DEBUG] Generating {len(needed)} captions...")
        capper = _build_captioner(caption_backend, device=device, hf_model=caption_model)
        for i, idx in enumerate(needed):
            print(f"[DEBUG] Captioning frame {idx} ({i+1}/{len(needed)})")
            frame_arr = _read_frames(vr, [idx])
            pil = _to_pils(frame_arr)
            cap_text = ''
            try:
                cap_text = capper.caption_batch(pil)[0] if pil else ''
                print(f"[DEBUG] Caption result: {cap_text[:100]}...")
            except Exception as e:
                print(f"[DEBUG] Caption error: {e}")
                cap_text = ''
            cap_map[str(idx)] = {
                'ts': round((idx / avg_fps) if avg_fps > 0 else 0.0, 2),
                'caption': cap_text
            }
            caps_json['captions'] = cap_map
            # incremental safe write
            _write_json_atomic(cap_json_path, caps_json)
        # accumulate caption stage time
        try:
            elapsed = max(0.0, time.time() - cap_stage_start_ts)
            caps_json.setdefault('meta', {})
            prev = float(caps_json['meta'].get('total_time_sec', 0.0))
            caps_json['meta']['total_time_sec'] = round(prev + elapsed, 3)
            caps_json['meta']['last_updated_at'] = _now_iso()
            _write_json_atomic(cap_json_path, caps_json)
        except Exception:
            pass
    elif cache_status['caption']:
        print("[DEBUG] Using existing captions from cache")

    # Build caps list for current candidate set in order
    caps = [
        {
            'frame_idx': int(idx),
            'ts': float(cap_map.get(str(idx), {}).get('ts', round((idx / avg_fps) if avg_fps > 0 else 0.0, 2))),
            'caption': str(cap_map.get(str(idx), {}).get('caption', ''))
        }
        for idx in cand_idxs
    ]
    print(f"[DEBUG] Built {len(caps)} caption entries")

    if mode == 'caption_only':
        # Stop here for this video; caller can skip heavy steps
        print("[DEBUG] caption_only mode, stopping here")
        raise CaptionOnlyCompleted(cap_json_path)

    # Stage 2: reasoning (per-question file, incremental per-frame writes + raw log)
    qh = _hash(question)
    # Prefer per-question file: <vid>_reasoning_<qh>.json
    rsn_q_path = os.path.join(vdir, f'{vid_stem}_reasoning_{qh}.json')
    rsn_list: List[Dict]
    def _load_rsn_list_from(path: str) -> Optional[List[Dict]]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data  # legacy list format
            if isinstance(data, dict):
                # allow wrappers like {"data": [...]}
                for key in ('data', 'list', 'items'):
                    v = data.get(key)
                    if isinstance(v, list):
                        return v
            return None
        except Exception:
            return None

    # Load or initialize reasoning store with meta+items
    rsn_store: Dict[str, object] = {}
    rsn_list = []
    if os.path.exists(rsn_q_path):
        try:
            with open(rsn_q_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get('items'), list):
                rsn_store = existing
                rsn_list = existing['items']  # type: ignore
            else:
                rsn_list = _load_rsn_list_from(rsn_q_path) or []
                rsn_store = {'meta': {}, 'items': rsn_list}
        except Exception:
            rsn_store = {'meta': {}, 'items': []}
            rsn_list = []
    else:
        rsn_store = {
            'meta': {
                'video_path': video_path,
                'vid': vid_stem,
                'question': question,
                'question_hash': qh,
                'avg_fps': avg_fps,
                'total_frames': total,
                'created_at': _now_iso(),
            },
            'items': []
        }

    if not rsn_list:
        # Fallback to legacy aggregated file with by_q
        rsn_agg_path = os.path.join(vdir, f'{vid_stem}_reasoning.json')
        try:
            if os.path.exists(rsn_agg_path):
                with open(rsn_agg_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and 'by_q' in loaded and isinstance(loaded['by_q'], dict):
                    legacy = loaded['by_q'].get(qh)
                    if isinstance(legacy, list):
                        rsn_list = legacy
                        rsn_store['items'] = rsn_list  # type: ignore
                        if 'meta' not in rsn_store:
                            rsn_store['meta'] = {
                                'video_path': video_path,
                                'vid': vid_stem,
                                'question': question,
                                'question_hash': qh,
                                'avg_fps': avg_fps,
                                'total_frames': total,
                                'created_at': _now_iso(),
                            }
                        _write_json_atomic(rsn_q_path, rsn_store)  # type: ignore
        except Exception:
            rsn_list = []

    # Per-frame incremental reasoning
    reasoner = _build_reasoner(reason_backend, device=device, hf_model=reason_model)
    rsn_stage_start_ts = time.time()
    existing_rsn_idx = {int(it.get('frame_idx', -1)) for it in (rsn_list or [])}
    rsn_log_path = os.path.join(vdir, f'{vid_stem}_reasoning_{qh}.log')
    _last_call_ts = 0.0
    for pos, c in enumerate(caps):
        fidx = int(c['frame_idx'])
        if fidx in existing_rsn_idx:
            continue
        # simple throttle to avoid hammering local server in tight loops
        try:
            now = time.time()
            if now - _last_call_ts < 0.05:
                time.sleep(0.05)
        except Exception:
            pass
        if isinstance(reasoner, _OllamaReasoner):
            r_text, raw = reasoner.reason_single({'id': 0, 'caption': c['caption']}, question)
        else:
            out = reasoner.reason_chunk([{'id': 0, 'caption': c['caption']}], question)
            r_text, raw = (out.get(0, ''), '')
        # Log raw response per call
        try:
            with open(rsn_log_path, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({'ts': _now_iso(), 'pos': pos, 'frame_idx': fidx, 'raw': raw}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # Append and write
        entry = {'frame_idx': fidx, 'reasoning': r_text}
        rsn_store.setdefault('items', [])  # type: ignore
        items_list = rsn_store['items']  # type: ignore
        if isinstance(items_list, list):
            items_list.append(entry)
        else:
            rsn_store['items'] = [entry]  # type: ignore
        _write_json_atomic(rsn_q_path, rsn_store)  # type: ignore
        try:
            _last_call_ts = time.time()
        except Exception:
            pass
    # accumulate reasoning stage time
    try:
        elapsed = max(0.0, time.time() - rsn_stage_start_ts)
        if isinstance(rsn_store, dict):
            meta = rsn_store.setdefault('meta', {})  # type: ignore
            prev = float(meta.get('total_time_sec', 0.0) or 0.0)
            meta['total_time_sec'] = round(prev + elapsed, 3)
            meta['last_updated_at'] = _now_iso()
            _write_json_atomic(rsn_q_path, rsn_store)  # type: ignore
    except Exception:
        pass

    # Stage 3: scoring (per-question file, incremental per-frame writes)
    scr_q_path = os.path.join(vdir, f'{vid_stem}_scoring_{qh}.json')
    def _load_scores_from(path: str) -> Optional[List[Dict]]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ('data', 'list', 'items'):
                    v = data.get(key)
                    if isinstance(v, list):
                        return v
            return None
        except Exception:
            return None

    # Load or init scoring store with meta+items
    scr_store: Dict[str, object] = {}
    scores = []
    if os.path.exists(scr_q_path):
        try:
            with open(scr_q_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get('items'), list):
                scr_store = existing
                scores = existing['items']  # type: ignore
            else:
                scores = _load_scores_from(scr_q_path) or []
                scr_store = {'meta': {}, 'items': scores}
        except Exception:
            scr_store = {'meta': {}, 'items': []}
            scores = []
    else:
        scr_store = {
            'meta': {
                'video_path': video_path,
                'vid': vid_stem,
                'question': question,
                'question_hash': qh,
                'avg_fps': avg_fps,
                'total_frames': total,
                'created_at': _now_iso(),
            },
            'items': []
        }

    if not scores:
        # Fallback to aggregated legacy scoring file
        scr_agg_path = os.path.join(vdir, f'{vid_stem}_scoring.json')
        try:
            if os.path.exists(scr_agg_path):
                with open(scr_agg_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and 'by_q' in loaded and isinstance(loaded['by_q'], dict):
                    legacy_scores = loaded['by_q'].get(qh)
                    if isinstance(legacy_scores, list):
                        scores = legacy_scores
                        scr_store['items'] = scores  # type: ignore
                        if 'meta' not in scr_store:
                            scr_store['meta'] = {
                                'video_path': video_path,
                                'vid': vid_stem,
                                'question': question,
                                'question_hash': qh,
                                'avg_fps': avg_fps,
                                'total_frames': total,
                                'created_at': _now_iso(),
                            }
                        _write_json_atomic(scr_q_path, scr_store)  # type: ignore
        except Exception:
            scores = []

    # Map frame_idx -> reasoning (from rsn_store)
    rsn_items = rsn_store.get('items') if isinstance(rsn_store, dict) else []  # type: ignore
    rsn_map = {int(it.get('frame_idx')): str(it.get('reasoning', '') or '') for it in (rsn_items or [])}  # type: ignore
    scorer = _build_scorer(score_backend, device=device, hf_model=score_model)
    scr_stage_start_ts = time.time()
    existing_score_idx = {int(it.get('frame_idx', -1)) for it in (scores or [])}
    for c in caps:
        fidx = int(c['frame_idx'])
        if fidx in existing_score_idx:
            continue
        frame = _read_frames(vr, [fidx])
        pil_list = _to_pils(frame)
        img = pil_list[0] if pil_list else None
        rsn_text = rsn_map.get(fidx, '')
        s_val = 0.0
        if img is not None:
            s_val = scorer.score(img, question, rsn_text)
        entry = {
            'frame_idx': fidx,
            'score': float(round(s_val, 4)),
            'ts': float(c['ts']),
            'reasoning': rsn_text,
        }
        scr_store.setdefault('items', [])  # type: ignore
        s_items = scr_store['items']  # type: ignore
        if isinstance(s_items, list):
            s_items.append(entry)
        else:
            scr_store['items'] = [entry]  # type: ignore
        _write_json_atomic(scr_q_path, scr_store)  # type: ignore
    # accumulate scoring stage time
    try:
        elapsed = max(0.0, time.time() - scr_stage_start_ts)
        if isinstance(scr_store, dict):
            meta = scr_store.setdefault('meta', {})  # type: ignore
            prev = float(meta.get('total_time_sec', 0.0) or 0.0)
            meta['total_time_sec'] = round(prev + elapsed, 3)
            meta['last_updated_at'] = _now_iso()
            _write_json_atomic(scr_q_path, scr_store)  # type: ignore
    except Exception:
        pass

    scores = scr_store.get('items') or []  # type: ignore

    # Select Top-K by score
    sel_stage_start_ts = time.time()
    scores_sorted = sorted(scores, key=lambda x: x.get('score', 0.0), reverse=True)
    k = max(1, int(k_top))
    sel = scores_sorted[: min(k, len(scores_sorted))]
    sel_idxs = [int(s['frame_idx']) for s in sel]

    # Prepare outputs
    out_frames = _read_frames(vr, sel_idxs)
    # frame_time string using true seconds
    if avg_fps > 0:
        times = [i / avg_fps for i in sel_idxs]
    else:
        times = list(range(len(sel_idxs)))
    frame_time = ','.join([f"{t:.2f}s" for t in times])
    video_time = (total / avg_fps) if avg_fps > 0 else 0.0

    # Save selection summary per-question file
    sel_q_path = os.path.join(vdir, f'{vid_stem}_selected_{qh}.json')
    try:
        sel_elapsed = max(0.0, time.time() - sel_stage_start_ts)
    except Exception:
        sel_elapsed = 0.0
    sel_obj = {
        'k': k,
        'indices': sel_idxs,
        'frame_time': frame_time,
        'video_time': round(video_time, 2),
        'meta': {
            'created_at': _now_iso(),
            'total_time_sec': round(sel_elapsed, 3),
        }
    }
    _write_json_atomic(sel_q_path, sel_obj)

    return out_frames, frame_time, float(video_time)
