#!/usr/bin/env python3
"""Independent Bilibili embedded-ad detection experiment.

Embedded-ad detection for bili-monitor. The module can run standalone for
tests, and bili-monitor can invoke detect_local_video() on an already downloaded
file to avoid a second download.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = DATA / "runs"
SHARED_ASR = None


@dataclass
class Segment:
    start: float
    end: float
    brand: str = ""
    confidence: float = 0.0
    source: str = ""
    evidence: str = ""

    @property
    def start_mmss(self) -> str:
        return mmss(self.start)

    @property
    def end_mmss(self) -> str:
        return mmss(self.end)


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_settings():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    monitor_path = Path(os.environ.get("BILI_MONITOR_CONFIG", settings.get("monitor_config", "config.json")))
    if not monitor_path.is_absolute():
        monitor_path = ROOT / monitor_path
    with open(monitor_path, encoding="utf-8") as f:
        monitor = json.load(f)
    blacklist = []
    with open(ROOT / "blacklist.txt", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                blacklist.append(value.lower())
    settings["blacklist"] = blacklist
    settings["deepseek"] = monitor.get("deepseek", {})
    return settings, monitor


def http_get(url: str, monitor: dict, timeout: int = 20) -> dict:
    b = monitor["bilibili"]
    cookie = f"SESSDATA={b['sessdata']}; bili_jct={b['bili_jct']}"
    r = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Cookie": cookie,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def video_info(bv: str, monitor: dict) -> dict:
    data = http_get(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}", monitor)
    if data.get("code") != 0:
        raise RuntimeError(f"B站视频信息失败 {bv}: {data.get('message')}")
    return data["data"]


def prepare_cookies(monitor: dict) -> Path:
    b = monitor["bilibili"]
    path = DATA / "bili_cookies.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    expiry = int(time.time()) + 180 * 24 * 3600
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".bilibili.com\tTRUE\t/\tFALSE\t{expiry}\tSESSDATA\t{b['sessdata']}\n"
        f".bilibili.com\tTRUE\t/\tTRUE\t{expiry}\tbili_jct\t{b['bili_jct']}\n",
        encoding="utf-8",
    )
    return path


def download(bv: str, monitor: dict, settings: dict) -> Path:
    outdir = RUNS / bv
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "video.mp4"
    if output.exists() and output.stat().st_size > 1024:
        return output
    cookies = prepare_cookies(monitor)
    fmt = settings["download"].get("format", "bestvideo*+bestaudio/best")
    cmd = [
        "yt-dlp", f"https://www.bilibili.com/video/{bv}",
        "-f", fmt, "--merge-output-format", "mp4",
        "--cookies", str(cookies), "--no-playlist", "--no-progress",
        "-o", str(output),
    ]
    run_cmd(cmd, timeout=1800)
    if not output.exists():
        raise RuntimeError(f"下载失败: {bv}")
    return output


def run_cmd(cmd: list[str], timeout: int = 600, cwd: Path | None = None):
    print("  $", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n{p.stderr[-1200:]}")
    return p


def media_duration(path: Path) -> float:
    p = run_cmd([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path)
    ], timeout=30)
    return float(p.stdout.strip())


def visual_interval(duration: float, settings: dict) -> float:
    """视觉广告检测的目标抽帧间隔(秒): 全片最多约 max_frames 帧, 且不小于10s、不大于60s。"""
    max_frames = int(settings["visual"]["max_frames"])
    interval = max(10.0, math.ceil(duration / max_frames))
    return min(interval, 60.0)


def extract_frames(video: Path, duration: float, settings: dict) -> tuple[list[Path], float]:
    interval = visual_interval(duration, settings)
    frames_dir = video.parent / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("frame_*.jpg"):
        old.unlink()
    run_cmd([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps=1/{interval:.6f},scale=640:-2", "-q:v", "4",
        str(frames_dir / "frame_%04d.jpg")
    ], timeout=1200)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    return frames, interval


def call_visual_model(images: list[Path], timestamps: list[float], settings: dict, monitor: dict) -> dict:
    prompt = f"""你要判断B站视频连续截帧中是否出现商业广告/赞助口播。截帧按时间顺序排列，时间点分别是：{', '.join(mmss(x) for x in timestamps)}。
必须多帧联合判断：关注从正常内容突然转入UP主推销、品牌 logo、产品展示、优惠码、下载号召、广告字幕等。
不要把单纯讨论品牌或新闻报道误判为广告；要看起来像视频内插入的推广段落。
只输出 JSON，不要 Markdown：{{"ad_like":true/false,"start_seconds":数字,"end_seconds":数字,"brand":"商家或空","confidence":0到1,"evidence":"简短依据"}}。时间必须落在这些截帧时间范围内。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()}
        })
    api_key = monitor["zhipu"]["api_key"]
    models = settings["visual"]["models"]
    for model in models:
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 400},
                    timeout=60,
                )
                if r.status_code == 429:
                    time.sleep((attempt + 1) * 3)
                    continue
                data = r.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                if text:
                    return parse_json_object(text)
            except Exception as exc:
                print(f"  visual {model} attempt {attempt+1}: {exc}", flush=True)
            time.sleep(1)
    return {"ad_like": False, "confidence": 0.0, "evidence": "视觉模型调用失败"}


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    candidates = [text]
    # Extract the first balanced object, ignoring braces inside quoted strings.
    start_pos = text.find("{")
    if start_pos >= 0:
        depth = 0; in_string = False; escaped = False
        for i, ch in enumerate(text[start_pos:], start_pos):
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"' and in_string:
                in_string = False
            elif ch == '"':
                in_string = True
            elif not in_string and ch == "{":
                depth += 1
            elif not in_string and ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.insert(0, text[start_pos:i+1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            # Some VLM outputs omit a closing quote around evidence. Repair once.
            repaired = candidate.rstrip()
            if repaired.endswith("}") and not repaired[:-1].rstrip().endswith('"'):
                repaired = repaired[:-1].rstrip() + '"}'
            try:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    raise ValueError(f"模型未返回有效JSON: {text[:200]}")



def clamp_visual_bounds(ans: dict, ts: list[float], interval: float, duration: float) -> tuple[float, float]:
    """Clamp model-reported bounds to the current frame window.

    Vision models sometimes interpret 06:30 as 6.3 seconds, or return a distant
    timestamp from an adjacent scene. Visual evidence is only valid for the
    frames actually sent to the model, so never let it escape this window.
    """
    window_start = float(ts[0])
    window_end = min(float(duration), float(ts[-1]) + float(interval))

    def number(key: str, fallback: float) -> float:
        try:
            return float(ans.get(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    raw_start = number("start_seconds", window_start)
    raw_end = number("end_seconds", window_end)
    if raw_end <= raw_start or raw_end < window_start or raw_start > window_end:
        # Invalid or non-overlapping model timestamps cannot refine the window.
        return window_start, window_end
    start = max(window_start, min(raw_start, window_end))
    end = max(start, min(window_end, max(raw_end, start)))
    if end <= start:
        return window_start, window_end
    return start, end


def _visual_windows(frames: list[Path], timestamps: list[float], interval: float,
                    duration: float, settings: dict, monitor: dict) -> list[Segment]:
    """对给定帧列表执行滑动窗口视觉广告判定(最多3路并发)。frames/timestamps 由调用方提供。"""
    window = int(settings["visual"]["window_frames"])
    step = int(settings["visual"]["step_frames"])
    indices = list(range(0, max(1, len(frames) - window + 1), step))

    def _process_window(i: int):
        idxs = list(range(i, min(len(frames), i + window)))
        if len(idxs) < 2:
            return None
        batch = [frames[j] for j in idxs]
        ts = [timestamps[j] for j in idxs]
        try:
            ans = call_visual_model(batch, ts, settings, monitor)
        except Exception as exc:
            print(f"  visual parse error: {exc}", flush=True)
            return None
        brand_text = (str(ans.get("brand", "")) + " " + str(ans.get("evidence", ""))).lower()
        visual_blacklist_hits = [b for b in settings["blacklist"] if b in brand_text]
        seg = None
        if visual_blacklist_hits:
            start, end = clamp_visual_bounds(ans, ts, interval, duration)
            seg = Segment(start, end, visual_blacklist_hits[0], 0.99, "visual_blacklist", str(ans.get("evidence", "")))
        elif ans.get("ad_like"):
            start, end = clamp_visual_bounds(ans, ts, interval, duration)
            seg = Segment(start, end, str(ans.get("brand", "")), float(ans.get("confidence", 0)), "visual", str(ans.get("evidence", "")))
        print(f"  visual {mmss(ts[0])}-{mmss(ts[-1])}: ad={ans.get('ad_like')} conf={ans.get('confidence')}", flush=True)
        return seg

    result: list[Segment] = []
    workers = min(3, max(1, len(indices)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_window, i) for i in indices]
        for fut in futures:
            try:
                seg = fut.result()
            except Exception as exc:
                print(f"  visual worker error: {exc}", flush=True)
                continue
            if seg is not None:
                result.append(seg)
    return result


def visual_detection(video: Path, duration: float, settings: dict, monitor: dict) -> list[Segment]:
    frames, interval = extract_frames(video, duration, settings)
    timestamps = [min(duration, i * interval) for i in range(len(frames))]
    return _visual_windows(frames, timestamps, interval, duration, settings, monitor)


def visual_detection_from_frames(frames: list[Path], timestamps: list[float],
                                 duration: float, settings: dict, monitor: dict) -> list[Segment]:
    """复用已抽好的关键帧做广告视觉判定(不再重新ffmpeg切帧)。"""
    interval = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    return _visual_windows(frames, timestamps, interval, duration, settings, monitor)

def extract_asr_chunks(video: Path, settings: dict) -> list[dict]:
    chunk_sec = int(settings["asr"]["chunk_seconds"])
    audio_dir = video.parent / "asr_chunks"
    audio_dir.mkdir(exist_ok=True)
    for old in audio_dir.glob("chunk_*.wav"):
        old.unlink()
    run_cmd([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "segment",
        "-segment_time", str(chunk_sec), "-c:a", "pcm_s16le",
        str(audio_dir / "chunk_%05d.wav")
    ], timeout=1200)
    return [
        {"path": p, "start": i * chunk_sec, "end": (i + 1) * chunk_sec}
        for i, p in enumerate(sorted(audio_dir.glob("chunk_*.wav")))
    ]


class SenseVoiceRunner:
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def ensure(self):
        if self.model is not None:
            return self.model
        import torch
        from funasr import AutoModel
        torch.set_num_threads(int(self.settings["asr"].get("threads", 6)))
        requested = str(self.settings["asr"].get("device", "auto")).lower()
        if requested not in ("auto", "cuda", "cpu"):
            requested = "auto"
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        device = "cuda" if requested in ("auto", "cuda") and cuda_available else "cpu"
        if requested == "cuda" and not cuda_available:
            print("  local ASR: cuda unavailable, fallback to cpu", flush=True)
        self.model = AutoModel(
            model=self.settings["asr"]["model"],
            disable_update=True, device=device, disable_pbar=True,
        )
        print(f"  local ASR device: {device}", flush=True)
        return self.model

    def transcribe(self, wav: Path) -> str:
        try:
            model = self.ensure()
            res = model.generate(input=str(wav), cache={}, language="auto", use_itn=True, batch_size_s=60)
            raw = "".join(x.get("text", "") for x in res if isinstance(x, dict))
            return re.sub(r"<\|[^|]+\|>", "", raw).strip()
        except Exception as exc:
            print(f"  local ASR error: {exc}", flush=True)
            return ""


def asr_analyze_chunks(chunks: list[dict], duration: float, settings: dict, monitor: dict) -> list[Segment]:
    """对已带文本的分块ASR做LLM分组判定 + 黑名单直判, 返回广告候选段。"""
    llm_segments: list[Segment] = []
    group_size = 24
    step = 16
    for begin in range(0, max(1, len(chunks)), step):
        group = chunks[begin:begin + group_size]
        if not group:
            break
        text = "\n".join(f"[{mmss(c['start'])}-{mmss(c['end'])}] {c['text']}" for c in group)
        ans = call_text_model(ASR_PROMPT.replace("{transcript}", text), settings, monitor)
        for x in ans.get("ads", []):
            try:
                llm_segments.append(Segment(
                    float(x["start_seconds"]), float(x["end_seconds"]),
                    str(x.get("brand", "")), float(x.get("confidence", 0)), "asr_llm", str(x.get("evidence", "")),
                ))
            except Exception:
                pass
        if begin + group_size >= len(chunks):
            break
    # Direct blacklist hits in ASR.
    for c in chunks:
        low = (c.get("text") or "").lower()
        hits = [b for b in settings["blacklist"] if b in low]
        if hits:
            llm_segments.append(Segment(c["start"], min(duration, c["end"]), hits[0], 0.99, "asr_blacklist", (c.get("text") or "")[:180]))
    return llm_segments


def asr_detection_from_chunks(chunks: list[dict], duration: float, settings: dict, monitor: dict) -> list[Segment]:
    """复用已识别好的分块ASR文本做广告判定(不再重新切音频/重新ASR)。"""
    return asr_analyze_chunks(chunks, duration, settings, monitor)


def asr_detection(video: Path, duration: float, settings: dict, monitor: dict) -> tuple[list[Segment], list[dict]]:
    chunks = extract_asr_chunks(video, settings)
    global SHARED_ASR
    if SHARED_ASR is None:
        SHARED_ASR = SenseVoiceRunner(settings)
    runner = SHARED_ASR
    for c in chunks:
        c["text"] = runner.transcribe(c["path"])
        print(f"  ASR {mmss(c['start'])}-{mmss(c['end'])}: {c['text'][:80]}", flush=True)
    transcript = "\n".join(f"[{mmss(c['start'])}-{mmss(c['end'])}] {c['text']}" for c in chunks)
    llm_segments = asr_analyze_chunks(chunks, duration, settings, monitor)
    (video.parent / "transcript.txt").write_text(transcript, encoding="utf-8")
    return llm_segments, chunks


ASR_PROMPT = """下面是B站视频30秒分段的ASR文本，每行带时间。请判断是否存在视频内插入的商业广告/赞助口播。
重点识别：从正常内容突然开始推销、商家名、产品卖点、优惠码、下载/注册号召，然后回到正常内容。二手回收类广告即使没有说出品牌，只要出现上门回收、旧手机/旧游戏机、估价、验机、回收红包、眼镜妹/橙色马甲等广告特征，也应判为广告。
单纯新闻讨论、吐槽、评测主题不算插入广告。广告段一般>=20秒。
只输出JSON：{"ads":[{"start_seconds":数字,"end_seconds":数字,"brand":"商家","confidence":0到1,"evidence":"依据"}]}

ASR:
{transcript}"""


def call_text_model(prompt: str, settings: dict, monitor: dict) -> dict:
    api_key = monitor["zhipu"]["api_key"]
    model = settings["text"]["model"]
    last_error = ""
    for attempt in range(3):
        try:
            r = requests.post(
                "https://open.bigmodel.cn/api/anthropic/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]},
                timeout=90,
            )
            data = r.json()
            text = "".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
            if text:
                return parse_json_object(text)
            err = data.get("error", {})
            last_error = err.get("message", str(data))[:200]
            print(f"  text model error: {last_error}", flush=True)
        except Exception as exc:
            last_error = str(exc)
            print(f"  text model attempt {attempt+1}: {exc}", flush=True)
        time.sleep(2)

    # 降级链: GLM 不可用 → DeepSeek V4 Pro
    ds = settings.get("deepseek", {}) or {}
    if ds.get("api_key"):
        try:
            r = requests.post(
                f"{ds.get('base_url', 'https://api.deepseek.com').rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {ds['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": ds.get("model", "deepseek-v4-pro"),
                    "max_tokens": 1200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=90,
            )
            data = r.json()
            text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if text:
                print("  text model degraded to DeepSeek", flush=True)
                return parse_json_object(text)
            err = data.get("error", {})
            print(f"  text model (deepseek) error: {err.get('message', str(data))[:200]}", flush=True)
        except Exception as exc:
            print(f"  text model (deepseek) attempt: {exc}", flush=True)
    else:
        print("  text model fallback skipped: deepseek.api_key 未配置", flush=True)
    return {"ads": []}


def normalize_segments(segments: list[Segment], duration: float, settings: dict) -> list[Segment]:
    segments = [s for s in segments if s.end - s.start >= int(settings["detection"]["min_duration"])]
    segments.sort(key=lambda x: (x.start, x.end))
    merged: list[Segment] = []
    gap = int(settings["detection"]["merge_gap"])
    for s in segments:
        if merged and s.start - merged[-1].end <= gap:
            m = merged[-1]
            m.end = min(duration, max(m.end, s.end))
            m.confidence = max(m.confidence, s.confidence)
            brands = [x for x in (m.brand, s.brand) if x]
            m.brand = "/".join(dict.fromkeys(brands))
            sources = [x for x in (m.source, s.source) if x]
            m.source = "+".join(dict.fromkeys(sources))
            m.evidence = (m.evidence + " | " + s.evidence).strip(" |")[:500]
        else:
            merged.append(Segment(max(0.0, s.start), min(duration, s.end), s.brand, s.confidence, s.source, s.evidence))
    return merged


def build_reply_prefix(ads: list[Segment]) -> str:
    """生成未来合并到 bili-monitor 回复正文前的广告提示。"""
    if not ads:
        return ""
    parts = []
    for x in ads:
        brand = x.brand or "商业"
        parts.append(f"检测到{brand}广告，大约位于{x.start_mmss}-{x.end_mmss}，跳过空降坐标{x.end_mmss}。")
    return " ".join(parts) + "\n"


def choose_ads(visual: list[Segment], asr: list[Segment], duration: float, settings: dict) -> list[Segment]:
    candidates = visual + asr
    strong = []
    for s in candidates:
        if s.source.endswith("blacklist") or s.confidence >= float(settings["detection"]["single_confidence"]):
            strong.append(s)
    # Cross-modal support.
    for a in visual:
        for b in asr:
            overlap = min(a.end, b.end) - max(a.start, b.start)
            if overlap >= int(settings["detection"]["min_duration"]):
                strong.append(Segment(
                    min(a.start, b.start), max(a.end, b.end), a.brand or b.brand,
                    max(a.confidence, b.confidence) + 0.08, "visual+asr", (a.evidence + " | " + b.evidence)[:400],
                ))
                break
    return normalize_segments(strong, duration, settings)


def detect_local_video(bv: str, video: Path, duration: float,
                       settings: dict, monitor: dict,
                       result_dir: Path, title: str = "") -> dict:
    """Run ad detection on a video already downloaded by bili-monitor.

    The persistent result/transcript are stored outside data/work. Bulky frame
    and WAV artifacts remain in the video work directory and are removed by the
    caller's normal cleanup after this function returns.
    """
    video = Path(video)
    visual = visual_detection(video, duration, settings, monitor)
    asr, _chunks = asr_detection(video, duration, settings, monitor)
    ads = choose_ads(visual, asr, duration, settings)
    result = _build_result(bv, title, duration, visual, asr, ads, result_dir)
    transcript = video.parent / "transcript.txt"
    if transcript.exists():
        shutil.move(str(transcript), str(result_dir / f"{bv}.transcript.txt"))
    return result


def _build_result(bv: str, title: str, duration: float, visual: list[Segment],
                  asr: list[Segment], ads: list[Segment], result_dir: Path) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "bv": bv,
        "title": title,
        "duration": duration,
        "duration_mmss": mmss(duration),
        "has_ad": bool(ads),
        "reply_prefix": build_reply_prefix(ads),
        "ads": [asdict(x) | {
            "start_mmss": x.start_mmss,
            "end_mmss": x.end_mmss,
            "skip": f"{x.start_mmss}-{x.end_mmss}",
        } for x in ads],
        "visual_candidates": [asdict(x) for x in visual],
        "asr_candidates": [asdict(x) for x in asr],
        "detected_at": time.time(),
    }
    (result_dir / f"{bv}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def detect_with_artifacts(bv: str, frames: list[Path], timestamps: list[float],
                          asr_chunks: list[dict], duration: float,
                          settings: dict, monitor: dict,
                          result_dir: Path, title: str = "") -> dict:
    """复用 bili-monitor 阶段2 已产出的关键帧与分块ASR文本做广告识别。

    不再重新ffmpeg切帧/切音频, 也不重新跑ASR, 仅保留广告所需的
    视觉窗口判定(GLM)与LLM分组判定/黑名单直判。
    """
    visual = visual_detection_from_frames(frames, timestamps, duration, settings, monitor)
    asr = asr_detection_from_chunks(asr_chunks, duration, settings, monitor)
    ads = choose_ads(visual, asr, duration, settings)
    return _build_result(bv, title, duration, visual, asr, ads, result_dir)


def qq_notify(text: str, settings: dict, monitor: dict):
    if not settings.get("qq_notify"):
        return
    q = monitor.get("channels", {}).get("qqbot", {})
    if not (q.get("appId") and q.get("clientSecret") and q.get("openid")):
        return
    r = requests.post("https://bots.qq.com/app/getAppAccessToken", json={"appId": q["appId"], "clientSecret": q["clientSecret"]}, timeout=15)
    r.raise_for_status()
    token = r.json()["access_token"]
    # Keep below C2C size limit.
    chunks, cur = [], ""
    for ch in text:
        if len((cur + ch).encode("utf-8")) > 1600:
            chunks.append(cur); cur = ch
        else:
            cur += ch
    if cur: chunks.append(cur)
    url = f"https://api.sgroup.qq.com/v2/users/{q['openid']}/messages"
    headers = {"Authorization": f"QQBot {token}", "Content-Type": "application/json"}
    for i, chunk in enumerate(chunks, 1):
        rr = requests.post(url, headers=headers, json={"content": chunk, "msg_type": 0}, timeout=15)
        print(f"  QQ notify {i}/{len(chunks)}: HTTP {rr.status_code}", flush=True)


def analyze_bv(bv: str, settings: dict, monitor: dict, keep=False) -> dict:
    bv = bv.strip()
    info = video_info(bv, monitor)
    title = info.get("title", "")
    duration_api = int(info.get("duration", 0))
    print(f"\n=== {bv} | {title} | {mmss(duration_api)} ===", flush=True)
    qq_notify(f"🧪 广告识别测试开始\nBV: {bv}\n标题: {title}\n时长: {mmss(duration_api)}", settings, monitor)
    video = download(bv, monitor, settings)
    duration = media_duration(video)
    visual = visual_detection(video, duration, settings, monitor)
    asr, chunks = asr_detection(video, duration, settings, monitor)
    ads = choose_ads(visual, asr, duration, settings)
    result = {
        "bv": bv, "title": title, "duration": duration,
        "duration_mmss": mmss(duration),
        "has_ad": bool(ads),
        "reply_prefix": build_reply_prefix(ads),
        "ads": [asdict(x) | {"start_mmss": x.start_mmss, "end_mmss": x.end_mmss, "skip": f"{x.start_mmss}-{x.end_mmss}"} for x in ads],
        "visual_candidates": [asdict(x) for x in visual],
        "asr_candidates": [asdict(x) for x in asr],
    }
    path = RUNS / bv / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ads:
        message = f"✅ 未识别到插入广告\nBV: {bv}\n标题: {title}\n时长: {mmss(duration)}"
    else:
        lines = [f"⚠️ 识别到 {len(ads)} 段广告", f"BV: {bv}", f"标题: {title}"]
        for x in ads:
            lines.append(f"品牌: {x.brand or '未知'} | {x.start_mmss}-{x.end_mmss} | 置信度 {x.confidence:.2f}\n依据: {x.evidence}")
        message = "\n".join(lines)
    qq_notify(message, settings, monitor)
    if not keep:
        # Keep result/transcript, remove bulky media artifacts.
        for p in video.parent.glob("video.mp4"):
            p.unlink()
        shutil.rmtree(video.parent / "frames", ignore_errors=True)
        shutil.rmtree(video.parent / "asr_chunks", ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bvs", nargs="+")
    parser.add_argument("--keep", action="store_true", help="保留下载视频和中间文件")
    args = parser.parse_args()
    settings, monitor = load_settings()
    DATA.mkdir(exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    all_results = []
    for bv in args.bvs:
        try:
            all_results.append(analyze_bv(bv, settings, monitor, keep=args.keep))
        except Exception as exc:
            print(f"❌ {bv}: {exc}", flush=True)
            qq_notify(f"❌ 广告识别失败\nBV: {bv}\n错误: {exc}", settings, monitor)
    out = RUNS / f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {out}", flush=True)
    for r in all_results:
        print(f"{r['bv']} has_ad={r['has_ad']} ads={[(x['brand'],x['skip']) for x in r['ads']]}")


if __name__ == "__main__":
    main()
