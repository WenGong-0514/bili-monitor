# -*- coding: utf-8 -*-
"""
本地串行推理流水线: ASR → 视觉 → 文本总结 (v5.11.0)

设计约束(22GB 显存, 不做量化):
  - 一个阶段只加载一个大模型, 用完立即释放显存, 再加载下一阶段模型
  - ASR 由调用方(bili_monitor)使用现有 SenseVoice 全局单例完成(小模型常驻)
  - 视觉阶段: Qwen2.5-VL-7B-Instruct (fp16), 3帧窗口 + 512px, 参考ASR文本做广告/画面判断
  - 文本阶段: Qwen3-8B (fp16, 禁用思考), 整合视觉+ASR 生成总结与广告空降提示

模型路径(容器内, 由命名卷持久化):
  - VL:  /root/.cache/modelscope/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master
  - LLM: /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master
"""
import base64
import gc
import io
import json
import os
import re
import sys
import time

import torch

MODEL_CACHE = os.environ.get("LOCAL_MODEL_CACHE", "/root/.cache/modelscope/models")
VL_MODEL_ID = os.environ.get("LOCAL_VL_MODEL",
                             f"{MODEL_CACHE}/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master")
LLM_MODEL_ID = os.environ.get("LOCAL_LLM_MODEL",
                              f"{MODEL_CACHE}/Qwen--Qwen3-8B/snapshots/master")

# 视觉参数(实测: 3帧+512px 每窗口约7s, 6帧+全分辨率约141s)
VL_MAX_PIXELS = int(os.environ.get("LOCAL_VL_MAX_PIXELS", "262144"))   # 512*512
VL_WIN_FRAMES = int(os.environ.get("LOCAL_VL_WIN_FRAMES", "3"))
VL_STEP_FRAMES = int(os.environ.get("LOCAL_VL_STEP_FRAMES", "3"))
VL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_VL_MAX_TOKENS", "220"))

# 文本参数
LLM_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "250"))
LLM_TEMPERATURE = float(os.environ.get("LOCAL_LLM_TEMPERATURE", "0.7"))

# 广告段判定 (v5.13.1): 时长 >= AD_MIN_DURATION 才标记为广告,
# 低于该阈值视为视频中粗略提及/一闪而过; 相邻广告窗口按截帧间隔自动合并成一段
AD_MIN_DURATION = float(os.environ.get("LOCAL_AD_MIN_DURATION", "10"))


def _free_model(model, proc=None):
    """释放模型显存, 供下一阶段使用。"""
    try:
        if model is not None:
            del model
        if proc is not None:
            del proc
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _load_vl():
    """加载 Qwen2.5-VL-7B (fp16)。返回 (model, proc, load_sec)。"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    print(f"    [本地视觉] 加载 {VL_MODEL_ID.split('/')[-2]} (fp16)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VL_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VL_MODEL_ID,
                                         min_pixels=224 * 224,
                                         max_pixels=VL_MAX_PIXELS)
    dt = time.time() - t0
    print(f"    [本地视觉] 模型加载完成 {dt:.1f}s", flush=True)
    return model, proc, dt


def _vl_predict(model, proc, imgs, prompt):
    """单窗口多帧推理, 返回模型文本输出。"""
    content = [{"type": "image", "image": img} for img in imgs] + \
              [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], images=imgs, return_tensors="pt")
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=VL_MAX_NEW_TOKENS,
                             do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)


# 视觉窗口提示词: 多帧联合判断是否广告, 同时描述画面
_VL_WINDOW_PROMPT = (
    "这些是同一个视频中连续时间段内按时间顺序抽取的{win}个关键帧(覆盖约{secs}秒画面)。"
    "请多帧联合判断: 该时间段是否出现广告推销/带货/赞助内容(UP主突然开始推销某产品、某品牌、某商家、某APP等)。\n"
    "同时简要描述该时间段画面展示了什么内容(场景/人物/动作), 40字以内。\n"
    "请只输出如下JSON(不要输出其他内容):\n"
    '{"ad_like": true或false, "brand": "广告品牌(无则空字符串)", "confidence": 0到1的小数, "scene": "画面描述"}'
)


def _parse_vl_json(text):
    """从VLM输出中尽力提取JSON字段。"""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    raw = m.group(0) if m else text
    try:
        d = json.loads(raw)
    except Exception:
        # 宽松解析: 逐字段正则
        d = {}
        for key in ("ad_like", "brand", "confidence", "scene"):
            km = re.search(rf'"{key}"\s*:\s*("?[^",}}]+"?)', raw)
            if km:
                v = km.group(1).strip().strip('"')
                if key == "ad_like":
                    d[key] = str(v).lower() in ("true", "yes", "1", "是")
                elif key == "confidence":
                    try:
                        d[key] = float(v)
                    except Exception:
                        d[key] = 0.0
                else:
                    d[key] = v
    return d


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def build_ad_prefix(ad_segments):
    """多段广告逐段标记: 生成回复前的广告空降提示。

    - 单段: "⚠️ 本视频包含转转广告(约3:00-3:24) 跳过空降坐标3:00\n\n"
    - 多段: "⚠️ 本视频包含2段广告: 转转广告(...); 某品牌广告(...)\n\n"
    """
    if not ad_segments:
        return ""
    seg_parts = []
    for ad in ad_segments:
        brand = ad.get('brand') or '某'
        seg_parts.append(
            f"{brand}广告(约{_mmss(ad.get('start', 0))}-{_mmss(ad.get('end', 0))})"
            f" 跳过空降坐标{_mmss(ad.get('start', 0))}")
    if len(seg_parts) == 1:
        return f"⚠️ 本视频包含{seg_parts[0]}\n\n"
    return f"⚠️ 本视频包含{len(seg_parts)}段广告: " + "; ".join(seg_parts) + "\n\n"


def merge_ad_segments(segments, min_duration=10.0, max_gap=2.0):
    """合并相邻广告窗口并过滤过短广告段(时长 < min_duration 的不标记)。

    截帧策略: 每2秒一帧, 视觉窗口3帧覆盖约4秒, 相邻窗口间隔=帧间隔。
    因此一段真实广告会被拆成多个窗口段; 这里按实际截帧时间戳把连续窗口
    合并成一段, 得到反映真实广告时长的合并段。max_gap 取帧间隔,
    与截帧策略保持一致(视频越长截帧越稀疏时, 该值应随帧间隔放大)。
    """
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.get("start", 0), s.get("end", 0)))
    merged = []
    for s in segs:
        if merged and s.get("start", 0) - merged[-1]["end"] <= max_gap:
            m = merged[-1]
            m["end"] = max(m["end"], s.get("end", m["end"]))
            m["confidence"] = max(float(m.get("confidence", 0) or 0),
                                  float(s.get("confidence", 0) or 0))
            brands = [b for b in (m.get("brand"), s.get("brand"))
                      if b and b != "未知"]
            m["brand"] = brands[0] if brands else (m.get("brand") or "未知")
            evs = [e for e in (m.get("evidence"), s.get("evidence")) if e]
            m["evidence"] = " | ".join(evs)[:500]
        else:
            merged.append(dict(s))
    kept = [m for m in merged if (m["end"] - m["start"]) >= min_duration]
    dropped = len(merged) - len(kept)
    if dropped:
        print(f"  [广告] 合并后 {len(merged)} 段, 过滤 <{min_duration:.0f}s 的 {dropped} 段, "
              f"保留 {len(kept)} 段", flush=True)
    return kept


def visual_stage(frame_paths, frame_ts, asr_text="", notify=None):
    """本地视觉阶段: 加载 Qwen2.5-VL, 对关键帧窗口做广告/画面判断。

    Args:
        frame_paths: 关键帧jpg路径列表(按时间升序)
        frame_ts: 与 frame_paths 等长的时间戳列表(秒)
        asr_text: 阶段1的ASR转录文本(供视觉参考)

    Returns:
        (visual_desc, ad_segments)
          visual_desc: 各窗口画面描述拼接
          ad_segments: [{brand, start, end, confidence, evidence}]
    """
    if not frame_paths:
        return "", []
    from PIL import Image

    # 结合ASR文本: 让视觉模型知道语音内容, 提升广告判断准确率
    asr_ctx = (asr_text or "").strip()
    if len(asr_ctx) > 1500:
        asr_ctx = asr_ctx[:1500] + "……"

    release_llm()  # 先释放常驻 LLM, 给视觉模型腾显存
    model, proc, _ = _load_vl()
    try:
        # 按窗口滑动
        win, step = VL_WIN_FRAMES, VL_STEP_FRAMES
        visual_parts = []
        ad_segments = []
        idx = 0
        total = len(frame_paths)
        while idx < total:
            window = frame_paths[idx:idx + win]
            ts = frame_ts[idx:idx + win]
            if not window:
                break
            t_start = ts[0] if ts else 0.0
            t_end = ts[-1] if ts else t_start
            secs = max(1, int(t_end - t_start))
            imgs = []
            for p in window:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
            if not imgs:
                idx += step
                continue
            prompt = (_VL_WINDOW_PROMPT
                       .replace('{win}', str(len(imgs)))
                       .replace('{secs}', str(secs)))
            if asr_ctx:
                prompt += f"\n\n该时间段语音内容(可能不完整):\n{asr_ctx}"
            ans = _vl_predict(model, proc, imgs, prompt)
            d = _parse_vl_json(ans) or {}
            scene = str(d.get("scene", "")).strip()
            if scene:
                visual_parts.append(f"时间{t_start:.0f}-{t_end:.0f}s: {scene}")
            ad_like = bool(d.get("ad_like"))
            brand = str(d.get("brand", "")).strip()
            conf = float(d.get("confidence", 0) or 0)
            if ad_like or brand:
                if not brand and asr_ctx:
                    # 视觉没给出品牌, 但语音上下文可能包含
                    pass
                ad_segments.append({
                    "brand": brand or "未知",
                    "start": float(t_start),
                    "end": float(t_end),
                    "confidence": conf if conf > 0 else 0.6,
                    "evidence": (scene or ans)[:200],
                })
                if notify:
                    notify(f"🛒 本地视觉窗口 {_mmss(t_start)}-{_mmss(t_end)}: "
                           f"疑似广告 brand={brand or '未知'} conf={conf:.2f}")
            print(f"    [本地视觉] 窗口 {_mmss(t_start)}-{_mmss(t_end)}: "
                  f"ad={ad_like} conf={conf} scene={scene[:40]}", flush=True)
            idx += step
        visual_desc = "\n".join(visual_parts)
        # 按截帧策略合并相邻广告窗口并过滤过短广告段(时长<10s视为粗略提及)
        if ad_segments:
            frame_interval = (frame_ts[1] - frame_ts[0]) if len(frame_ts) > 1 else 2.0
            window_gap = frame_interval * max(1, VL_STEP_FRAMES - VL_WIN_FRAMES + 1)
            print(f"  [广告] 原始窗口 {len(ad_segments)} 段, 帧间隔{frame_interval:.1f}s, "
                  f"合并间隔{window_gap:.1f}s", flush=True)
            ad_segments = merge_ad_segments(ad_segments, AD_MIN_DURATION, window_gap)
            for s in ad_segments:
                print(f"  [广告] 保留: {_mmss(s['start'])}-{_mmss(s['end'])} "
                      f"({s['end'] - s['start']:.0f}s) brand={s.get('brand') or '未知'}", flush=True)
        return visual_desc, ad_segments
    finally:
        _free_model(model, proc)


def _load_llm():
    """加载 Qwen3-8B (fp16)。返回 (model, tokenizer, load_sec)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"    [本地文本] 加载 {LLM_MODEL_ID.split('/')[-2]} (fp16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    dt = time.time() - t0
    print(f"    [本地文本] 模型加载完成 {dt:.1f}s", flush=True)
    return model, tok, dt



# ============================================================================
# 常驻 LLM 会话: 视频总结/意图分类/对话回复 共用 Qwen3-8B, 避免反复加载
# (消息处理为串行, 无并发; 加载视觉模型前会先 release_llm 腾显存)
# ============================================================================
_LLM_SESSION = {"model": None, "tok": None}


def _get_llm():
    """获取(或首次加载)常驻 Qwen3-8B。模型保持常驻, 由 release_llm() 显式释放。"""
    if _LLM_SESSION["model"] is None:
        _LLM_SESSION["model"], _LLM_SESSION["tok"], _ = _load_llm()
    return _LLM_SESSION["model"], _LLM_SESSION["tok"]


def release_llm():
    """释放常驻 LLM 显存(加载视觉模型前会自动调用)。"""
    _free_model(_LLM_SESSION["model"], _LLM_SESSION["tok"])
    _LLM_SESSION["model"] = _LLM_SESSION["tok"] = None


def local_generate(messages, system=None, max_tokens=250, temperature=None,
                   label="本地文本"):
    """本地 Qwen3-8B 文本生成(对话回复/意图分类/动态总结等), 复用常驻模型。

    Args:
        messages: [{"role": "user"/"assistant", "content": ...}, ...]
        system: 可选 system 提示
        max_tokens: 最大生成 token 数
        temperature: None=贪心解码(确定性, 适合分类); 数值=采样温度(适合对话)
        label: 日志标签

    Returns:
        生成文本; 失败/无输出返回 ""
    """
    try:
        model, tok = _get_llm()
    except Exception as e:
        print(f"    [本地文本] {label} 模型加载失败: {e}", flush=True)
        return ""
    try:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        inputs = tok([text], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            if temperature is None:
                out = model.generate(**inputs, max_new_tokens=max_tokens,
                                     do_sample=False)
            else:
                out = model.generate(**inputs, max_new_tokens=max_tokens,
                                     do_sample=True, temperature=temperature)
        dt = time.time() - t0
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        ntok = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"    [本地文本] {label} 生成 {dt:.1f}s ({ntok} tokens)", flush=True)
        return ans
    except Exception as e:
        print(f"    [本地文本] {label} 推理失败: {e}", flush=True)
        return ""


def _make_summary_prompt(char_limit: int) -> str:
    """构造本地文本总结提示词, 字数目标随时长动态调整(v5.13.0)。"""
    return ("你是B站的一名观众,刚看完一个视频。请把视频内容总结成一段自然、连贯、口语化的中文文字。"
            "语气轻松自然,偶尔带点调侃,但不要过度娱乐化。把视频讲了什么、展示了什么、核心话题是什么说清楚,"
            f"根据视频长度,总结控制在{char_limit}字以内,说清楚就停。"
            "如果语音识别文本不准,主要靠画面判断。如果信息不足以判断视频真实内容,回复'信息不足,无法准确总结',不要编造。"
            "不要提及'识别''语音识别''画面分析'等技术过程。")


def text_stage(visual_desc, asr_text, ad_segments=None, notify=None, max_tokens=None):
    """本地文本阶段: 使用常驻 Qwen3-8B, 生成总结 + 广告空降提示(模型常驻供后续对话回复复用)。

    Returns:
        (summary, ad_prefix)
          summary: 纯总结文本
          ad_prefix: 若有广告, 形如 "⚠️ 本视频包含XX广告(约mm:ss-ss:ss) 跳过空降坐标mm:ss\n\n"; 否则 ""
    """
    parts = []
    if visual_desc:
        parts.append(f"【画面分析】\n{visual_desc[:1500]}")
    asr_clean = (asr_text or "").strip()
    if asr_clean and not asr_clean.startswith("语音识别失败") and \
            not asr_clean.startswith("本地语音识别失败"):
        chinese_len = len([c for c in asr_clean if '\u4e00' <= c <= '\u9fff'])
        english_words = len([w for w in asr_clean.split()
                             if w.strip() and any(c.isalpha() for c in w)])
        if max(chinese_len, english_words) >= 4:
            parts.append(f"【语音识别文本】\n{asr_clean[:2000]}")
    if not parts:
        return "无法生成视频总结：画面识别和语音识别均未能获取到足够信息", ""

    combined = "\n\n".join(parts)
    if ad_segments:
        ad_desc = "; ".join(
            f"{s.get('brand', '未知')}({_mmss(s.get('start', 0))}-{_mmss(s.get('end', 0))})"
            for s in ad_segments)
        combined += f"\n\n【广告检测提示】画面分析检测到疑似广告: {ad_desc}"

    max_tokens = max_tokens or LLM_MAX_NEW_TOKENS
    char_limit = max(50, min(1000, int(max_tokens * 0.85)))  # 与B站1000字评论上限对齐
    model, tok = _get_llm()
    try:
        style_note = ("\n\n请直接输出总结正文: 只输出一段连贯的中文文字, "
                      "严禁使用任何标题、序号、列表、加粗、斜体、Markdown符号、emoji、分隔线, "
                      "不要出现'画面分析''语音内容''总结'等小标题或标签。")
        msgs = [
            {"role": "system", "content": _make_summary_prompt(char_limit)},
            {"role": "user", "content": combined + style_note},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        inputs = tok([text], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=True, temperature=LLM_TEMPERATURE)
        dt = time.time() - t0
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        ntok = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"    [本地文本] 生成 {dt:.1f}s ({ntok} tokens)", flush=True)
        if not ans:
            ans = "无法生成视频总结：模型无输出"
        # 广告空降前缀: 多段广告逐段标记(时长<10s的已在 visual_stage 过滤)
        ad_prefix = build_ad_prefix(ad_segments)
        return ans, ad_prefix
    except Exception as e:
        print(f"    [本地文本] 总结推理失败: {e}", flush=True)
        return "无法生成视频总结：模型推理失败", ""
