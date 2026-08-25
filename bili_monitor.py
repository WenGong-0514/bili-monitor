#!/usr/bin/env python3
"""
===========================================================================
 B站@消息监控 + 自动视频总结 + 评论区聊天 + QQ通知
===========================================================================

 架构:
   - 常驻后台进程(daemon),每15秒轮询一次B站未读@消息
   - 使用B站 unread 接口做轻量检查(at=0 则跳过,几乎零开销)
   - 发现新@消息 → 判断类型(视频/动态/官方内容) → 处理 → 回复
   - 首次@必做视频分析(下载+截帧+ASR+总结),后续复用缓存
   - 评论区聊天: @Bot + 文字 → GLM意图分类 → 回复总结/结合视频对话
   - 同一视频总结只触发一次,后续请求复用缓存
   - 主评论间消息隔离,子评论上下文仅包含@Bot与Bot回复
   - 同时通过QQ Bot主动通知用户

 ⚠️  启动前必读:
   - 配置: 必须在同目录下放置 config.json (参考 config.example.json)
   - 代理: 启动时自动检测 127.0.0.1:7890, 有则走代理无则直连, 两种都正常
   - 心跳: 没有新消息时 5 分钟输出一次心跳, 启动后别慌, 等 5 分钟

 启动:
   python3 -u bili_monitor.py >> data/logs/bili_monitor.log 2>&1 &
   (或使用 nohup: nohup python3 -u bili_monitor.py >> data/logs/bili_monitor.log 2>&1 &)

 停止:
   kill $(pgrep -f bili_monitor.py)

 日志:
   tail -f data/logs/bili_monitor.log

 依赖:
   - yt-dlp (视频下载)
   - ffmpeg (截帧 + 音频提取)
   - 阿里云百炼 Paraformer (语音识别, HTTP API调用, 无需本地模型)
   - GLM-4.6V-Flash (智谱免费视觉模型, 128K上下文, 支持视频/图片)
   - GLM-5.1 (智谱免费文本模型, 综合总结 + 聊天对话 + 意图分类)
   - requests (HTTP, GLM API调用)
   - OpenClaw CLI (QQ消息通知)
===========================================================================
"""

import json, sys, os, subprocess, time, re, glob, base64, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 强制无缓冲输出,避免 nohup 日志不完整
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# 环境初始化
# ============================================================================

os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# ============================================================================
# 配置加载 -- 从 config.json 读取
# ============================================================================

def _load_config() -> dict:
    """从脚本同目录下的 config.json 加载配置。
    
    优先级:
      1. 命令行参数指定的配置文件路径: --config <path>
      2. 环境变量 BILI_CONFIG 指定的路径
      3. 脚本同目录下的 config.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = None
    
    # 优先级1: 命令行 --config
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--config' and i < len(sys.argv) - 1:
            config_path = sys.argv[i + 1]
            break
        elif arg.startswith('--config='):
            config_path = arg.split('=', 1)[1]
            break
    
    # 优先级2: 环境变量
    if not config_path:
        config_path = os.environ.get('BILI_CONFIG')
    
    # 优先级3: 脚本同目录
    if not config_path:
        config_path = os.path.join(script_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}", file=sys.stderr)
        print(f"   请复制 config.example.json 为 config.json 并填写真实配置:", file=sys.stderr)
        print(f"     cp config.example.json config.json", file=sys.stderr)
        print(f"     nano config.json", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    print(f"📋 配置已加载: {config_path}", flush=True)
    return cfg


# 加载配置
_CFG = _load_config()

# --- B站认证 ---
SESSDATA = _CFG["bilibili"]["sessdata"]
BILI_JCT = _CFG["bilibili"]["bili_jct"]
BOT_MID = _CFG["bilibili"]["bot_mid"]
BOT_NAME = _CFG["bilibili"].get("bot_name", "")  # Bot在B站的昵称, 用于识别@消息

# --- 智谱AI API ---
ZHIPU_API_KEY = _CFG["zhipu"]["api_key"]

# --- 本地 ASR (SenseVoiceSmall + FSMN-VAD, funasr) ---
# 唯一 ASR 路径: 本地推理。在线 ASR 已全部过期下线, 不再配置云端降级链。
_ASR_CFG = _CFG.get("asr", {})
ASR_LOCAL_FIRST = _ASR_CFG.get("local_first", True)
ASR_LOCAL_MODEL = _ASR_CFG.get("local_model", "iic/SenseVoiceSmall")
ASR_LOCAL_VAD_MODEL = _ASR_CFG.get("local_vad_model", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
ASR_LOCAL_MAX_SEG_MS = int(_ASR_CFG.get("local_max_seg_ms", 15000))
ASR_LOCAL_THREADS = int(_ASR_CFG.get("local_threads", 8))
ASR_LOCAL_DEVICE = str(_ASR_CFG.get("device", "auto")).lower()
_SENSEVOICE_DEVICE = None  # 实际使用的推理设备(cpu/cuda), 模型加载后写入
_SENSEVOICE_MODEL = None  # 懒加载, 首次调用 _do_local_transcribe_sensevoice 时初始化

# --- 视觉模型降级链 ---
VISUAL_MODEL_CHAIN = _CFG.get("visual", {}).get("model_chain", [
    "glm-4.6v-flash",
    "glm-4v-flash",
])

# --- 文本模型: 兼作显示名与 API model ID（自动 lowercase 以匹配智谱接口）---
MODEL_NAME = _CFG.get("monitor", {}).get("model_name", "GLM-5.1")
TEXT_MODEL = MODEL_NAME.lower()

# --- QQ通知配置 (channels.qqbot: 官方 Bot API 直连) ---
_ch_cfg = _CFG.get("channels", {})
_qqbot_cfg = _ch_cfg.get("qqbot", {})
QQ_OPENID = _qqbot_cfg.get("openid", "")
QQ_APP_ID = _qqbot_cfg.get("appId", "")
QQ_CLIENT_SECRET = _qqbot_cfg.get("clientSecret", "")
_QQ_TOKEN_CACHE = {"token": None, "expire_at": 0.0}

# --- 代理配置 ---
_proxy_cfg = _CFG.get("proxy", {})
_proxy_host = _proxy_cfg.get("host", "127.0.0.1")
_proxy_port = _proxy_cfg.get("port", 7890)

# ============================================================================
# 路径 & 常量
# ============================================================================

_monitor_cfg = _CFG.get("monitor", {})
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

def _project_path(config_value: str, default_relative: str) -> str:
    """把配置路径限制在项目目录内，所有运行期文件都持久化保存。"""
    candidate = Path(config_value or default_relative)
    if not candidate.is_absolute():
        candidate = Path(_SCRIPT_DIR) / candidate
    try:
        candidate.resolve().relative_to(Path(_SCRIPT_DIR).resolve())
    except ValueError:
        print(f"⚠️ 拒绝项目外路径 {config_value!r}, 改用项目内 {default_relative}", flush=True)
        candidate = Path(_SCRIPT_DIR) / default_relative
    return str(candidate)

STATE_FILE = _project_path(_monitor_cfg.get("state_file"), "data/state/replied_ids.txt")
SUMMARY_FILE = _project_path(_monitor_cfg.get("summary_file"), "data/cache/video_summaries.json")
ACTIVE_THREADS_FILE = _project_path(_monitor_cfg.get("active_threads_file"), "data/state/active_threads.json")
WORK_DIR = _project_path(_monitor_cfg.get("work_dir"), "data/work")
COOKIES = f"SESSDATA={SESSDATA}; bili_jct={BILI_JCT}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
POLL_INTERVAL = _monitor_cfg.get("poll_interval", 15)
AT_FALLBACK_INTERVAL = _monitor_cfg.get("at_fallback_interval", 3600)
REPLY_FALLBACK_INTERVAL = _monitor_cfg.get("reply_fallback_interval", 3600)

# --- 文本模型降级链: GLM(付费) → DeepSeek V4 Pro ---
_deepseek_cfg = _CFG.get("deepseek", {})
DEEPSEEK_API_KEY = _deepseek_cfg.get("api_key", "")
DEEPSEEK_MODEL = _deepseek_cfg.get("model", "deepseek-v4-pro")
DEEPSEEK_BASE_URL = _deepseek_cfg.get("base_url", "https://api.deepseek.com").rstrip("/")

# --- 内嵌广告识别: 检测参数在 config.yaml, 黑名单在 blacklist.txt ---
AD_DETECTION_ENABLED = bool(_CFG.get("ad_detection", {}).get("enabled", True))
AD_RESULTS_DIR = os.path.join(_DATA_DIR, "ad_detection")
os.makedirs(AD_RESULTS_DIR, exist_ok=True)

for _persistent_file in (STATE_FILE, SUMMARY_FILE, ACTIVE_THREADS_FILE):
    os.makedirs(os.path.dirname(_persistent_file), exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# 生成 yt-dlp 使用的 Netscape cookies 文件
# B站对 --add-header 传 Cookie 的方式会返回 412, 必须用 --cookies 文件
COOKIES_FILE = os.path.join(WORK_DIR, 'bili_cookies.txt')
def _write_cookies_file():
    """将B站Cookie写入 Netscape 格式的 cookies 文件, 供 yt-dlp 使用。"""
    with open(COOKIES_FILE, 'w') as f:
        f.write('# Netscape HTTP Cookie File\n')
        f.write(f'.bilibili.com\tTRUE\t/\tFALSE\t1795513094\tSESSDATA\t{SESSDATA}\n')
        f.write(f'.bilibili.com\tTRUE\t/\tTRUE\t1795513094\tbili_jct\t{BILI_JCT}\n')
_write_cookies_file()

# ============================================================================
# 代理初始化 (依赖配置)
# ============================================================================

import socket as _socket
def _proxy_running(host='127.0.0.1', port=7890):
    try:
        s = _socket.create_connection((host, port), timeout=1)
        s.close()
        return True
    except Exception:
        return False

if _proxy_running(_proxy_host, _proxy_port):
    os.environ["http_proxy"]  = f"http://{_proxy_host}:{_proxy_port}"
    os.environ["https_proxy"] = f"http://{_proxy_host}:{_proxy_port}"
    print(f"🔄 代理已启用 ({_proxy_host}:{_proxy_port})", flush=True)
else:
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    print("🔄 代理未运行,直连模式", flush=True)


# ============================================================================
# QQ 通知
# ============================================================================

def _get_qq_access_token():
    """用 AppID+ClientSecret 换 QQ Bot access_token, 缓存到过期前 60s。"""
    now = time.time()
    if _QQ_TOKEN_CACHE["token"] and now < _QQ_TOKEN_CACHE["expire_at"] - 60:
        return _QQ_TOKEN_CACHE["token"]
    import requests as req
    try:
        r = req.post(
            "https://bots.qq.com/app/getAppAccessToken",
            json={"appId": QQ_APP_ID, "clientSecret": QQ_CLIENT_SECRET},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token")
        expires_in = float(data.get("expires_in", 7200))
        if token:
            _QQ_TOKEN_CACHE["token"] = token
            _QQ_TOKEN_CACHE["expire_at"] = now + expires_in
            return token
        print(f"  ⚠️  QQ access_token 响应异常: {r.text[:200]}")
    except Exception as e:
        print(f"  ⚠️  获取 QQ access_token 失败: {e}")
    return None


def _qq_split_text(text: str, max_bytes=1800):
    """按 UTF-8 字节长度分段(QQ C2C 文本上限 ~2000 字节,留余量)。"""
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    chunks, cur = [], ""
    for ch in text:
        if len((cur + ch).encode("utf-8")) > max_bytes:
            chunks.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        chunks.append(cur)
    return chunks


def notify_qq(text: str):
    # 优先: 官方 Bot API 直连 (channels.qqbot 配置齐全时)
    if QQ_APP_ID and QQ_CLIENT_SECRET and QQ_OPENID:
        import requests as req
        token = _get_qq_access_token()
        if not token:
            return
        url = f"https://api.sgroup.qq.com/v2/users/{QQ_OPENID}/messages"
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
        }
        chunks = _qq_split_text(text)
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            try:
                r = req.post(url, json={"content": chunk, "msg_type": 0},
                             headers=headers, timeout=15)
                if r.status_code in (200, 201, 204):
                    tag = f" ({i+1}/{total})" if total > 1 else ""
                    print(f"  📱 QQ通知已发送{tag}")
                else:
                    # token 失效时清缓存重试一次
                    if r.status_code == 401 and i == 0:
                        _QQ_TOKEN_CACHE["token"] = None
                        token = _get_qq_access_token()
                        if token:
                            headers["Authorization"] = f"QQBot {token}"
                            r = req.post(url, json={"content": chunk, "msg_type": 0},
                                         headers=headers, timeout=15)
                            if r.status_code in (200, 201, 204):
                                print(f"  📱 QQ通知已发送(token 重试)")
                                continue
                    print(f"  ⚠️  QQ通知失败: HTTP {r.status_code} {r.text[:200]}")
                    break
            except Exception as e:
                print(f"  ⚠️  QQ通知异常: {e}")
                break
        return

    # 回退: OpenClaw CLI (旧版, 需 openclaw 在 PATH)
    if not QQ_OPENID:
        return
    try:
        r = subprocess.run([
            'openclaw', 'message', 'send',
            '-t', f'qqbot:c2c:{QQ_OPENID}',
            '-m', text,
            '--json'
        ], capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and 'messageId' in r.stdout:
            print(f"  📱 QQ通知已发送(OpenClaw)")
        else:
            print(f"  ⚠️  QQ通知失败(OpenClaw): {r.stdout[:150]}")
    except Exception as e:
        print(f"  ⚠️  QQ通知异常(OpenClaw): {e}")


# ============================================================================
# 状态管理
# ============================================================================

def load_state() -> set:
    try:
        with open(STATE_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def save_state(source_id: str):
    with open(STATE_FILE, 'a') as f:
        f.write(source_id + '\n')


# ============================================================================
# 视频总结缓存
# ============================================================================

def load_summaries() -> dict:
    """加载视频总结缓存: {bv: {summary, duration, time}}"""
    try:
        with open(SUMMARY_FILE) as f:
            return json.loads(f.read() or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def load_blacklist_keywords() -> list[str]:
    """读取广告商家黑名单，供启动诊断输出。"""
    path = os.path.join(_SCRIPT_DIR, "blacklist.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return [x.strip().lower() for x in f if x.strip() and not x.strip().startswith("#")]
    except (FileNotFoundError, OSError):
        return []

def load_ad_result(bv: str) -> dict:
    """加载一个视频的持久化广告检测结果。"""
    path = os.path.join(AD_RESULTS_DIR, f"{bv}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

def save_summary(bv: str, summary: str, duration_str: str, ad_result: dict | None = None):
    """保存一条视频总结; 广告前缀单独存储，避免污染后续对话上下文。"""
    summaries = load_summaries()
    ad_result = ad_result or {}
    summaries[bv] = {
        "summary": summary,
        "duration": duration_str,
        "time": time.time(),
        "ad_prefix": ad_result.get("reply_prefix", ""),
        "ad_has": bool(ad_result.get("has_ad", False)),
        "ad_detected_at": ad_result.get("detected_at", 0),
    }
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


# ============================================================================
# 活跃评论区线程监控
# ============================================================================
# B站@消息推送只包含主评论@,不包含子评论(评论的评论)中的@。
# 因此需要额外维护"活跃线程表":记录我们回复过的主评论,
# 定期扫描这些评论下是否有新的子评论@Bot。


def load_active_threads() -> dict:
    """加载活跃评论区监控表。
    格式: {key: {oid, root_rpid, comment_type, bv, replied_rpids: [..], last_checked}}"""
    try:
        with open(ACTIVE_THREADS_FILE) as f:
            return json.loads(f.read() or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_active_thread(key: str, oid: str, root_rpid: str, comment_type, bv: str = '',
                      initial_rpids: list = None):
    """注册/更新一个活跃评论区线程。
    initial_rpids: 初始已处理rpid列表(如主评论自身),避免重复处理。"""
    threads = load_active_threads()
    if key not in threads:
        threads[key] = {
            "oid": str(oid),
            "root_rpid": str(root_rpid),
            "comment_type": comment_type,
            "bv": bv,
            "replied_rpids": initial_rpids or [str(root_rpid)],
            "last_checked": time.time()
        }
    else:
        # 已存在:更新last_checked,合并initial_rpids
        existing = set(threads[key].get("replied_rpids", []))
        for rpid in (initial_rpids or []):
            existing.add(str(rpid))
        threads[key]["replied_rpids"] = list(existing)[-200:]
        threads[key]["last_checked"] = time.time()
    with open(ACTIVE_THREADS_FILE, 'w') as f:
        json.dump(threads, f, ensure_ascii=False, indent=2)

# ============================================================================
# B站 API 封装
# ============================================================================

_API_RISK_CODES = {-412, -799, 429}
_API_BACKOFF_SECONDS = 15 * 60
_API_BACKOFF_UNTIL = 0.0
_api_consecutive_failures = 0

def _record_api_failure(reason: str, detail: str = "") -> dict:
    """记录连续传输/风控失败；达到阈值后暂停请求并通知。"""
    global _api_consecutive_failures, _API_BACKOFF_UNTIL
    if time.time() < _API_BACKOFF_UNTIL:
        return {"code": -429, "message": "API backoff active"}
    _api_consecutive_failures += 1
    message = f"{reason}{(': ' + detail) if detail else ''}"
    if _api_consecutive_failures >= 3:
        _API_BACKOFF_UNTIL = time.time() + _API_BACKOFF_SECONDS
        _api_consecutive_failures = 0
        print(f"[{time.strftime('%H:%M:%S')}] 🛡️ B站API连续异常，暂停请求{_API_BACKOFF_SECONDS}秒: {message}", flush=True)
        try:
            notify_qq(f"⚠️ bili-monitor进入API退避\n原因: {message}\n暂停: {_API_BACKOFF_SECONDS}秒")
        except Exception:
            pass
    return {"code": -1, "message": message}

def _curl_json(args: list) -> dict:
    global _api_consecutive_failures
    if time.time() < _API_BACKOFF_UNTIL:
        return {"code": -429, "message": "API backoff active"}
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as e:
        return _record_api_failure("请求超时", str(e))
    except Exception as e:
        return _record_api_failure("请求失败", str(e))
    try:
        result = json.loads(r.stdout)
    except json.JSONDecodeError:
        return _record_api_failure("响应不是JSON", r.stdout[:120])
    code = result.get("code")
    if code == 0:
        _api_consecutive_failures = 0
        return result
    if code in _API_RISK_CODES:
        return _record_api_failure(f"B站风控/限流 code={code}", result.get("message", ""))
    return result

def api_get(url: str) -> dict:
    return _curl_json([
        'curl', '-s', url,
        '-H', f'User-Agent: {UA}',
        '-H', 'Referer: https://www.bilibili.com/',
        '-H', f'Cookie: {COOKIES}'
    ])

def api_post_form(url: str, data: dict) -> dict:
    args = [
        'curl', '-s', '-X', 'POST', url,
        '-H', f'User-Agent: {UA}',
        '-H', 'Referer: https://www.bilibili.com/',
        '-H', f'Cookie: {COOKIES}',
        '-H', 'Content-Type: application/x-www-form-urlencoded'
    ]
    for k, v in data.items():
        args += ['--data-urlencode', f'{k}={v}']
    return _curl_json(args)


# ============================================================================
# B站 API: 回复评论
# ============================================================================

def reply_comment(oid: str, root_id: str, parent_id: str, message: str, comment_type) -> dict:
    """
    在B站内容下回复评论(视频/动态通用)。

    Args:
        oid: 内容的subject_id
        root_id: 根评论ID(被@的评论所在的最上层评论)
        parent_id: 被直接回复的评论ID
        message: 回复文本
        comment_type: 1=视频, 11=动态, 12=专栏等 (来自@消息的business_id)

    Returns:
        API响应dict, code=0表示成功
    """
    data = {
        'oid':     str(oid),
        'type':    str(comment_type),
        'root':    str(root_id),
        'parent':  str(parent_id),
        'message': message,
        'csrf':    BILI_JCT,
    }
    return api_post_form('https://api.bilibili.com/x/v2/reply/add', data)


def check_reply_audit(oid: str, rpid: str, comment_type) -> int:
    """检查刚发送的评论是否被B站审核系统拦截。

    通过 GET /x/v2/reply/reply 接口查询该评论的 state 字段:
      - state=1: 正常可见
      - state=17: 审核中/仅自己可见(被拦截)

    Args:
        oid: 内容的subject_id
        rpid: 评论ID (reply/add 返回的 rpid)
        comment_type: 评论区类型

    Returns:
        state 值 (1=可见, 17=被拦), 0=查询失败
    """
    try:
        d = api_get(
            f"https://api.bilibili.com/x/v2/reply/reply"
            f"?type={comment_type}&oid={oid}&root={rpid}&pn=1&ps=1"
        )
        if d.get('code') == 0:
            root = d.get('data', {}).get('root', {})
            if root:
                return root.get('state', 0)
    except Exception as e:
        print(f"    审核状态查询异常: {e}")
    return 0


def send_reply_with_audit_check(oid: str, root_id: str, parent_id: str,
                                 message: str, comment_type,
                                 notify_callback=None) -> tuple:
    """发送评论回复,并在发送后检查是否被审核拦截。

    流程:
      1. 调用 reply_comment 发送回复
      2. 检查返回值中的 state 字段(立即判断)
      3. 如果 state=1, 等45秒后再查一次(确认未被延迟拦截)
      4. 如果被拦截(state=17), 发送一条抱歉通知

    Args:
        oid, root_id, parent_id, message, comment_type: 同 reply_comment
        notify_callback: QQ通知回调

    Returns:
        (success: bool, resp: dict) 元组
    """
    # 1. 发送回复
    resp = reply_comment(oid, root_id, parent_id, message, comment_type)

    if resp.get('code') != 0:
        return False, resp

    # 2. 从返回值中获取 rpid 和 state
    reply_data = resp.get('data', {})
    rpid = str(reply_data.get('rpid', ''))
    immediate_state = reply_data.get('state', 0)
    print(f"  ✅ 回复发送成功 (rpid={rpid}, state={immediate_state})")

    # 3. 如果立即就知道被拦了
    if immediate_state == 17:
        print(f"  ⚠️  回复被审核拦截(仅自己可见),发送抱歉通知")
        _send_audit_fail_notice(oid, root_id, rpid, comment_type, notify_callback)
        return True, resp  # 发送本身是成功的,只是被审核拦了

    # 4. state=1 或未知 → 等45秒后再确认(防止延迟审核)
    if rpid:
        # 启动异步检查(不阻塞主循环)
        _schedule_audit_check(oid, rpid, comment_type, notify_callback, delay=45)

    return True, resp


def _send_audit_fail_notice(oid: str, root_id: str, parent_id: str,
                             comment_type, notify_callback=None):
    """发送审核被拦的抱歉通知。"""
    notice = "抱歉，刚才的回复似乎被B站审核系统拦截了（仅自己可见），正在排查原因。"
    try:
        resp = reply_comment(oid, root_id, parent_id, notice, comment_type)
        if resp.get('code') == 0:
            print(f"  📢 已发送审核拦截通知")
        else:
            print(f"  ⚠️  审核拦截通知发送失败: {resp.get('message', '')}")
    except Exception as e:
        print(f"  ⚠️  审核拦截通知异常: {e}")


def _schedule_audit_check(oid: str, rpid: str, comment_type,
                           notify_callback, delay: int = 45):
    """延迟检查审核状态(不阻塞主循环)。

    使用简单的 threading.Timer 在后台等待后检查。
    """
    import threading

    def _check():
        try:
            state = check_reply_audit(oid, rpid, comment_type)
            if state == 17:
                print(f"  ⚠️  延迟审核拦截: rpid={rpid} (发送{delay}秒后被标记为仅自己可见)")
                # 查找该评论的 root 和 parent 来发通知
                # 由于我们没有保存 root_id, 这里用 rpid 作为 root
                _send_audit_fail_notice(oid, rpid, rpid, comment_type, notify_callback)
                if notify_callback:
                    notify_callback("⚠️ B站回复被审核拦截(延迟)\n已发送抱歉通知")
            elif state == 1:
                print(f"  ✅ 审核确认通过: rpid={rpid}")
            else:
                print(f"  ℹ️  审核状态未知: rpid={rpid}, state={state}")
        except Exception as e:
            print(f"  ⚠️  延迟审核检查异常: {e}")

    timer = threading.Timer(delay, _check)
    timer.daemon = True  # 主进程退出时自动结束
    timer.start()
    print(f"  ⏱️  已安排{delay}秒后审核状态检查")


# ============================================================================
# B站 API: 获取评论下的所有回复(子评论)
# ============================================================================

def fetch_comment_thread(oid, root_rpid, comment_type, max_pages=5):
    """
    获取某个根评论下的所有回复(子评论)。

    接口: /x/v2/reply/reply
    参数:
      - type: 评论区类型(1=视频,11=动态)
      - oid: 内容ID
      - root: 根评论rpid
      - pn: 页码
      - ps: 每页数量(最大20)

    返回: 按时间排序的子评论列表(最新的在前)
    注意: 也包含root评论本身的content(用于提取@消息的原始文本)
    """
    all_replies = []
    for pn in range(1, max_pages + 1):
        d = api_get(
            f"https://api.bilibili.com/x/v2/reply/reply"
            f"?type={comment_type}&oid={oid}&root={root_rpid}&pn={pn}&ps=20"
        )
        if d.get('code') != 0:
            break
        data = d.get('data', {})

        # 第一页:也把root评论加入(它包含原始@消息的文本)
        if pn == 1:
            root_comment = data.get('root')
            if root_comment:
                all_replies.append(root_comment)

        replies = data.get('replies')
        if replies is None:
            # replies可能为null(无子评论或API返回异常)
            break
        if not replies:
            break
        all_replies.extend(replies)
        # 检查是否还有下一页
        page_info = data.get('page', {})
        if page_info.get('num', 0) * page_info.get('size', 0) >= page_info.get('count', 0):
            break
        time.sleep(0.3)
    return all_replies


def fetch_top_level_comments(oid, comment_type, max_pages=3):
    """
    获取内容的顶级评论列表。

    用于: 当通过子评论@Bot时,需要找到它的根评论。

    接口: /x/v2/reply/main
    """
    all_replies = []
    for pn in range(1, max_pages + 1):
        d = api_get(
            f"https://api.bilibili.com/x/v2/reply/main"
            f"?type={comment_type}&oid={oid}&mode=3&pn={pn}&ps=20"
        )
        if d.get('code') != 0:
            break
        data = d.get('data', {})
        replies = data.get('replies', [])
        if not replies:
            break
        all_replies.extend(replies)
        page_info = data.get('page', {})
        if page_info.get('num', 0) * page_info.get('size', 0) >= page_info.get('count', 0):
            break
        time.sleep(0.3)
    return all_replies


# ============================================================================
# 评论区上下文提取
# ============================================================================

def extract_dialog_context(sub_replies: list, our_mid: int) -> list:
    """
    从子评论列表中提取 @Bot 与 Bot回复 的对话上下文。

    规则:
    - 只保留包含@Bot(或@到Bot)的消息
    - 以及Bot(mid=our_mid)自己发出的回复
    - 按时间排序

    Args:
        sub_replies: fetch_comment_thread 返回的子评论列表
        our_mid: Bot的B站UID

    Returns:
        按时间顺序排列的消息列表, 每项: {role: "user"/"assistant", content: str, mid: int}
    """
    context = []

    for reply in sub_replies:
        mid = reply.get('mid', 0)
        content_obj = reply.get('content', {})
        message = content_obj.get('message', '')
        rpid = reply.get('rpid', 0)
        ctime = reply.get('ctime', 0)

        # Bot自己的回复 → assistant
        if mid == our_mid:
            # 过滤掉已知的fallback消息(避免污染对话上下文)
            skip_patterns = [
                "抱歉,当前模型暂时无法响应",
                "当前模型无法访问到",
                "已经帮您总结过了",
                "内容由于官方机制",
            ]
            should_skip = any(p in message for p in skip_patterns)
            if not should_skip:
                context.append({
                    "role": "assistant",
                    "content": message,
                    "mid": mid,
                    "rpid": rpid,
                    "ctime": ctime
                })
            continue

        # 检查是否@了Bot (多种检测方式)
        has_at_bot = False

        # 方式1: 文本中包含 "@Bot"
        if f'@{BOT_NAME}' in message:
            has_at_bot = True

        # 方式2: members字段中查找 (B站新版API在members中包含@信息)
        if not has_at_bot:
            members = content_obj.get('members', [])
            if members and isinstance(members, list):
                for member in members:
                    if member.get('mid') == str(our_mid) or member.get('mid') == our_mid:
                        has_at_bot = True
                        break

        # 方式3: at_uids字段 (旧版或特定接口)
        if not has_at_bot:
            at_uids_raw = content_obj.get('at_uids', '')
            if at_uids_raw:
                try:
                    if isinstance(at_uids_raw, str):
                        at_uids = [int(x) for x in at_uids_raw.split(',') if x.strip().isdigit()]
                    else:
                        at_uids = [int(at_uids_raw)] if isinstance(at_uids_raw, (int,)) else []
                    if our_mid in at_uids:
                        has_at_bot = True
                except (ValueError, TypeError):
                    pass

        if has_at_bot:
            # 提取@Bot之后的有效文字(去掉@Bot前缀)
            user_text = extract_message_after_at(message, BOT_NAME)
            context.append({
                "role": "user",
                "content": user_text,
                "mid": mid,
                "rpid": rpid,
                "ctime": ctime
            })

    # 按时间排序
    context.sort(key=lambda x: x.get('ctime', 0))
    return context


def extract_message_after_at(text: str, at_name: str = "") -> str:
    """
    提取@Bot之后的文字内容。

    规则:
    - "@Bot" 在开头 → 取后面内容
    - "@Bot" 在中间/末尾且后无文字 → 返回空(表示纯@,触发总结)
    - 忽略@之前的无关文字
    - B站子评论前缀 "回复 @Bot :" 会被自动去掉

    Args:
        text: 原始评论文本
        at_name: 被@的名字

    Returns:
        @之后的有效文字(可能为空字符串)
    """
    # 先去除B站自动添加的 "回复 @xxx :" 前缀
    # 格式: "回复 @Bot :" 或 "回复 @某人 :@Bot xxx"
    reply_prefix = re.match(rf'回复\s*@{at_name}\s*[::]\s*', text)
    if reply_prefix:
        text = text[reply_prefix.end():]

    # 匹配 @Bot (可能有空格、标点等前缀)
    pattern = rf'@{at_name}\s*'
    match = re.search(pattern, text)
    if not match:
        return text.strip()

    after = text[match.end():].strip()
    return after


# ============================================================================
# 意图分类 (GLM-5.1)
# ============================================================================

def load_frame_cache(bv: str) -> list:
    """加载已缓存的关键帧base64列表。"""
    cache_path = f"{WORK_DIR}/frames_cache_{bv}.json"
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def visual_query_frames(frame_cache: list, user_question: str, api_key: str) -> str:
    """用 GLM-4.6V-Flash 针对用户的具体问题重新查看关键帧。
    
    当用户追问视频中的具体内容时,把关键帧重新发给视觉模型,
    让它针对用户的提问仔细查看画面细节。
    
    Args:
        frame_cache: load_frame_cache() 返回的关键帧base64列表
        user_question: 用户的具体问题
        api_key: 智谱API Key
    
    Returns:
        视觉模型的回答
    """
    if not frame_cache:
        return ""
    import requests as req

    # 选取均匀分布的帧(最多15帧,避免请求过大)
    n_frames = min(15, len(frame_cache))
    step = len(frame_cache) / n_frames
    selected = [frame_cache[int(i * step)] for i in range(n_frames)]

    content = []
    for frame in selected:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame['b64']}"}
        })
    content.append({
        "type": "text",
        "text": f"请仔细查看这些视频关键帧,回答用户的问题。用户问: {user_question}\n\n请根据画面内容给出准确、具体的回答。如果画面中能看到答案就详细描述,如果看不清就如实说看不清。"
    })

    return _call_visual_model(content, api_key, max_tokens=500, timeout=60)


def _call_text_with_fallback(system: str, messages: list, max_tokens: int,
                             timeout: int = 60, label: str = "文本模型") -> str:
    """文本生成主链路: GLM(付费) → DeepSeek V4 Pro 降级。

    返回空字符串表示两条链路都失败。
    """
    import requests as _req
    errors = []

    # 主链路: 智谱 GLM (Anthropic 兼容接口)
    if ZHIPU_API_KEY:
        payload = {"model": TEXT_MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            payload["system"] = system
        try:
            resp = _req.post(
                "https://open.bigmodel.cn/api/anthropic/v1/messages",
                headers={
                    "x-api-key": ZHIPU_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
                timeout=timeout
            )
            data = resp.json()
            text = "".join(
                x.get("text", "") for x in data.get("content", [])
                if x.get("type") == "text" and x.get("text")
            ).strip()
            if text:
                return text
            err = data.get("error", {})
            errors.append(f"GLM: {err.get('message', str(data))[:120]}")
        except Exception as e:
            errors.append(f"GLM: {e}")
    else:
        errors.append("GLM: 未配置 zhipu.api_key")

    # 降级链: DeepSeek V4 Pro (OpenAI 兼容接口)
    if DEEPSEEK_API_KEY:
        ds_messages = []
        if system:
            ds_messages.append({"role": "system", "content": system})
        ds_messages.extend(messages)
        try:
            resp = _req.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "max_tokens": max_tokens,
                    "messages": ds_messages
                },
                timeout=timeout
            )
            data = resp.json()
            text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if text:
                print(f"  ⚠️ {label} GLM不可用,已降级至 {DEEPSEEK_MODEL}", flush=True)
                return text
            err = data.get("error", {})
            errors.append(f"DeepSeek: {err.get('message', str(data))[:120]}")
        except Exception as e:
            errors.append(f"DeepSeek: {e}")
    else:
        errors.append("DeepSeek: 未配置 api_key")

    print(f"  ⚠️ {label} 主链路与降级链均失败: {' | '.join(errors)}", flush=True)
    return ""


def classify_user_intent(user_message: str, has_video_summary: bool, api_key: str) -> str:
    """
    用GLM-5.1判断用户意图: 'summary' 还是 'chat'。

    规则基础:
    - 纯@Bot(无额外文字) → summary(由调用方判断,此函数处理有文字的情况)
    - "@Bot 总结一下" / "@Bot 总结" → summary
    - 其他带文字的@ → 用LLM分类

    Args:
        user_message: @Bot之后的有效文字(已去除@前缀)
        has_video_summary: 该视频是否已有总结

    Returns:
        "summary" 或 "chat" 或 "video_chat"
    """
    if not user_message or not user_message.strip():
        return "summary"

    msg_lower = user_message.strip().lower()

    # 快速规则匹配:明确的总结请求
    summary_keywords = [
        '总结', '总结一下', '帮忙总结', '概括', '概括一下',
        '讲什么', '讲了什么', '什么内容', '内容是什么',
        '分析一下这个视频', '分析视频', '视频总结',
        '评价', '评价一下', '点评', '点评一下',
    ]
    for kw in summary_keywords:
        if kw in msg_lower:
            return "summary"

    # 短消息启发式:只有明确像"看看/看下/瞧瞧"之类的才判为 summary
    # 像"搞莫子/干嘛/在吗/哈喽"等日常对话不应触发视频下载
    short_summary_patterns = [
        '看看', '看下', '看一下', '瞧瞧', '瞧瞧这个',
        '这是啥', '啥内容', '这是什么', '这啥', '讲啥', '讲什么',
        '概括', '帮我看看',
    ]
    # 常见的闲聊/水评论/语气词 → 直接判为 chat,不走LLM(省钱省时间)
    chat_patterns = [
        '阿巴', '啊啊', '哈哈哈哈', '哈哈哈', '呵呵', '嘿嘿', '嘻嘻',
        '666', '777', '888', '999', '好', '好的', '收到', 'OK',
        '牛', '牛逼', 'nb', 'NB', '厉害', '强',
        '草', '艹', '啊这', '乐', '笑死', '绷不住',
        'doge', '是的', '对的', '没错', '确实',
        '在吗', '你好', '哈喽', 'hello', 'hi',
        '？', '?', '。。', '...', '。。。',
        '搞莫子', '干嘛', '咋了', '怎么', '什么鬼',
        '谢谢', '感谢', '多谢', '辛苦了',
        '加油', '支持', '赞', '酷', 'nice',
    ]
    if len(msg_lower) <= 8:
        # 先检查是否是闲聊/水评论
        for p in chat_patterns:
            if p in msg_lower:
                return "chat"
        is_summary_like = any(p in msg_lower for p in short_summary_patterns)
        if is_summary_like:
            return "summary"
        # 不匹配 → 走LLM分类,不要让"搞莫子"之类触发视频下载

    # LLM分类
    import requests as req
    prompt = f"""分析用户消息,判断意图。只回复一个词: summary 或 chat 或 video_chat。

- summary: 用户想要获取视频/动态内容的总结、概括、分析。包括纯@、泛泛的问候、没有明确问题的情况。**当用户消息不明确或比较模糊时,优先判为 summary**
- video_chat: 用户在追问视频中的具体内容细节(如某个画面、某个时刻、某个人说了什么、穿了什么等,需要重新查看视频画面)
- chat: 用户有明确的、与视频总结无关的问题或话题,且不是泛泛的问候

⚠️ 判断原则: 很多时候用户@了多个人(包括你),但不一定想和你聊天。如果用户的文字比较泛(如问候、感叹、日常用语),应判为 summary 而不是 chat。只有用户确实提出了明确的问题或话题才判为 chat。

用户消息: {user_message}

意图:"""

    result = _call_text_with_fallback(
        None, [{"role": "user", "content": prompt}],
        max_tokens=10, timeout=20, label="意图分类"
    ).strip().lower()
    if 'video_chat' in result:
        return "video_chat"
    elif 'summary' in result:
        return "summary"
    elif result:
        return "chat"

    # 默认:有文字就当作聊天
    return "chat"


# ============================================================================
# 对话回复 (GLM-5.1)
# ============================================================================

def generate_chat_reply(
    context: list,
    user_message: str,
    video_summary: str,
    video_title: str,
    api_key: str,
    visual_context: str = ''
) -> str:
    """
    用GLM-5.1生成对话回复。

    将对话上下文(只包含@Bot的消息和Bot的回复)、视频总结、
    当前用户消息一起传给大模型,生成自然回复。

    Args:
        context: extract_dialog_context() 返回的对话历史
        user_message: 当前用户的@消息文字(去除@前缀)
        video_summary: 该视频的总结文本(如有)
        video_title: 视频标题
        api_key: 智谱API Key

    Returns:
        生成的回复文本
    """
    import requests as req

    # 构建消息
    messages = []

    # System prompt
    system_text = "你是一个B站AI助手。你在B站评论区与用户对话。"
    if video_summary and video_summary not in ("", "暂无总结"):
        system_text += f"\n\n重要:你已经看过这个视频《{video_title}》,以下是视频内容总结:\n{video_summary[:1500]}"
        system_text += "\n\n你的回复必须结合视频内容。即使对方只是在闲聊,也要自然地关联到视频相关的话题。不要无视视频内容进行空对空的对话。"
    else:
        system_text += "\n\n注意:你没有看过相关视频,不要假装了解视频内容。"
    if visual_context:
        system_text += f"\n\n你刚刚重新仔细查看了视频画面,以下是你看到的:\n{visual_context[:1000]}\n请根据画面内容给出准确回答。"
    system_text += "\n\n要求:回复简洁自然,不要过度客套。可以用适度的幽默感。如果有上下文(之前对话过),要带入上下文。回复上限1000字,但不要凑字数——能把事情说清楚就够了,简短有力比冗长更好。如果几句话就能说明白,就不要写长。\n\n重要:不要在回复中加@用户名。B站评论区会自动把你的回复放在对应评论下方,不需要你手动@对方。绝对不要输出任何'@xxx'格式的内容。\n\n关键:你在回复中使用的所有素材(画面/语音/字幕等)都来自你自己的观察,不要提及\"识别\"、\"语音识别\"、\"字幕机翻\"、\"识别错误\"、\"听不清\"、\"画面分析\"等任何暗示技术过程或识别质量的说法。即使素材明显有误,也要直接给出你的判断,不要解释\"识别错了\"、\"大概是空耳\"、\"可能是误识别\"。把可能出错的片段当成你自己听到/看到的真实内容来叙述。"

    # GLM-5.1 Anthropic接口不支持system角色,改为放在第一条user消息前面
    messages = [
        {"role": "user", "content": f"[系统指令]\n{system_text}"},
        {"role": "assistant", "content": "收到,我会按照要求回复。"}
    ]

    # 对话历史(最近10轮,避免过长)
    recent_context = context[-10:]
    for msg in recent_context:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user" and content:
            messages.append({"role": "user", "content": content})
        elif role == "assistant" and content:
            # 去除系统前缀(如 "Hello,是你召唤来了我..." 等)
            clean_content = re.sub(
                r'^Hello,是你召唤来了我.*?\n(?:视频时长:.*?\n)?\n?',
                '', content
            ).strip()
            # 去除 "回复 @xxx :" 前缀 — B站已通过API处理回复链,不需要在文本中@
            clean_content = re.sub(r'^回复\s*@\S+\s*[::：]\s*', '', clean_content).strip()
            if clean_content:
                messages.append({"role": "assistant", "content": clean_content})

    # 当前用户消息
    messages.append({"role": "user", "content": user_message if user_message else "你好"})

    reply_text = _call_text_with_fallback(
        None, messages, max_tokens=500, timeout=90, label="对话回复"
    )
    if reply_text:
        # 安全兜底: 去掉模型可能生成的@xxx前缀
        reply_text = re.sub(r'^回复\s*@\S+\s*[::：]\s*', '', reply_text).strip()
        reply_text = re.sub(r'@\S+', '', reply_text).strip()
        return reply_text
    return "抱歉,当前模型暂时无法响应,请稍后再试。"


# ============================================================================
# 视频信息 & 版权检测
# ============================================================================

def get_video_info(bv: str) -> dict:
    """
    获取B站视频详情。

    接口: /x/web-interface/view

    返回字段:
    - title: 视频标题
    - duration: 时长(秒)
    - copyright: 1=自制, 非1=转载/官方
    - rights.download: 0=不可下载(番剧/电影等)
    - stat.view/stat.danmaku/...

    Returns:
        视频信息dict, 失败返回空dict
    """
    d = api_get(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}")
    if d.get('code') == 0:
        return d.get('data', {})
    print(f"  ⚠️  视频信息API错误: {d.get('message', '')}")
    return {}


def is_official_content(bv: str) -> tuple:
    """
    检测视频是否为官方内容(番剧/电影/纪录片等,无法下载分析)。

    Returns:
        (is_official: bool, reason: str)
    """
    info = get_video_info(bv)
    if not info:
        return (True, "无法获取视频信息")

    # 检测1(已移除): copyright != 1 只说明是转载而非官方内容
    # B站 copyright 字段: 1=自制 2=转载。普通up经常发转载视频,
    # 转载视频完全可以下载分析,不应被误拦截。
    # 真正的官方内容由下面的检测2(版权保护)和检测3(特殊分区)来判断。

    # 检测2: rights.download = 0 → 明确不可下载(番剧/电影等)
    rights = info.get('rights', {})
    if rights.get('download', 1) == 0:
        return (True, "该内容受版权保护(番剧/电影/纪录片)")

    # 检测3: 特殊分区(番剧/电影/纪录片/电视剧)
    # tid: 1=动画, 13=番剧, 23=电影, 167=纪录片, 11=电视剧, 177=国创
    official_tids = {13, 23, 167, 11, 177}
    if info.get('tid', 0) in official_tids:
        return (True, f"该内容属于官方专区(番剧/电影/纪录片等)")

    # 检测4: 尝试下载,看yt-dlp报错
    # 这里做预检:通过API返回的pages字段判断
    pages = info.get('pages', [])
    if not pages:
        return (True, "视频无有效分P")

    return (False, "")


# ============================================================================
# 视频下载
# ============================================================================

def get_video_duration(bv: str) -> int:
    try:
        d = api_get(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}")
        if d.get('code') == 0:
            return d['data'].get('duration', 0)
    except Exception:
        pass
    return 0


def _download_single_p(bv: str, p_index: int, output_path: str, fmt: str) -> bool:
    """下载分P视频的指定P。"""
    url = f"https://www.bilibili.com/video/{bv}?p={p_index}"
    r = subprocess.run([
        'yt-dlp',
        '-f', fmt,
        '--merge-output-format', 'mp4',
        '-o', output_path,
        '--cookies', COOKIES_FILE,
        '--add-header', 'Referer: https://www.bilibili.com/',
        '--no-warnings',
        url
    ], capture_output=True, text=True, timeout=300)
    return os.path.exists(output_path)


def download_video(bv: str, output_path: str) -> bool:
    """下载B站视频（含全部分P）。

    使用 --cookies 文件传递Cookie(而非 --add-header),
    避免 B站 playinfo API 返回 412 Precondition Failed。
    格式优先 30016+30216 (360p+64k audio), 失败则回退到 best。

    分P处理: 先检测分P数量, 多P时逐个下载再合并为单文件。
    """
    url = f"https://www.bilibili.com/video/{bv}"

    # 检测分P数量
    playlist_info = subprocess.run([
        'yt-dlp',
        '--flat-playlist',
        '--dump-json',
        '--cookies', COOKIES_FILE,
        '--add-header', 'Referer: https://www.bilibili.com/',
        '--no-warnings',
        url
    ], capture_output=True, text=True, timeout=60)

    p_count = 0
    for line in playlist_info.stdout.strip().split('\n'):
        if line.strip():
            p_count += 1

    if p_count <= 1:
        # 单P视频, 直接下载
        # 格式链: 30016+30216(旧版合并流) → bestvideo+bestaudio(新版分离流) → best(兜底)
        for fmt in ['30016+30216', 'bestvideo+bestaudio', 'best']:
            r = subprocess.run([
                'yt-dlp',
                '-f', fmt,
                '--merge-output-format', 'mp4',
                '-o', output_path,
                '--cookies', COOKIES_FILE,
                '--add-header', 'Referer: https://www.bilibili.com/',
                '--no-warnings',
                url
            ], capture_output=True, text=True, timeout=300)
            if os.path.exists(output_path):
                return True
            if fmt == '30016+30216' and 'Requested format is not available' not in r.stderr:
                print(f"  yt-dlp 错误: {r.stderr[:200]}", flush=True)
                break
        return os.path.exists(output_path)

    # 多P视频: 逐P下载 → ffmpeg concat 合并
    print(f"  检测到 {p_count} 个分P, 开始逐P下载...", flush=True)
    tmp_dir = os.path.dirname(output_path)
    part_files = []
    # 格式链: 30016+30216(旧版合并流) → bestvideo+bestaudio(新版分离流) → best(兜底)
    formats_to_try = ['30016+30216', 'bestvideo+bestaudio', 'best']

    for i in range(1, p_count + 1):
        part_path = f"{tmp_dir}/part_p{i}.mp4"
        print(f"    下载 P{i}/{p_count}...", flush=True)
        downloaded = False
        for fmt in formats_to_try:
            if _download_single_p(bv, i, part_path, fmt):
                downloaded = True
                break
        if not downloaded:
            print(f"    ⚠️ P{i} 下载失败", flush=True)
            # 已下载的部分仍然可用, 继续合并
            break
        if os.path.exists(part_path):
            part_files.append(part_path)

    if not part_files:
        return False

    if len(part_files) == 1:
        # 只成功下载了一个P
        import shutil
        shutil.move(part_files[0], output_path)
        return os.path.exists(output_path)

    # 用 ffmpeg concat demuxer 合并
    concat_list = f"{tmp_dir}/concat_list.txt"
    with open(concat_list, 'w') as f:
        for pf in part_files:
            f.write(f"file '{os.path.basename(pf)}'\n")

    concat_r = subprocess.run([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list,
        '-c', 'copy',
        '-y',
        output_path
    ], capture_output=True, text=True, timeout=180, cwd=tmp_dir)

    # concat copy 失败时重编码
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        print(f"    concat copy 失败, 尝试重编码合并...", flush=True)
        concat_r = subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list,
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-y',
            output_path
        ], capture_output=True, text=True, timeout=600, cwd=tmp_dir)

    # 清理临时文件
    for pf in part_files:
        if os.path.exists(pf):
            os.remove(pf)
    if os.path.exists(concat_list):
        os.remove(concat_list)

    success = os.path.exists(output_path) and os.path.getsize(output_path) > 1024
    if success:
        print(f"  ✅ {len(part_files)}P 合并完成", flush=True)
    return success


# ============================================================================
# 截帧 + 视觉分析(GLM-4V-Flash)
# ============================================================================

def extract_frames(video_path: str, frames_dir: str, interval: int = 2) -> list:
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps=1/{interval},scale=640:-1',
        '-q:v', '3',
        '-y',
        f'{frames_dir}/frame_%04d.jpg'
    ], capture_output=True, text=True, timeout=300)
    return sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))


def _call_visual_model(content: list, api_key: str, max_tokens: int = 150, timeout: int = 30) -> str:
    """调用智谱视觉模型,按 VISUAL_MODEL_CHAIN 降级重试。
    
    每个模型最多3次重试(含429限流/1305并发),3次均失败自动切换下一个模型。
    所有模型都失败才返回空字符串。
    """
    import requests as req
    for model in VISUAL_MODEL_CHAIN:
        for attempt in range(3):
            try:
                resp = req.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": max_tokens
                    },
                    timeout=timeout
                )
                if resp.status_code == 429:
                    wait = (attempt + 1) * 3
                    print(f"    {model} 429限流,等待{wait}秒重试...", flush=True)
                    time.sleep(wait)
                    continue
                data = resp.json()
                if 'choices' in data:
                    return data['choices'][0]['message']['content']
                err = data.get('error', {})
                if err:
                    err_msg = err.get('message', str(err))
                    print(f"    {model} 错误(第{attempt+1}次): {err_msg[:100]}", flush=True)
                    if err.get('code') == '1305':  # 并发过大,切换模型而非重试当前模型
                        break
            except Exception as e:
                print(f"    {model} 异常(第{attempt+1}次): {e}", flush=True)
            if attempt < 2:
                time.sleep(1)
        print(f"    {model} 3次尝试均失败,切换下一模型...", flush=True)
    print(f"    所有视觉模型均失败 ({VISUAL_MODEL_CHAIN})", flush=True)
    return ""


def analyze_frames_batch(frames: list, api_key: str) -> str:
    """使用视觉模型分析一批关键帧。按 VISUAL_MODEL_CHAIN 自动降级。"""
    content = []
    for f in frames:
        with open(f, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    content.append({
        "type": "text",
        "text": "这是视频的一部分关键帧。简要描述这部分画面展示了什么内容、场景、人物和动作。50字以内。"
    })
    return _call_visual_model(content, api_key, max_tokens=150)


def visual_analyze(all_frames: list, api_key: str, target_samples: int = 20, cache_b64: bool = False) -> str:
    if not all_frames:
        return ""
    n = min(target_samples, len(all_frames))
    step = len(all_frames) / n
    sampled = [all_frames[int(i * step)] for i in range(n)]
    batch_size = 5
    batches = [sampled[i:i + batch_size] for i in range(0, len(sampled), batch_size)]
    results = {}

    def _analyze(i: int, batch: list):
        try:
            desc = analyze_frames_batch(batch, api_key)
        except Exception as exc:
            print(f"    视觉批次{i + 1}异常: {exc}", flush=True)
            desc = ""
        return i, desc

    # 多个视觉批次并行调用(最多2路并发, 受GLM限流自动重试), 减少串行等待
    workers = min(2, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_analyze, i, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futs):
            i, desc = fut.result()
            results[i] = desc
            if desc:
                print(f"    视觉批次{i + 1}/{len(batches)}: {desc[:60]}...")
            time.sleep(0.3)
    return "\n".join([f"第{i + 1}部分: {results[i]}" for i in range(len(batches)) if results.get(i)])

def cache_frame_b64_list(all_frames: list, target_samples: int = 20) -> list:
    """将关键帧读取为 base64 列表并缓存,供后续追问时使用。
    
    Returns:
        [{"index": int, "b64": str}] 列表
    """
    n = min(target_samples, len(all_frames))
    step = len(all_frames) / n
    sampled = [all_frames[int(i * step)] for i in range(n)]
    result = []
    for i, fpath in enumerate(sampled):
        try:
            with open(fpath, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            result.append({"index": i, "b64": b64})
        except Exception:
            pass
    return result


# ============================================================================
# 语音识别(阿里云百炼 qwen3-asr-flash)
# ============================================================================

def extract_audio(video_path: str, output_path: str) -> bool:
    r = subprocess.run([
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-codec:a', 'libmp3lame',
        '-q:a', '4',
        '-y', output_path
    ], capture_output=True, text=True, timeout=300)
    return os.path.exists(output_path)


def get_audio_info(audio_path: str):
    try:
        r = subprocess.run([
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration,size',
            '-of', 'csv=p=0', audio_path
        ], capture_output=True, text=True, timeout=10)
        parts = r.stdout.strip().split(',')
        duration = float(parts[0]) if parts[0] else 0
        size = os.path.getsize(audio_path)
        if len(parts) > 1 and parts[1]:
            size = int(float(parts[1]))
        return duration, size
    except Exception:
        return 0, os.path.getsize(audio_path)


def _get_sensevoice_model():
    """懒加载 SenseVoiceSmall + FSMN-VAD 模型(进程内全局单例)。"""
    global _SENSEVOICE_MODEL
    if _SENSEVOICE_MODEL is not None:
        return _SENSEVOICE_MODEL
    import torch
    from funasr import AutoModel
    torch.set_num_threads(ASR_LOCAL_THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # 已设置过, 第二次调用会抛 RuntimeError
    torch.set_float32_matmul_precision("high")
    torch.backends.mkldnn.enabled = True
    requested = ASR_LOCAL_DEVICE if ASR_LOCAL_DEVICE in ("auto", "cuda", "cpu") else "auto"
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        print(f"    [本地ASR] CUDA检测失败,使用CPU: {exc}", flush=True)
    device = "cuda" if requested in ("auto", "cuda") and cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        print("    [本地ASR] 配置请求cuda但CUDA不可用,自动回退CPU", flush=True)
    global _SENSEVOICE_DEVICE
    _SENSEVOICE_DEVICE = device
    print(f"    [本地ASR] 首次加载 {ASR_LOCAL_MODEL} (含 VAD {ASR_LOCAL_VAD_MODEL}, device={device})...", flush=True)
    _SENSEVOICE_MODEL = AutoModel(
        model=ASR_LOCAL_MODEL,
        vad_model=ASR_LOCAL_VAD_MODEL,
        vad_kwargs={"max_single_segment_time": ASR_LOCAL_MAX_SEG_MS},
        disable_update=True,
        device=device,
        disable_pbar=True,
    )
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            print(f"    [本地ASR] CUDA设备: {props.name} | {props.total_memory / 1024**3:.2f}GB", flush=True)
        except Exception:
            pass
    return _SENSEVOICE_MODEL


def _do_local_transcribe_sensevoice(audio_path: str) -> tuple:
    """本地 SenseVoiceSmall + FSMN-VAD (funasr) 整段识别。

    返回 (text, status):
      status="ok"             — 识别成功 (text 可能为空字符串)
      status="no_funasr"      — venv 未装 funasr, 调用方应降级到云端
      status="error"          — 真实异常, text 含错误信息
    """
    try:
        import torch
        from funasr import AutoModel
    except ImportError as e:
        return ("", f"no_funasr:{e.__class__.__name__}")

    wav_path = audio_path
    if not audio_path.lower().endswith('.wav'):
        wav_path = audio_path.rsplit('.', 1)[0] + '_local16k.wav'
        r = subprocess.run([
            'ffmpeg', '-i', audio_path,
            '-ar', '16000', '-ac', '1',
            '-y', wav_path
        ], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(wav_path):
            return (f"本地ASR音频预处理失败: {r.stderr[:200]}", "error")

    try:
        res = _get_sensevoice_model().generate(
            input=wav_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        raw = "".join(
            r.get("text", "") for r in res
            if isinstance(r, dict) and r.get("text")
        )
        # 去掉 SenseVoice 输出的 <|zh|><|HAPPY|><|BGM|><|withitn|> 等元数据标签
        text = re.sub(r'<\|[^|]+\|>', '', raw).strip()
        return (text, "ok")
    except Exception as e:
        return (f"本地SenseVoice错误: {e}", "error")
    finally:
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


def _split_audio_chunks(audio_path: str, chunk_dir: str, chunk_sec: int = 30) -> list:
    """将整段音频切为固定秒数的16k单声道wav分块(用于ASR时间定位与广告复用)。"""
    os.makedirs(chunk_dir, exist_ok=True)
    for old in glob.glob(f"{chunk_dir}/chunk_*.wav"):
        try:
            os.remove(old)
        except OSError:
            pass
    r = subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', audio_path,
        '-ar', '16000', '-ac', '1', '-f', 'segment',
        '-segment_time', str(chunk_sec), '-c:a', 'pcm_s16le',
        f'{chunk_dir}/chunk_%05d.wav'
    ], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"音频分块失败: {r.stderr[-300:]}")
    paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.wav"))
    return [
        {"path": p, "start": i * chunk_sec, "end": (i + 1) * chunk_sec}
        for i, p in enumerate(paths)
    ]


def _transcribe_chunks_batch(chunks: list) -> list:
    """一次批量调用SenseVoice识别所有分块, 按key对齐回填text, 返回原chunk列表。"""
    if not chunks:
        return chunks
    res = _get_sensevoice_model().generate(
        input=[c["path"] for c in chunks],
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )
    def _stem(p):
        return os.path.splitext(os.path.basename(str(p)))[0]

    by_key = {}
    for r in res:
        if isinstance(r, dict) and r.get("key"):
            by_key[_stem(r["key"])] = re.sub(
                r'<\|[^|]+\|>', '', str(r.get("text", ""))).strip()
    for c in chunks:
        c["text"] = by_key.get(_stem(c["path"]), "")
    return chunks


def transcribe_audio_chunked(audio_path: str, video_dir: str, notify_callback=None) -> tuple:
    """分块本地ASR: 切30s分块 → 一次批量识别 → 返回 (chunks, 全文)。

    chunks 形如 [{start, end, path, text}], 带时间定位, 供广告检测直接复用
    (无需重新切音频/重新ASR)。失败时回退整段识别, 返回 ([], 全文)。
    """
    try:
        import torch
        from funasr import AutoModel
    except ImportError as e:
        print(f"    [本地ASR] funasr 不可用: {e.__class__.__name__}, 回退整段识别", flush=True)
        return [], transcribe_audio(audio_path, notify_callback)

    t0 = time.time()
    chunk_dir = os.path.join(video_dir, "asr_chunks")
    try:
        chunks = _split_audio_chunks(audio_path, chunk_dir)
        print(f"    [本地ASR] 切分 {len(chunks)} 个分块, 批量识别中 (device={_SENSEVOICE_DEVICE or ASR_LOCAL_DEVICE}, cpu_threads={ASR_LOCAL_THREADS})...", flush=True)
        if notify_callback:
            notify_callback(f"🎧 本地ASR (SenseVoiceSmall, {len(chunks)}分块) 推理中...")
        chunks = _transcribe_chunks_batch(chunks)
    except Exception as e:
        print(f"    [本地ASR] 分块识别失败, 回退整段识别: {e}", flush=True)
        return [], transcribe_audio(audio_path, notify_callback)

    text = "".join(c.get("text", "") for c in chunks).strip()
    dt = time.time() - t0
    duration, _ = get_audio_info(audio_path)
    rtf = (dt / duration) if duration else 0
    print(f"    [本地ASR] 完成 {dt:.1f}s ({rtf:.2f}x 实时), {len(chunks)}块 {len(text)}字", flush=True)
    return chunks, text


def transcribe_local(audio_path: str, notify=None) -> str:
    try:
        from faster_whisper import WhisperModel
        if notify:
            notify("🐌 本地SenseVoice不可用,切换本地Whisper(medium),处理速度较慢请耐心等待...")
        print(f"    本地Whisper(medium)识别中...", flush=True)
        wav_path = audio_path
        if not audio_path.endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '_local.wav'
            subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-ar', '16000', '-ac', '1',
                '-y', wav_path
            ], capture_output=True, text=True, timeout=60)
        model = WhisperModel("medium", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(wav_path, language="zh")
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        del model
        import gc; gc.collect()
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)
        return text if text else "(本地语音识别无结果)"
    except Exception as e:
        return f"本地语音识别失败: {e}"


def transcribe_audio(audio_path: str, notify_callback=None) -> str:
    duration, file_size = get_audio_info(audio_path)
    print(f"    音频: {duration:.0f}秒, {file_size/1024:.0f}KB", flush=True)

    # 唯一 ASR 路径: 本地 SenseVoice 推理 (在线 ASR 已全部过期下线)
    print(f"    [本地ASR] SenseVoiceSmall 推理中 (threads={ASR_LOCAL_THREADS}, max_seg={ASR_LOCAL_MAX_SEG_MS}ms)...", flush=True)
    if notify_callback:
        notify_callback("🎧 本地ASR (SenseVoiceSmall) 推理中...")
    t0 = time.time()
    text, status = _do_local_transcribe_sensevoice(audio_path)
    dt = time.time() - t0
    if status == "ok" and text:
        rtf = (dt / duration) if duration else 0
        print(f"    [本地ASR] 完成 {dt:.1f}s ({rtf:.2f}x 实时): {text[:200]}{'...' if len(text)>200 else ''}", flush=True)
        return text
    if status == "no_funasr":
        print(f"    [本地ASR] funasr 不可用: {text}", flush=True)
        if notify_callback:
            notify_callback("⚠️ 本地ASR未就绪 (funasr未安装)")
    else:
        print(f"    [本地ASR] 失败: {text[:200]}", flush=True)
        if notify_callback:
            notify_callback(f"⚠️ 本地ASR异常: {text[:80]}")
    # 本地兜底: faster-whisper (仍为本地推理)
    return transcribe_local(audio_path, notify_callback)


def final_summarize(visual_desc: str, asr_text: str, api_key: str) -> str:
    import requests as req
    parts = []
    if visual_desc:
        parts.append(f"【画面分析】\n{visual_desc}")
    # 清理ASR文本：过滤掉太短/无意义的识别结果
    asr_clean = asr_text or ""
    if asr_clean and not asr_clean.startswith("语音识别失败") and not asr_clean.startswith("本地语音识别失败"):
        # 滤除过短的无意义结果(纯英文单字等)
        # 同时统计中文和英文字符，支持英文ASR内容
        chinese_len = len([c for c in asr_clean if '\u4e00' <= c <= '\u9fff'])
        english_words = len([w for w in asr_clean.split() if w.strip() and any(c.isalpha() for c in w)])
        meaningful_len = max(chinese_len, english_words)
        if meaningful_len >= 4:  # 至少4个中文字或4个英文词才认为是有效内容
            parts.append(f"【语音识别文本】\n{asr_clean[:2000]}")
        else:
            print(f"    ASR内容过短(中文{chinese_len}字/英文{english_words}词),忽略: '{asr_clean[:50]}'", flush=True)
    # 内容质量检查：视觉和语音都严重不足时，不调用总结模型（防止幻觉）
    if not parts:
        return "无法生成视频总结：画面识别和语音识别均未能获取到足够信息"
    # 只有视觉描述没有语音，且视觉描述也很短（<20字），也可能是质量问题
    if len(parts) == 1:
        total_content = " ".join(parts)
        total_chinese = len([c for c in total_content if '\u4e00' <= c <= '\u9fff'])
        total_english = len([w for w in total_content.split() if w.strip() and any(c.isalpha() for c in w)])
        if max(total_chinese, total_english) < 10:
            print(f"    总结内容不足(中文{total_chinese}字/英文{total_english}词),放弃总结", flush=True)
            return "无法生成视频总结：分析内容不足，请稍后重试"
    combined = "\n\n".join(parts)
    prompt = """你是一个B站观众,刚看完一个视频,要用自然的语气总结内容。

要求:
1. 语气轻松自然,偶尔可以带点调侃,但不要过度娱乐化或戏谑
2. 把视频内容说清楚--讲了什么、展示了什么、核心话题是什么
3. 如果语音识别文本不准,主要靠画面判断
4. 上限1000字,但不要凑字数——说清楚就停,简洁比冗长更好,写成一段连贯的文字,不要分点,不要换行
5. 保持适度幽默感,但内容准确比花哨重要
6. 重要:如果输入信息不足以判断视频真实内容,请回复"信息不足,无法准确总结"而不是编造内容
7. 关键:你在回复中使用的所有素材(画面/语音/字幕等)都来自你自己的观察,不要提及"识别"、"语音识别"、"字幕机翻"、"识别错误"、"听不清"、"画面分析"等任何暗示技术过程或识别质量的说法。即使素材明显有误,也要直接给出你的判断,不要解释"识别错了"、"大概是空耳"、"可能是误识别"。把可能出错的片段当成你自己听到/看到的真实内容来叙述"""
    summary = _call_text_with_fallback(
        prompt, [{"role": "user", "content": combined}],
        max_tokens=1024, timeout=60, label="视频总结"
    )
    if summary:
        return summary
    print(f"    {MODEL_NAME} 主链路(GLM)与降级链(DeepSeek)均失败,标记为不可用", flush=True)
    return "__MODEL_UNAVAILABLE__"


# ============================================================================
# 动态(非视频)内容处理
# ============================================================================

def fetch_dynamic_detail(dynamic_id: str) -> dict:
    d = api_get(f"https://api.bilibili.com/x/polymer/web-dynamic/v1/detail?id={dynamic_id}")
    if d.get('code') == 0:
        return d.get('data', {}).get('item', {})
    print(f"  ⚠️  动态API错误: {d.get('message', '')}")
    return {}


def process_dynamic(uri: str, subject_id: str, root_id: str, title: str = '',
                    image_url: str = '', notify_callback=None) -> tuple:
    """
    处理非视频类型的@(动态/专栏等)。

    Returns:
        (summary, comment_type) 元组
    """
    import requests as req

    dynamic_id = ''
    opus_match = re.search(r'opus/(\d+)', uri)
    read_match = re.search(r'read/(cv\d+)', uri)

    text_content = title or ''
    image_urls = []
    comment_type = 11  # 默认动态评论(由business_id传入,这里设默认值)

    if read_match:
        comment_type = 12  # 专栏评论

    if opus_match:
        dynamic_id = opus_match.group(1)
        print(f"  动态ID: {dynamic_id}", flush=True)
        detail = fetch_dynamic_detail(dynamic_id)

        if detail:
            modules = detail.get('modules', {})

            # 文字内容
            module_dynamic = modules.get('module_dynamic', {})
            desc = module_dynamic.get('desc', {})
            if desc:
                text_content = desc.get('text', '') or text_content

            # 图片
            module_dynamic_major = module_dynamic.get('major', {})
            if module_dynamic_major:
                opus = module_dynamic_major.get('opus', {})
                if opus:
                    pics = opus.get('pics', [])
                    for pic in pics:
                        url = pic.get('url', '')
                        if url:
                            url = url.replace('http://', 'https://')
                            image_urls.append(url)

                archive = module_dynamic_major.get('archive', {})
                if archive:
                    aid = archive.get('aid', '')
                    if aid:
                        text_content += f"\n[转发了视频: {archive.get('title', '')}]"

            # 动态类型
            dyn_type = detail.get('basic', {}).get('comment_type', 0)
            if dyn_type:
                comment_type = dyn_type

    if not text_content and not image_urls:
        if image_url:
            image_url = image_url.replace('http://', 'https://')
            image_urls = [image_url]
        if not text_content and not image_urls:
            print(f"  动态内容为空,跳过", flush=True)
            return "动态内容为空,无法总结", comment_type

    print(f"  动态文字: {text_content[:100]}{'...' if len(text_content)>100 else ''}", flush=True)
    print(f"  动态图片: {len(image_urls)}张", flush=True)

    # 图片分析
    image_desc = ''
    if image_urls:
        print(f"  GLM-4V 分析动态图片({len(image_urls)}张)...", flush=True)
        descriptions = []
        for i, url in enumerate(image_urls[:5]):
            try:
                img_resp = req.get(url, timeout=15,
                    proxies={'http': f'http://{_proxy_host}:{_proxy_port}', 'https': f'http://{_proxy_host}:{_proxy_port}'})
                if img_resp.status_code == 200:
                    img_b64 = base64.b64encode(img_resp.content).decode()
                    content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": "简要描述这张图片的内容,30字以内。"}
                    ]
                    desc = _call_visual_model(content, ZHIPU_API_KEY, max_tokens=80)
                    if desc:
                        descriptions.append(f"图片{i+1}: {desc}")
                        print(f"    图片{i+1}: {desc[:60]}", flush=True)
            except Exception as e:
                print(f"    图片{i+1}分析失败: {e}", flush=True)
            time.sleep(0.5)
        image_desc = "\n".join(descriptions)

    # GLM 总结
    print(f"  {MODEL_NAME} 总结动态...", flush=True)
    parts = []
    if text_content:
        parts.append(f"【动态文字内容】\n{text_content[:2000]}")
    if image_desc:
        parts.append(f"【图片内容】\n{image_desc}")

    if not parts:
        return "动态内容为空", comment_type

    combined = "\n\n".join(parts)

    prompt = """你是一个B站用户,看到了一条动态内容。请用自然的语气总结这条动态讲了什么。
要求:
1. 语气轻松自然,偶尔可以带点调侃,但不要过度娱乐化
2. 把动态内容说清楚--发了什么、核心话题是什么
3. 上限1000字,但不要凑字数——说清楚就停,简洁比冗长更好,写成一段连贯的文字,不要分点,不要换行
4. 关键:你在回复中使用的所有素材(图文/语音等)都来自你自己的观察,不要提及"识别"、"语音识别"、"识别错误"、"画面分析"等任何暗示技术过程或识别质量的说法"""

    summary = _call_text_with_fallback(
        prompt, [{"role": "user", "content": combined}],
        max_tokens=500, timeout=60, label="动态总结"
    )
    if not summary:
        summary = "__MODEL_UNAVAILABLE__"

    return summary, comment_type


# ============================================================================
# 视频处理: 下载+截帧+语音识别+总结
# ============================================================================

def detect_embedded_ads(bv: str, video_path: str, fallback_duration: int,
                         notify_callback=None, all_frames=None, asr_chunks=None) -> dict:
    """在已下载的本地视频上执行广告识别；失败不影响原总结流程。

    若传入 all_frames(阶段2关键帧) 与 asr_chunks(阶段2分块ASR文本),
    则广告检测直接复用它们(只做视觉窗口判定+LLM分组/黑名单),
    不再重新ffmpeg切帧/切音频, 也不重新跑ASR。
    """
    empty = {
        "bv": bv, "has_ad": False, "reply_prefix": "",
        "ads": [], "detected_at": time.time(), "error": ""
    }
    if not AD_DETECTION_ENABLED:
        return empty
    try:
        import ad_detector
        settings, monitor = ad_detector.load_settings()
        # Reuse the SenseVoice model loaded by summary ASR to avoid double memory.
        if _SENSEVOICE_MODEL is not None and ad_detector.SHARED_ASR is None:
            runner = ad_detector.SenseVoiceRunner(settings)
            runner.model = _SENSEVOICE_MODEL
            ad_detector.SHARED_ASR = runner
        try:
            duration = ad_detector.media_duration(Path(video_path))
        except Exception:
            duration = float(fallback_duration)
        if all_frames:
            # 复用阶段2产物: 关键帧(2s间隔)按广告视觉间隔抽稀 + 分块ASR文本
            ad_interval = ad_detector.visual_interval(duration, settings)
            frame_step = max(1, round(ad_interval / 2.0))
            ad_frames = [Path(f) for f in all_frames[::frame_step]]  # ad_detector 需要 Path 对象
            ad_ts = [min(duration, i * 2.0 * frame_step) for i in range(len(ad_frames))]
            print(f"  广告检测复用: {len(all_frames)}帧→抽{len(ad_frames)}帧({int(ad_interval)}s间隔) + {len(asr_chunks)}块ASR", flush=True)
            result = ad_detector.detect_with_artifacts(
                bv, ad_frames, ad_ts, asr_chunks, duration,
                settings, monitor, Path(AD_RESULTS_DIR)
            )
        else:
            result = ad_detector.detect_local_video(
                bv, Path(video_path), duration, settings, monitor,
                Path(AD_RESULTS_DIR)
            )
        if result.get("has_ad") and notify_callback:
            ads = result.get("ads", [])
            lines = [f"⚠️ 视频识别到 {len(ads)} 段广告", f"BV: {bv}"]
            for x in ads:
                lines.append(
                    f"品牌: {x.get('brand') or '未知'} | "
                    f"{x.get('start_mmss')}-{x.get('end_mmss')} | "
                    f"置信度 {float(x.get('confidence', 0)):.2f}"
                )
            notify_callback("\n".join(lines))
        return result
    except Exception as exc:
        print(f"  ⚠️ 广告识别失败(不影响总结): {exc}", flush=True)
        if notify_callback:
            notify_callback(f"⚠️ 广告识别失败(不影响总结)\nBV: {bv}\n错误: {exc}")
        empty["error"] = str(exc)
        return empty


def _transcribe_audio_worker(audio_path: str, video_dir: str, audio_ok: bool, notify_callback=None) -> tuple:
    """并行阶段2的ASR任务: 分块识别, 返回 (chunks, text), 供总结与广告检测共用。"""
    if not audio_ok:
        print("  音频提取失败,跳过ASR", flush=True)
        return [], ""
    return transcribe_audio_chunked(audio_path, video_dir, notify_callback)

def process_video(bv: str, notify_callback=None):
    """
    对一个B站视频执行完整的分析流程。

    Returns:
        (duration_str, summary) 元组
    """
    video_dir   = f"{WORK_DIR}/video_{bv}"
    video_path  = f"{video_dir}/video.mp4"
    frames_dir  = f"{video_dir}/frames"
    audio_path  = f"{video_dir}/audio.mp3"

    try:
        os.makedirs(video_dir, exist_ok=True)


        duration = get_video_duration(bv)
        duration_str = f"{duration // 60}分{duration % 60}秒" if duration else "未知"

        if duration and duration > 6000:
            print(f"  视频过长({duration_str}),跳过识别与总结", flush=True)
            return duration_str, "视频过长,暂未开放长视频总结"

        print(f"  下载视频({duration_str})...", flush=True)
        if not download_video(bv, video_path):
            return duration_str, "视频下载失败"

        # 并行阶段1: 切帧 + 提取音频 (两者互不依赖, 同时进行)
        print(f"  并行截取关键帧 + 提取音频...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_frames = ex.submit(extract_frames, video_path, frames_dir, 2)
            f_audio = ex.submit(extract_audio, video_path, audio_path)
            all_frames = f_frames.result()
            audio_ok = f_audio.result()
        print(f"  获得 {len(all_frames)} 帧 | 音频提取{'成功' if audio_ok else '失败'}")

        # 并行阶段2: 视觉AI(云端GLM) 与 本地ASR(CPU) 同时进行 —— 3次AI结果中的前2次
        # ASR改为分块批量识别: 结果同时供总结与内嵌广告检测复用, 避免广告侧重复切音频/重复ASR
        print(f"  并行: GLM视觉分析 + 本地ASR(分块, 供广告复用)...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_visual = ex.submit(visual_analyze, all_frames, ZHIPU_API_KEY)
            f_asr = ex.submit(_transcribe_audio_worker, audio_path, video_dir, audio_ok, notify_callback)
            f_cache = ex.submit(cache_frame_b64_list, all_frames)  # 缓存关键帧base64, 供后续对话追问
            visual_desc = f_visual.result()
            asr_chunks, asr_text = f_asr.result()
            frame_cache = f_cache.result()
        # 持久化分块ASR(项目内, 供广告检测与排查复用); 剥离已删除的wav路径字段
        if asr_chunks:
            transcript_path = f"{video_dir}/asr_transcript.json"
            slim = [{k: c[k] for k in ("start", "end", "text") if k in c} for c in asr_chunks]
            with open(transcript_path, 'w') as f:
                json.dump(slim, f, ensure_ascii=False, indent=2)
            print(f"  已保存 {len(slim)} 块ASR时间戳 → {transcript_path}")
        if asr_text and not asr_text.startswith("语音识别失败") and not asr_text.startswith("API错误") and not asr_text.startswith("本地语音识别失败"):
            print(f"  语音识别完成,{len(asr_text)}字")
        elif asr_text:
            print(f"  ⚠️  {asr_text[:120]}")

        # 并行阶段3: LLM整合(第3次AI结果) 与 内嵌广告识别(复用阶段2产物) 同时进行
        print(f"  并行: GLM综合总结 + 内嵌广告识别(复用帧/ASR)...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_summary = ex.submit(final_summarize, visual_desc, asr_text, ZHIPU_API_KEY)
            f_ad = ex.submit(detect_embedded_ads, bv, video_path, duration, notify_callback,
                             all_frames=all_frames, asr_chunks=asr_chunks)
            summary = f_summary.result()
            ad_result = f_ad.result()

        # 保存关键帧缓存到文件,供后续追问使用
        if frame_cache:
            cache_path = f"{WORK_DIR}/frames_cache_{bv}.json"
            with open(cache_path, 'w') as f:
                json.dump(frame_cache, f)
            print(f"  缓存了 {len(frame_cache)} 帧base64 → {cache_path}")

        return duration_str, summary

    finally:
        # 不再立即删除视频目录,保留帧缓存用于后续追问
        # 只删除视频文件和音频文件(体积大),保留frames
        for f in [f"{video_dir}/video.mp4", f"{video_dir}/audio.mp3"]:
            if os.path.exists(f):
                os.remove(f)
        # 清理帧目录中的jpg文件(已缓存为base64)
        frames_jpg_dir = f"{video_dir}/frames"
        if os.path.exists(frames_jpg_dir):
            subprocess.run(['rm', '-rf', frames_jpg_dir], timeout=10)
        # 清理ASR分块wav(已持久化为asr_transcript.json, 无需保留体积大的wav)
        asr_chunk_dir = f"{video_dir}/asr_chunks"
        if os.path.exists(asr_chunk_dir):
            subprocess.run(['rm', '-rf', asr_chunk_dir], timeout=10)
        # 清理遗留chunk目录
        for d in glob.glob(f"{video_dir}/chunks_*"):
            subprocess.run(['rm', '-rf', d], timeout=10)


# ============================================================================
# 🔥 评论区对话处理 (新增核心逻辑)
# ============================================================================

def handle_chat_message(item: dict, bv: str, comment_type: int,
                        notify_callback=None) -> bool:
    """
    处理一条评论区@消息:判断意图(总结/聊天),生成回复。

    Args:
        item: @消息的完整item dict
        bv: 视频BV号(如果是动态则为空)
        comment_type: 评论区类型(来自business_id)
        notify_callback: QQ通知回调

    Returns:
        True=已成功处理, False=处理失败
    """
    source_id = str(item.get('source_id', ''))
    subject_id = str(item.get('subject_id', ''))
    root_id = str(item.get('root_id', ''))
    target_id = str(item.get('target_id', ''))
    source_content = item.get('source_content', '')
    title = item.get('title', '')
    user_nickname = ''  # 从外层获取

    # 1. 确定要回复的目标评论ID
    # source_id 就是这条@消息所在的评论ID(即需要回复的评论)
    parent_id = source_id  # 回复到这条评论

    # 如果 root_id 为 0, 说明这是一条主评论级别的@
    # 根评论就是 source_id 本身
    if root_id and root_id != '0':
        top_root_id = root_id  # 有父评论,使用父评论作为根
    else:
        top_root_id = source_id  # 这是一条主评论,@就在主评论上

    print(f"  评论类型: {'主评论' if (not root_id or root_id=='0') else '子评论'}")
    print(f"  root_id={top_root_id}, parent_id={parent_id}")
    print(f"  原始内容: {source_content[:100]}")

    # 2. 提取@Bot之后的有效文字
    user_text = extract_message_after_at(source_content, BOT_NAME)
    print(f"  有效文字: '{user_text}'" if user_text else "  (纯@,无文字)")

    # 3. 加载视频总结缓存
    summaries = load_summaries()
    video_summary_data = summaries.get(bv, {}) if bv else {}
    existing_summary_text = video_summary_data.get('summary', '')
    has_existing_summary = bool(existing_summary_text and existing_summary_text != '__MODEL_UNAVAILABLE__')

    # 4. 获取对话上下文(从评论区抓取)
    dialog_context = []
    if top_root_id and comment_type:
        print(f"  获取评论区上下文(type={comment_type}, oid={subject_id}, root={top_root_id})...")
        try:
            sub_replies = fetch_comment_thread(subject_id, top_root_id, comment_type, max_pages=5)
            print(f"  获取到 {len(sub_replies)} 条子评论")
            dialog_context = extract_dialog_context(sub_replies, BOT_MID)
            print(f"  对话上下文: {len(dialog_context)} 条有效消息")
            for m in dialog_context:
                print(f"    [{m['role']}] {m['content'][:60]}...")
        except Exception as e:
            print(f"  ⚠️  获取评论区上下文失败: {e}")

    # ----------------------------------------------------------------
    # 新逻辑 (v5.4): 首次@必做视频分析, 再根据意图决定回复方式
    # ----------------------------------------------------------------
    #
    # 核心改动:
    #   1. 首次@(无缓存) → 无论用户写了什么, 都先下载视频做分析
    #   2. 已有缓存 → 直接复用, 不重复下载
    #   3. 意图分类 → 决定"回复什么", 而非"是否下载"
    #      - summary → 回复总结
    #      - chat/video_chat → 结合视频内容对话 (不再空对空聊天)
    #   4. 长视频(>6000s) / 官方内容 / 无BV号 仍走原有逻辑

    video_summary_for_reply = existing_summary_text if has_existing_summary else ''
    ad_prefix = video_summary_data.get('ad_prefix', '') if bv else ''

    # 5. 首次@: 确保视频已分析 (有BV号且无缓存时必做)
    if bv and not has_existing_summary:
        print(f"  🎬 首次@该视频, 开始视频分析...")
        duration_str, summary = process_video(bv, notify_callback)
        ad_result = load_ad_result(bv)
        ad_prefix = ad_result.get('reply_prefix', '')
        print(f"  分析结果: {summary[:100]}...")

        if summary == "__MODEL_UNAVAILABLE__":
            video_summary_for_reply = ""
        elif summary.startswith("视频下载失败"):
            reply_text = summary
            # 直接跳到发送回复
            success, resp = send_reply_with_audit_check(subject_id, top_root_id, parent_id, reply_text, comment_type, notify_callback=notify_qq)
            if success:
                if notify_callback:
                    notify_callback(f"✅ B站回复完成\n内容: {reply_text[:300]}")
                return True, top_root_id
            else:
                print(f"  ❌ 回复失败: {resp.get('message', '')}")
                return False, top_root_id
        else:
            save_summary(bv, summary, duration_str, ad_result)
            video_summary_for_reply = summary
    elif bv and has_existing_summary:
        print(f"  📋 视频已有缓存, 直接复用总结")

    # 6. 意图分类
    if not user_text or not user_text.strip():
        intent = "summary"
        print(f"  意图: 总结 (纯@)")
    else:
        intent = classify_user_intent(user_text, has_existing_summary or bool(video_summary_for_reply), ZHIPU_API_KEY)
        print(f"  意图: {intent}")

    # 7. 检查是否已经在这个 root 评论下发过总结
    #    核心规则: Bot在此线程回复过 + 已有视频缓存 → 视为已发过总结,不再重复
    #    (不用内容相似度判断,因为B站可能拦截/替换回复内容)
    already_posted_summary_in_thread = False
    if dialog_context and video_summary_for_reply:
        for msg in dialog_context:
            if msg.get('role') == 'assistant':
                already_posted_summary_in_thread = True
                break

    # 如果已经发过总结但意图仍是 summary,强制转为 chat
    if already_posted_summary_in_thread and intent == "summary":
        print(f"  ⚠️ 该线程下已发过总结,意图从 summary → chat")
        intent = "chat"
        if not user_text or not user_text.strip():
            user_text = "又来啦"

    # 8. 根据意图决定回复内容
    if intent == "summary":
        # --- 回复总结 ---
        if video_summary_for_reply:
            # 只在总结回复前加广告空降提示；追问/chat保持原交互样式。
            reply_text = f"{ad_prefix}{video_summary_for_reply}" if ad_prefix else video_summary_for_reply
        else:
            reply_text = "已经帮您总结过了,你可以在之前的评论中查看。"
            print(f"  ⚠️ 视频总结为空,回复提示")

    else:  # intent == "chat" or intent == "video_chat"
        # --- 结合视频内容对话 ---
        if not user_text or not user_text.strip():
            user_text = "你好"

        # video_chat: 让视觉模型重新查看关键帧
        visual_context = ""
        if intent == "video_chat" and bv:
            frame_cache = load_frame_cache(bv)
            if frame_cache:
                print(f"  GLM-4.6V 重新查看关键帧({len(frame_cache)}帧)...")
                visual_context = visual_query_frames(frame_cache, user_text, ZHIPU_API_KEY)
                if visual_context:
                    print(f"  视觉追问结果: {visual_context[:100]}...")
                else:
                    print(f"  ⚠️  视觉追问无结果")
            else:
                print(f"  ⚠️  无关键帧缓存,无法重新查看视频")

        print(f"  生成回复 (结合视频内容)...")
        reply_text = generate_chat_reply(
            context=dialog_context,
            user_message=user_text,
            video_summary=video_summary_for_reply,
            video_title=title,
            api_key=ZHIPU_API_KEY,
            visual_context=visual_context
        )
        print(f"  回复: {reply_text[:100]}...")

    # 7. 发送回复
    success, resp = send_reply_with_audit_check(subject_id, top_root_id, parent_id, reply_text, comment_type, notify_callback=notify_qq)
    if success:
        print(f"  ✅ 回复成功!")
        if notify_callback:
            snippet = reply_text[:300]
            notify_callback(f"✅ B站回复完成\n内容: {snippet}")
        return True, top_root_id  # 返回root_id供调用方注册线程
    else:
        print(f"  ❌ 回复失败: {resp.get('message', '')} | 完整响应: {json.dumps(resp, ensure_ascii=False)[:300]}")
        if notify_callback:
            notify_callback(f"❌ B站回复失败: {resp.get('message', '')}")
        return False, top_root_id


# ============================================================================
# 主处理:处理所有新的@消息
# ============================================================================

def process_new_at_messages():
    """
    获取并处理所有新的@消息(主评论级别)。

    处理逻辑:
    1. 对每条新@消息,判断类型:
       a. 官番/电影等不可下载内容 → 回复提示
       b. 动态内容 → process_dynamic 处理
       c. 视频内容 → 进入评论区对话处理(handle_chat_message)
    """
    replied = load_state()
    items = fetch_at_messages()
    new_count = 0

    for outer_item in items:
        item = outer_item.get('item', {})
        source_id = str(item.get('source_id', ''))

        # 跳过已处理的
        if not source_id or source_id in replied:
            continue

        # 解析消息信息
        user       = outer_item.get('user', {}).get('nickname', '?')
        uri        = item.get('uri', '')
        bv_match   = re.search(r'BV[\w]+', uri)
        bv         = bv_match.group(0) if bv_match else ''
        subject_id = str(item.get('subject_id', ''))
        title      = item.get('title', '')
        image_url  = item.get('image', '')
        business_id = item.get('business_id', 1)  # 1=视频, 11=动态
        is_dynamic = (not bv)

        print(f"\n{'=' * 50}")
        if is_dynamic:
            print(f"[{time.strftime('%H:%M:%S')}] 新@消息(动态): {user} @ {uri[:60]}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 新@消息(视频): {user} @ {bv}")
        print(f"{'=' * 50}")

        # QQ通知
        if is_dynamic:
            notify_qq(f"🔔 B站新@消息(动态)\n用户: {user}\n正在分析动态内容...")
        else:
            notify_qq(f"🔔 B站新@消息\n用户: {user}\n视频: {bv}\n正在处理...")

        new_count += 1

        # --- 分支1: 视频内容 ---
        if bv:
            # 检测是否为官方/番剧等无法访问的内容
            print(f"  检查视频版权状态...")
            is_official, official_reason = is_official_content(bv)
            if is_official:
                print(f"  ⚠️  官方内容: {official_reason}")
                root_id = str(item.get('root_id', ''))
                if not root_id or root_id == '0':
                    root_id = source_id

                reply_text = "内容由于官方机制,暂时无法访问,无法分析总结。"
                success, resp = send_reply_with_audit_check(subject_id, root_id, source_id, reply_text, business_id, notify_callback=notify_qq)
                if success:
                    print(f"  ✅ 已回复官方内容提示")
                    notify_qq(f"⚠️ B站回复(官方内容)\n用户: {user}\n视频: {bv}\n原因: {official_reason}")
                else:
                    print(f"  ❌ 回复失败: {resp.get('message', '')}")
                save_state(source_id)
                time.sleep(3)
                continue

            # 正常视频:走评论区对话处理
            handle_chat_message(item, bv, business_id, notify_callback=notify_qq)
            save_state(source_id)
            time.sleep(3)
            continue

        # --- 分支2: 动态内容 ---
        if is_dynamic:
            root_id = str(item.get('root_id', ''))
            if not root_id or root_id == '0':
                root_id = source_id

            summary, comment_type = process_dynamic(
                uri, subject_id, root_id,
                title=title, image_url=image_url,
                notify_callback=notify_qq
            )
            print(f"\n  动态总结: {summary[:100]}...")

            if summary == "__MODEL_UNAVAILABLE__":
                reply_text = "当前模型无法访问到,请稍后重试"
                success, resp = send_reply_with_audit_check(subject_id, root_id, source_id, reply_text, comment_type, notify_callback=notify_qq)
                if success:
                    print(f"  ⚠️  模型不可用,已回复提示")
                    notify_qq(f"⚠️ B站回复(模型不可用)\n用户: {user}\n已回复提示")
                save_state(source_id)
                time.sleep(3)
                continue

            reply_text = summary

            success, resp = send_reply_with_audit_check(subject_id, root_id, source_id, reply_text, comment_type, notify_callback=notify_qq)
            if success:
                print(f"  ✅ 回复成功!")
                notify_qq(f"✅ B站动态回复完成\n用户: {user}\n\n{summary[:300]}")
            else:
                print(f"  ❌ 回复失败: {resp.get('message', '')}")

            save_state(source_id)
            time.sleep(3)
            continue

    if new_count == 0:
        print(f"[{time.strftime('%H:%M:%S')}] 本轮无新@消息")
    else:
        print(f"\n本轮@消息处理完毕: {new_count} 条")
    return new_count


def process_new_reply_messages():
    """
    获取并处理所有新的评论区回复通知(子评论中@Bot)。

    msgfeed/reply 推送的是"有人回复了你的评论"的通知。
    我们只处理其中包含@Bot的消息,忽略普通回复。

    数据结构与 at 消息类似,包含:
    - source_content: 回复内容
    - target_reply_content: 被回复的Bot评论内容
    - at_details: @的用户列表
    - root_id, source_id, target_id, subject_id, business_id 等
    """
    replied = load_state()
    items = fetch_reply_messages()
    new_count = 0

    for outer_item in items:
        item = outer_item.get('item', {})
        source_id = str(item.get('source_id', ''))

        # 用 reply_ 前缀区分,避免与 at 的 source_id 冲突
        reply_source_key = f"reply_{source_id}"

        # 跳过已处理的
        if not source_id or source_id in replied or reply_source_key in replied:
            continue

        # 检查是否@了Bot (at_details 字段)
        at_details = item.get('at_details', [])
        has_at_bot = False
        if at_details:
            for at_user in at_details:
                if at_user.get('mid') == BOT_MID or str(at_user.get('mid')) == str(BOT_MID):
                    has_at_bot = True
                    break

        # 也检查文本中是否包含 @Bot
        source_content = item.get('source_content', '')
        if not has_at_bot and f'@{BOT_NAME}' not in source_content:
            continue

        user = outer_item.get('user', {}).get('nickname', '?')
        uri = item.get('uri', '')
        bv_match = re.search(r'BV[\w]+', uri)
        bv = bv_match.group(0) if bv_match else ''
        subject_id = str(item.get('subject_id', ''))
        title = item.get('title', '')
        business_id = item.get('business_id', 1)

        # 修正 item 字段名以兼容 handle_chat_message
        # msgfeed/reply 的 item 字段名与 msgfeed/at 略有不同
        # target_id 在 reply 中是被回复的Bot评论的 rpid,我们需要 reply 到 source_id
        item['source_content'] = source_content

        print(f"\n{'=' * 50}")
        print(f"[{time.strftime('%H:%M:%S')}] 新回复(子评论): {user}")
        if bv:
            print(f"  视频: {bv}")
        print(f"  内容: {source_content[:80]}")
        print(f"{'=' * 50}")

        notify_qq(f"💬 B站评论新回复\n用户: {user}\n内容: {source_content[:100]}")

        new_count += 1

        # 处理(复用 handle_chat_message)
        success, root_id_replied = handle_chat_message(
            item, bv, business_id, notify_callback=notify_qq
        )
        save_state(reply_source_key)
        time.sleep(3)

    if new_count == 0:
        # 不打印——太啰嗦,只在有消息时才输出
        pass
    else:
        print(f"\n本轮回复处理完毕: {new_count} 条")
    return new_count


# ============================================================================
# 快速检查:是否有新的@消息
# ============================================================================

def check_unread() -> dict:
    """检查未读消息数量。返回 {'at': int, 'reply': int} 或 None(失败)"""
    d = api_get("https://api.bilibili.com/x/msgfeed/unread")
    if d.get('code') != 0:
        return None
    data = d.get('data', {})
    return {'at': data.get('at', 0), 'reply': data.get('reply', 0)}


def dry_populate_unread():
    """启动时把当前 unread 列表全部塞入 state file, 防止历史未读触发雪崩回复。

    场景: 进程被 kill/重启后 state file 可能不完整, B站 unread 列表会持续返回
    历史消息, 若不预处理则会被当成新消息逐条回复。本函数在启动主循环前调用一次,
    把现有 unread 的 source_id 全部标记为已知, 之后只处理启动后真正新到的消息。

    安全保证: 只追加不覆盖, 不会丢失既有 state。已读但未清除接口的消息也只会被
    幂等追加(state 是 set 语义)。
    """
    try:
        at_items = fetch_at_messages()
        rp_items = fetch_reply_messages()
    except Exception as e:
        print(f"  ⚠️  启动 dry-populate 抓取 unread 失败: {e}, 跳过", flush=True)
        return

    before = len(load_state())
    new_ids = set()
    for o in at_items + rp_items:
        sid = str(o.get('item', {}).get('source_id', ''))
        if sid:
            new_ids.add(sid)

    if not new_ids:
        print(f"  ✓ 启动 dry-populate: unread 为空, 无需标记", flush=True)
        return

    # 只追加 state 里没有的, 避免重写整个文件
    existing = load_state()
    fresh = new_ids - existing
    if fresh:
        with open(STATE_FILE, 'a') as f:
            for sid in sorted(fresh):
                f.write(sid + '\n')
    after = len(load_state())
    print(f"  ✓ 启动 dry-populate: unread 共 {len(new_ids)} 条, 新增标记 {len(fresh)} 条, state: {before} → {after}", flush=True)


def fetch_at_messages() -> list:
    d = api_get("https://api.bilibili.com/x/msgfeed/at?build=0&mobi_app=web")
    if d.get('code') != 0:
        print(f"  ⚠️  @消息API错误: {d.get('message')}")
        return []
    return d.get('data', {}).get('items', [])


def fetch_reply_messages() -> list:
    """获取评论区回复通知(含子评论中的@)。
    
    msgfeed/reply 推送的是:别人回复了你评论的通知。
    如果对方在回复中@了Bot, at_details 字段会包含Bot的信息。
    """
    d = api_get("https://api.bilibili.com/x/msgfeed/reply?build=0&mobi_app=web")
    if d.get('code') != 0:
        print(f"  ⚠️  reply消息API错误: {d.get('message')}")
        return []
    return d.get('data', {}).get('items', [])


# ============================================================================
# 主入口 -- 常驻后台进程
# ============================================================================

def main():
    print("=" * 60, flush=True)
    print(f" B站@消息监控已启动 (v5.10.0: unread快速轮询 + 小时级列表兜底 + API退避 + 项目内持久化 + 内嵌广告识别 + GPU ASR)", flush=True)
    print(f" 广告识别: {'开启' if AD_DETECTION_ENABLED else '关闭'} | 黑名单: {len(load_blacklist_keywords())}个关键词", flush=True)
    print(f" 轮询间隔: {POLL_INTERVAL}秒 | @列表兜底: {AT_FALLBACK_INTERVAL}秒 | 回复列表兜底: {REPLY_FALLBACK_INTERVAL}秒", flush=True)
    print(f" 工作目录: {WORK_DIR}", flush=True)
    print(f" 状态文件: {STATE_FILE}", flush=True)
    print(f" 总结缓存: {SUMMARY_FILE}", flush=True)
    print(f" 已回复数: {len(load_state())}", flush=True)
    print(f" 视频总结数: {len(load_summaries())}", flush=True)
    print(f" ASR: 本地 SenseVoiceSmall (funasr), 在线ASR已下线", flush=True)
    print(f" Bot UID: {BOT_MID}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # 启动时不做 dry-populate：它会吞掉 unread 计数异常导致漏处理的消息。
    # 列表兜底从启动后一个周期开始，避免重启后集中补发历史消息。
    poll_count = 0
    HEARTBEAT_EVERY = 20
    SUCCESSIVE_ERRORS_MAX = 3
    successive_errors = 0
    next_at_fallback = time.monotonic() + AT_FALLBACK_INTERVAL
    next_reply_fallback = time.monotonic() + REPLY_FALLBACK_INTERVAL

    while True:
        try:
            # API退避期间不做高频重试，最多每60秒记录一次状态
            remaining_backoff = _API_BACKOFF_UNTIL - time.time()
            if remaining_backoff > 0:
                poll_count += 1
                if poll_count % HEARTBEAT_EVERY == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] 🛡️ API退避中，剩余{int(remaining_backoff)}秒", flush=True)
                time.sleep(min(60, remaining_backoff + 1))
                continue

            unread = check_unread()
            poll_count += 1

            if unread is None:
                successive_errors += 1
                if successive_errors <= SUCCESSIVE_ERRORS_MAX:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ unread接口异常 (连续{successive_errors}次)", flush=True)
                elif successive_errors == SUCCESSIVE_ERRORS_MAX + 1:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 持续异常,之后每20次只告警一次", flush=True)
            else:
                successive_errors = 0
                at_count = unread.get('at', 0)
                reply_count = unread.get('reply', 0)

                # unread 是15秒快速通道
                if at_count > 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 📬 unread检测到 {at_count} 条新@消息,开始处理...")
                    process_new_at_messages()
                    next_at_fallback = time.monotonic() + AT_FALLBACK_INTERVAL
                elif time.monotonic() >= next_at_fallback:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 unread.at=0，执行@列表小时级兜底...")
                    process_new_at_messages()
                    next_at_fallback = time.monotonic() + AT_FALLBACK_INTERVAL

                if reply_count > 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 💬 unread检测到 {reply_count} 条新回复,开始处理...")
                    process_new_reply_messages()
                    next_reply_fallback = time.monotonic() + REPLY_FALLBACK_INTERVAL
                elif time.monotonic() >= next_reply_fallback:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 unread.reply=0，执行回复列表小时级兜底...")
                    process_new_reply_messages()
                    next_reply_fallback = time.monotonic() + REPLY_FALLBACK_INTERVAL

                # 心跳日志
                if at_count == 0 and reply_count == 0:
                    if poll_count % HEARTBEAT_EVERY == 0:
                        next_at = max(0, int(next_at_fallback - time.monotonic()))
                        next_reply = max(0, int(next_reply_fallback - time.monotonic()))
                        print(f"[{time.strftime('%H:%M:%S')}] 💓 心跳 #{poll_count} | unread.at=0 reply=0 | @兜底{next_at}s 回复兜底{next_reply}s", flush=True)

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ 处理异常: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
