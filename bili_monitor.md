# bili_monitor.py 架构文档

> ⚠️ **AI Generated Project** — 本项目全部代码由 AI 在人工提示词引导下生成，未经人工审核。使用本项目造成的任何损失与作者无关。完整免责声明见文档末尾。
>
> 最后更新: 2026-06-18
> 版本: v5.5.1 (修复新版合集视频下载失败 + 格式链完善)

---

## 概述

B站@消息AI自动监控脚本，常驻后台运行。检测到 `@Bot` 后自动处理（首次@必做视频分析，后续根据意图回复总结或结合视频内容对话），并通过 QQ Bot 通知用户。

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1 | 2026-05-08 | 初版: 视频总结 + B站回复 + QQ通知 |
| v2 | 2026-05-09 | ASR多模型降级链 + 评论区对话 |
| v3 | 2026-05-10 | msgfeed统一轮询 + 子评论@处理 |
| v4 | 2026-05-12 | GLM-4.6V-Flash + 关键帧缓存 + 视频追问 |
| v5 | 2026-05-14 | 意图分类修复(转载/短消息误判) + 视觉模型降级链 + 统一 `_call_visual_model()` |
| v5.1 | 2026-05-27 | 修复英文ASR内容被丢弃的bug, `final_summarize()` 支持中英文内容检测 |
| v5.2 | 2026-06-01 | 代理自动检测:启动时探测7890端口,有则走代理无则直连,不再硬编码代理 |
| v5.3 | 2026-06-01 | 回复控制在150字内,去掉机器人前缀,解决B站审核折叠问题 |
| v5.4 | 2026-06-03 | **首次@必做视频分析**,不再因意图分类跳过下载; chat回复强制结合视频内容; 意图分类更严格(模糊消息优先判为summary) |
| v5.4.3 | 2026-06-14 | **修复视频下载失败**: B站对 `--add-header Cookie` 方式返回 412, 改用 `--cookies` 文件传递 Cookie |
| v5.5 | 2026-06-17 | **分P视频支持**: 自动检测分P数量, 逐P下载后用 ffmpeg concat 合并为单文件, 确保完整分析 |
| v5.5.1 | 2026-06-18 | **修复新版合集视频下载失败**: B站新版 anthlogy 格式视频只有分离音视频流, 格式链新增 `bestvideo+bestaudio` 回退, 解决 `30016+30216` 和 `best` 均不可用的问题 |

## ⚡ 快速启动/停止

```bash
# 首次使用: 复制配置模板并填写真实配置
cp config.example.json config.json
nano config.json   # 填写 SESSDATA, bili_jct, API Key 等

# 停止
kill $(pgrep -f bili_monitor.py)

# 启动 (一行搞定)
nohup python3 -u bili_monitor.py >> /tmp/bili_monitor.log 2>&1 &

# 确认启动成功 (⚠️ 等至少5分钟再看日志！)
tail -5 /tmp/bili_monitor.log
```

配置文件查找顺序 (优先级从高到低):
1. `--config /path/to/config.json` 命令行参数
2. `BILI_CONFIG` 环境变量
3. 脚本同目录下的 `config.json`

---

## ⚠️ 重要: 启动前必读

### 1. 代理: 启动时自动检测, 无需手动干预

脚本启动时会自动探测 `127.0.0.1:7890`:
- ✅ 端口可达 → 自动走代理 (日志显示 `🔄 代理已启用`)
- ✅ 端口不可达 → 自动走直连 (日志显示 `🔄 代理未运行,直连模式`)

**两种模式都能正常工作。** B站 API 不需要代理也能直接访问。

如果希望走代理(推荐,更稳定), 先启动 mihomo:
```bash
/etc/init.d/mihomo start   # 或: clashon
```

代理管理工具: [clash-for-linux-install](https://github.com/nelvko/clash-for-linux-install)
- 安装目录: `/root/clashctl/`
- 服务管理: `/etc/init.d/mihomo` (SysVinit)
- 常用命令: `clashon` / `clashoff` / `clashctl status`

### 2. 心跳: 5分钟一次, 不要慌

> **脚本启动后 5 分钟内不会有任何日志输出，这是正常的！**

- 脚本每 15 秒轮询一次 B站 API
- 只有检测到 `at>0` 或 `reply>0` 时才立即打印日志
- **没有新消息时, 心跳日志每 20 轮输出一次 = 20 × 15秒 = 5分钟**
- 启动后不要在几秒内去看日志, 至少等 **5 分钟** 才能看到心跳
- 看到心跳即证明脚本正常: `[23:26:19] 💓 心跳 #20 | unread.at=0 reply=0 | 正常`

### 3. ASR 密钥: 在 config.json 中配置

在 `config.json` 的 `dashscope.api_key` 字段填写阿里云百炼 API Key:
```json
{
    "dashscope": {
        "api_key": "sk-xxxxxxxx"
    }
}
```
启动日志应显示 `ASR密钥: 已配置`。如果显示 `❌ 未配置`, 说明密钥未填写。

> 兼容: 仍支持 `DASHSCOPE_API_KEY` 环境变量, 但配置文件中的值优先。

---

## 状态文件

| 文件 | 说明 |
|------|------|
| `/tmp/bili_replied_ids.txt` | 已处理的消息ID。`at` 消息存 `source_id`，`reply` 消息存 `reply_{source_id}` |
| `/tmp/bili_video_summaries.json` | 视频总结缓存。`{bv: {summary, duration, time}}` |
| `/tmp/bili_active_threads.json` | 遗留文件(v2), 当前不再写入, 仅保留读取用于迁移 |
| `/tmp/bili_monitor/frames_cache_{BV}.json` | **v4新增** — 视频关键帧base64缓存, 供后续追问使用 |

---

## 整体架构

```
main()  - 15秒循环
  │
  ├── check_unread()                       GET /x/msgfeed/unread
  │   └── 返回 {at, reply} 数量
  │
  ├── at > 0  → process_new_at_messages()  GET /x/msgfeed/at
  │   │
  │   └── 对每条@消息:
  │       ├── 视频(有BV号):
  │       │   ├── is_official_content()    检测番剧/电影/官方内容(仅靠 rights.download + tid)
  │       │   └── handle_chat_message()    v5.4: 首次@必分析视频 → 意图分类 → 决定回复方式
  │       │       ├── [首次@无缓存] → process_video() → 缓存总结+帧
  │       │       ├── [已有缓存]   → 直接复用
  │       │       ├── summary    → 回复视频总结
  │       │       ├── chat       → generate_chat_reply(结合视频内容)
  │       │       └── video_chat → load_frame_cache() → visual_query_frames() → generate_chat_reply(visual_context=...)
  │       │
  │       └── 动态(无BV号):
  │           └── process_dynamic()        分析动态图文 → 回复总结 (视觉模型降级链)
  │
  └── reply > 0 → process_new_reply_messages()  GET /x/msgfeed/reply
      │
      └── 对每条回复通知:
          ├── 检查 at_details 是否包含Bot UID
          └── handle_chat_message()
```

---

## 轮询机制

**唯一的轮询**: 每 15 秒调用 `GET /x/msgfeed/unread`，返回:

```json
{
  "at": 0,        // @消息未读数
  "reply": 0,     // 评论回复未读数
  "like": 14,     // (不使用)
  ...
}
```

- `at > 0` → 调 `GET /x/msgfeed/at` 获取@消息详情
- `reply > 0` → 调 `GET /x/msgfeed/reply` 获取回复通知详情
- 全部是 **B站官方 API**，无风控风险

---

## 函数索引

### 环境 & 配置

| 函数/变量 | 说明 |
|-----------|------|
| `_load_config()` | 从 config.json 加载配置(支持 --config 参数 / BILI_CONFIG 环境变量 / 同目录 config.json) |
| `SESSDATA` / `BILI_JCT` | B站登录Cookie, 从 config.json `bilibili` 节读取 |
| `BOT_MID` | Bot的B站UID, 从 config.json `bilibili.bot_mid` 读取 |
| `ZHIPU_API_KEY` | 智谱AI API Key, 从 config.json `zhipu.api_key` 读取 |
| `DASHSCOPE_API_KEY` | 阿里云百炼语音识别, 从 config.json `dashscope.api_key` 读取(兼容环境变量) |
| `ASR_MODEL_CHAIN` | ASR降级链, 从 config.json `asr.model_chain` 读取 |
| `VISUAL_MODEL_CHAIN` | 视觉模型降级链, 从 config.json `visual.model_chain` 读取 |
| `MODEL_NAME` | 回复中展示的模型名, 从 config.json `monitor.model_name` 读取 |
| `QQ_OPENID` | QQ通知目标, 从 config.json `qq.openid` 读取 |

### QQ通知

| 函数 | 说明 |
|------|------|
| `notify_qq(text)` | 通过 `openclaw message send` 发QQ消息 |

### 状态管理

| 函数 | 说明 |
|------|------|
| `load_state()` | 读取 `/tmp/bili_replied_ids.txt`，返回已处理ID集合 |
| `save_state(source_id)` | 追加一个ID到已处理列表 |
| `load_summaries()` | 读取视频总结缓存 |
| `save_summary(bv, summary, duration)` | 保存视频总结 |

### B站 API 封装 (底层)

| 函数 | 说明 |
|------|------|
| `api_get(url)` | GET 请求, 带 Cookie/UA/Referer |
| `api_post_form(url, data)` | POST 表单请求 |
| `reply_comment(oid, root_id, parent_id, message, comment_type)` | 在B站下发回复 `POST /x/v2/reply/add` |
| `fetch_comment_thread(oid, root_rpid, comment_type)` | 获取某条评论的所有子评论 `GET /x/v2/reply/reply` |
| `fetch_top_level_comments(oid, comment_type)` | 获取顶级评论列表 `GET /x/v2/reply/main` |

### 轮询 & 消息获取 (核心)

| 函数 | 说明 |
|------|------|
| `check_unread()` | **唯一轮询入口** — `GET /x/msgfeed/unread`，返回 `{at, reply}` |
| `fetch_at_messages()` | `GET /x/msgfeed/at` — 获取@消息列表 |
| `fetch_reply_messages()` | `GET /x/msgfeed/reply` — 获取评论区回复通知 |

### 消息处理 (顶层)

| 函数 | 说明 |
|------|------|
| `process_new_at_messages()` | 处理所有新@消息(主评论级别): 视频→handle_chat_message, 动态→process_dynamic |
| `process_new_reply_messages()` | 处理回复通知(子评论中的@): 检查at_details过滤, 然后走 handle_chat_message |

### 评论区对话

| 函数 | 说明 |
|------|------|
| `handle_chat_message(item, bv, comment_type, notify_callback)` | **对话处理核心**: 提取文字→获取上下文→意图分类→执行总结/聊天/视频追问→发回复 |
| `extract_dialog_context(sub_replies, our_mid)` | 从子评论中提取 @Bot 的历史对话 |
| `extract_message_after_at(text, at_name)` | 提取 @Bot 之后的有效文字 |
| `classify_user_intent(user_message, has_video_summary, api_key)` | **v5修复** — 意图分类三层: 总结关键词 → 短消息白名单 → LLM分类 `summary/chat/video_chat` |
| `generate_chat_reply(context, user_message, ..., visual_context='')` | GLM生成对话回复, **v4新增 `visual_context` 参数支持视频追问** |

### 视频处理

| 函数 | 说明 |
|------|------|
| `get_video_info(bv)` | `GET /x/web-interface/view` — 获取视频信息(含 copyright/rights/tid 等) |
| `is_official_content(bv)` | **v5修复** — 判断是否番剧/电影等不可下载内容。仅靠 `rights.download=0` + 特殊分区 tid(13/23/167/11/177), 不再误判转载视频 |
| `get_video_duration(bv)` | 获取视频时长(秒) |
| `_download_single_p(bv, p_index, output_path, fmt)` | 下载分P视频的指定P |
| `download_video(bv, output_path)` | yt-dlp下载视频(自动检测分P, 多P逐个下载后用 ffmpeg concat 合并) |
| `extract_frames(video_path, frames_dir, interval)` | ffmpeg截帧 |
| `extract_audio(video_path, output_path)` | ffmpeg提取音频(mp3) |
| `process_video(bv, notify_callback)` | **视频处理主流程**: 下载→截帧→ASR→GLM总结→**缓存关键帧base64** |

### 视觉分析 (GLM 视觉模型降级链)

| 函数 | 说明 |
|------|------|
| `_call_visual_model(content, api_key, max_tokens, timeout)` | **v5新增** — 统一的视觉模型调用, 按 `VISUAL_MODEL_CHAIN` 依次尝试(每个模型3次重试), 全部失败才返回空 |
| `analyze_frames_batch(frames, api_key)` | 单批(≤5帧)送视觉模型描述 → 调用 `_call_visual_model()` |
| `visual_analyze(all_frames, api_key)` | 等距采样20帧→分4批→聚合描述 |
| `cache_frame_b64_list(all_frames)` | **v4新增** — 采样20帧转base64, 缓存到磁盘供追问使用 |
| `load_frame_cache(bv)` | **v4新增** — 读取缓存的关键帧base64列表 |
| `visual_query_frames(frame_cache, user_question, api_key)` | **v4新增** — 用户追问视频具体内容时, 重新查看画面细节 → 调用 `_call_visual_model()` |

### 语音识别 (ASR)

| 函数 | 说明 |
|------|------|
| `get_audio_info(audio_path)` | ffprobe获取音频时长和大小 |
| `split_audio(audio_path, chunk_dir, duration)` | 超280秒/9MB的音频用ffmpeg分段 |
| `_do_api_transcribe(audio_path, model)` | 单次云端ASR调用, 返回 (text, status) |
| `transcribe_local(audio_path, notify)` | 本地faster-whisper medium降级 |
| `transcribe_audio(audio_path, notify_callback)` | **ASR主流程**: 按模型链尝试, 分段, 降级 |

### 文本总结 (GLM-5.1)

| 函数 | 说明 |
|------|------|
| `final_summarize(visual_desc, asr_text, api_key)` | 融合视觉描述+语音文本→生成视频总结(含内容质量兜底:视觉+语音均不足则不编造) |

### 动态处理

| 函数 | 说明 |
|------|------|
| `fetch_dynamic_detail(dynamic_id)` | 获取动态详情 `GET /x/polymer/web-dynamic/v1/detail` |
| `process_dynamic(uri, subject_id, root_id, ...)` | **动态处理主流程**: 解析图文→视觉模型分析图片(走降级链)→GLM-5.1总结 |

### 主入口

| 函数 | 说明 |
|------|------|
| `main()` | 15秒循环: check_unread → 处理at → 处理reply → sleep |

---

## 意图分类详解 (v5.4 更新)

`classify_user_intent()` 采用三层判断。v5.4 核心变化: **首次@必做视频分析, 意图分类只决定回复方式而非是否下载**。

```
用户消息
  │
  ├── 纯@ (无文字) → summary
  │
  ├── 匹配 summary_keywords:
  │     “总结/概括/讲什么/分析视频” 等 → summary
  │
  ├── 短消息 (≤5字):
  │     ├── 匹配 short_summary_patterns:
  │     │     “看看/看下/瞧瞧/这是啥/讲啥” 等 → summary
  │     └── 不匹配 → 走 LLM 分类
  │
  └── 其他 → LLM 分类 (GLM-5.1) → summary / chat / video_chat
          │
          └── v5.4: 模糊消息(问候/感叹/日常用语) 优先判为 summary
              只有明确的问题或话题才判为 chat
```

### B站 copyright 字段说明

| 值 | 含义 | 是否可下载 | 旧逻辑 | v5修复后 |
|---|------|-----------|-------|---------|
| 1 | 自制(原创) | ✅ 是 | 正常处理 | 正常处理 |
| 2 | 转载 | ✅ 是(普通up可发) | ❌ 误判官方内容 | 正常处理 |
| 3+ | 联合投稿等其他 | ✅ 是 | ❌ 误判官方内容 | 正常处理 |

真正的官方内容通过 `rights.download=0` 和特殊分区 tid 来判断, 与 copyright 无关。

---

## 数据流: v5.4 统一处理路径

### v5.4 核心变化: 首次@必做视频分析

v5.4 之前: 意图分类决定是否下载视频 (chat 跳过下载, 导致空对空聊天)
v5.4 之后: 首次@必做视频分析, 意图分类只决定回复内容

### 路径1: 首次@ (无缓存) — 视频总结

```
用户发主评论 "@Bot 总结一下"  (或任何首次@)
  ↓
GET /x/msgfeed/unread  →  at=1
  ↓
GET /x/msgfeed/at  →  返回这条@消息
  ↓
process_new_at_messages()
  ↓ 解析: bv=BVxxx, business_id=1(视频)
  ↓
is_official_content()  →  False (普通视频)
  ↓
handle_chat_message()
  ├─ [v5.4] 检查缓存 → 无缓存 → 必做视频分析
  ├─ process_video()
  │   ├─ download_video()
  │   ├─ extract_frames()  →  visual_analyze() [_call_visual_model: 4.6v→4v降级]
  │   ├─ cache_frame_b64_list()  →  保存帧base64
  │   ├─ extract_audio()   →  transcribe_audio()
  │   ├─ final_summarize() [GLM-5.1]
  │   └─ dump frames_cache_{BV}.json
  ├─ save_summary()  →  缓存总结
  ├─ classify_user_intent()  →  "summary"
  └─ reply_comment()  →  回复总结 + notify_qq()
```

### 路径2: 首次@但意图为 chat — 结合视频内容对话

```
用户 "@Bot @其他人 哈哈这个视频有意思"
  ↓
handle_chat_message()
  ├─ [v5.4] 检查缓存 → 无缓存 → 必做视频分析
  ├─ process_video() → 生成总结并缓存
  ├─ classify_user_intent()
  │   └─ LLM 判为 "chat" (有明确情感但非总结请求)
  ├─ generate_chat_reply(
  │       video_summary="视频讲述了...",  ← 必须结合视频内容
  │       user_message="哈哈这个视频有意思"
  │   )
  │   └─ system prompt 要求: "回复必须结合视频内容"
  └─ reply_comment()  →  回复 (如: "确实有意思,UP主讲的xxx挺逗的")
```

### 路径3: 已有缓存 + 任意意图

```
用户 "@Bot 视频里那个人穿的什么衣服？"
  ↓
handle_chat_message()
  ├─ [v5.4] 检查缓存 → 有缓存 → 直接复用, 不重复下载
  ├─ classify_user_intent()  →  "video_chat"
  ├─ load_frame_cache(bv)    →  读取缓存的关键帧base64
  ├─ visual_query_frames()   →  _call_visual_model() 重新查看画面
  └─ generate_chat_reply(visual_context="画面中人物穿着...")  →  回复
```

---

## 模型使用一览

| 任务 | 模型 | API接口 | 价格 |
|------|------|---------|------|
| 视频帧分析 | `glm-4.6v-flash` → `glm-4v-flash` | `/api/paas/v4/chat/completions` | 免费(自动降级) |
| 动态图片分析 | `glm-4.6v-flash` → `glm-4v-flash` | `/api/paas/v4/chat/completions` | 免费(自动降级) |
| 视频追问(重新查看帧) | `glm-4.6v-flash` → `glm-4v-flash` | `/api/paas/v4/chat/completions` | 免费(自动降级) |
| 视频总结 | `glm-5.1` | `/api/anthropic/v1/messages` (Anthropic兼容) | 免费 |
| 意图分类 | `glm-5.1` | `/api/anthropic/v1/messages` | 免费 |
| 聊天对话 | `glm-5.1` | `/api/anthropic/v1/messages` | 免费 |
| 语音识别 | `qwen3-asr-flash` (多模型降级) | 阿里云百炼 DashScope | 免费(36000次/模型) |

---

## 降级链设计

所有有多个免费模型的场景都实现了自动降级, 不需要人工干预:

| 场景 | 链 | 触发条件 |
|------|-----|---------|
| 视觉分析 | glm-4.6v-flash → glm-4v-flash | 429限流 / 1305并发过大, 每模型3次重试后切换 |
| 语音识别 | flash-2026-02-10 → 2025-09-08 → flash → 本地Whisper medium | 免费额度耗尽(quota_exhausted), 每个模型有独立36,000次额度 |
| 综合总结 | 无降级(仅 glm-5.1) | 内容不足时直接返回"信息不足"而非编造 |

---

## 配置修改指南

所有配置项均在 `config.json` 中修改（参考 `config.example.json`）。脚本启动时自动加载。

### 修改轮询间隔

编辑 `config.json`:
```json
{
    "monitor": {
        "poll_interval": 15
    }
}
```

### 修改ASR模型降级链

编辑 `config.json`:
```json
{
    "asr": {
        "model_chain": [
            "qwen3-asr-flash-2026-02-10",
            "qwen3-asr-flash-2025-09-08",
            "qwen3-asr-flash"
        ],
        "chunk_duration": 280,
        "chunk_max_bytes": 9437184
    }
}
```
每个模型有独立36,000次免费额度, 用完后自动切换下一个。

### 修改视觉模型降级链

编辑 `config.json`:
```json
{
    "visual": {
        "model_chain": [
            "glm-4.6v-flash",
            "glm-4v-flash"
        ]
    }
}
```
三处视觉调用(帧分析/视频追问/动态图片)统一走 `_call_visual_model()`。
每个模型最多3次重试, 全部失败才返回空。

### 修改回复模型名

编辑 `config.json`:
```json
{
    "monitor": {
        "model_name": "GLM-5.1"
    }
}
```
v5.3: 已不再出现在回复文本中, 仅内部记录。

### 修改Bot UID (如果换号)

编辑 `config.json`:
```json
{
    "bilibili": {
        "bot_mid": 12345,
        "bot_name": "你的Bot名称"
    }
}
```

### 修改B站Cookie (过期时)

**方式1: 扫码登录 (推荐, 自动获取)**

通过B站二维码登录API自动获取新Cookie, 无需手动操作:

```bash
# 1. 生成二维码
curl -s 'https://passport.bilibili.com/x/passport-login/web/qrcode/generate' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' \
  -H 'Referer: https://www.bilibili.com/'

# 返回 {"data": {"url": "...", "qrcode_key": "xxx"}}

# 2. 将 url 生成二维码图片, 用B站APP扫码确认
python3 -c "import qrcode; qrcode.make('上面的url').save('/tmp/bili_qr.png')"

# 3. 轮询确认登录 (替换 qrcode_key)
curl -s 'https://passport.bilibili.com/x/passport-login/web/qrcode/poll?qrcode_key=<qrcode_key>' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' \
  -H 'Referer: https://www.bilibili.com/' \
  -D /tmp/bili_headers.txt

# code=0 表示登录成功, SESSDATA 和 bili_jct 在 set-cookie 响应头中
# 从 url 参数或 set-cookie 头提取 SESSDATA 和 bili_jct
```

**方式2: 浏览器手动获取**

在浏览器 F12 → Application → Cookies → bilibili.com 获取 `SESSDATA` 和 `bili_jct`,
然后更新 `config.json`:

```json
{
    "bilibili": {
        "sessdata": "...",
        "bili_jct": "..."
    }
}
```

Cookie 更新后需重启脚本生效。

### 修改ASR分段阈值

编辑 `config.json`:
```json
{
    "asr": {
        "chunk_duration": 280,
        "chunk_max_bytes": 9437184
    }
}
```
- `chunk_duration`: 每段最长280秒
- `chunk_max_bytes`: 每段最大9MB

### 修改意图分类的短消息白名单

```python
short_summary_patterns = [
    '看看', '看下', '看一下', '瞧瞧', '瞧瞧这个',
    '这是啥', '啥内容', '这是什么', '这啥', '讲啥', '讲什么',
    '概括', '帮我看看',
]
# 只有匹配这些的短消息(≤5字)才判为 summary
# 不匹配的短消息(如"搞莫子/干嘛/在吗")会走 LLM 分类
```

---

## 已知问题 & 注意事项

1. **配置文件必须存在** — 首次使用需复制 `config.example.json` 为 `config.json` 并填写真实配置
2. **B站 Cookie 会过期** — 如果回复失败增多，先检查 Cookie
3. **100分钟以上视频不处理** — `process_video()` 中硬编码了 `>6000秒` 的限制
4. **视频下载错误由 yt-dlp 报错** — 如果下载失败频繁，检查 yt-dlp 版本和 cookies 文件是否正确
5. **代理已改为自动检测** — 启动时探测7890端口, 有则走代理无则直连, 不再依赖手动启停
6. **关键帧缓存不自动清理** — `/tmp/bili_monitor/frames_cache_{BV}.json` 会积累，目前没有过期机制
7. **回复被B站审核折叠(state=17)** — 见下方说明

### B站评论审核折叠 (state=17)

B站对评论有自动审核机制，回复发送成功(code=0)后仍可能被折叠(state=17, 仅自己可见)。

**已采取的措施 (v5.3):**
- 回复控制在合理长度内（300字以内），发送后自动检查审核状态
- 去掉了 `Hello,是你召唤来了我,当前模型是:XXX` 的机器人前缀
- 总结 prompt 要求不换行、不分点，写成一段自然文字

**如果仍被折叠：** 可能是账号风控等级高，尝试在B站客户端手动发几条正常评论活跃一下账号。

### 已修复

- [x] 2026-05-14: **is_official_content() 误拦截转载视频** — 原逻辑 `copyright != 1` 将普通up的转载/联合投稿视频当成官方内容拦截。B站 copyright=2(转载)/3(联合投稿) 不代表不可下载。已移除该判断, 仅靠 `rights.download=0` 和特殊分区 tid(13/23/167/11/177) 判别真正的官方内容。
- [x] 2026-05-14: **短消息无条件判为 summary 导致误触发视频下载** — 原逻辑 `len(msg) <= 5` 对任何短消息(Chat)都判为 Summary, 导致"搞莫子"/"干嘛"等日常对话触发视频下载。改为三层判断: 总结关键词 → 短消息白名单("看看/看下/瞧瞧/这是啥"等) → LLM分类。不匹配白名单的短消息走 LLM 意图分类。
- [x] 2026-05-14: **新增视觉模型降级链 `VISUAL_MODEL_CHAIN`** — glm-4.6v-flash 3次重试失败后自动切换 glm-4v-flash。统一 `_call_visual_model()` 函数覆盖帧分析(`analyze_frames_batch`)、视频追问(`visual_query_frames`)、动态图片(`process_dynamic`) 三处调用, 每个模型3次重试 + 模型间降级, 全部失败才返回空。
- [x] 2026-05-27: **`final_summarize()` 英文ASR内容被丢弃** — 原逻辑仅统计中文字符数(>=4)判断ASR有效性，纯英文内容（如YouTube搬运视频7301词英文ASR）因0个中文字被丢弃，导致输出"无法生成视频总结"。修复为同时统计中文字数和英文词数，取max(中文,英文)>=4即视为有效。同理修复了仅视觉描述时的内容不足检查。
- [x] 2026-05-28: **文档更新** — 补充 Cookie 扫码登录方式、Clash 代理管理说明 (clashctl/mihomo)、心跳频率说明 (20轮=5分钟)、代理故障排查步骤。
- [x] 2026-06-01: **回复被B站审核折叠(state=17)** — 长回复(>300字)和带机器人前缀的回复频繁被B站审核系统折叠。已去掉机器人前缀，总结要求不换行不分点。经测试97字纯自然语言回复可正常显示(state=0)。
- [x] 2026-06-09: **v5.4.2 回复长度限制放宽 + 审核状态自动检查** — 移除150字截断限制，改为300字以内让AI把内容说完整。新增审核状态自动检查：发送回复后立即检查state字段，如被拦截(state=17)则自动发送抱歉通知；如state=1则45秒后延迟复查确认。审核被拦时用户会收到B站评论区的抱歉消息。
- [x] 2026-06-03: **v5.4 首次@必做视频分析 + chat回复结合视频内容** — 原逻辑中 chat 意图会跳过视频下载, 导致空对空聊天。用户@了多人但不打算聊天时, 回复内容与语境完全不符。改为: (1)首次@无缓存时无论意图都必做视频分析; (2)chat 回复的 system prompt 强制要求结合视频内容; (3)意图分类 prompt 增加模糊消息优先判为 summary 的原则。
- [x] 2026-06-09: **v5.4.1 配置外部化 + GitHub上传支持** — 所有硬编码密钥/参数改为从 config.json 读取。新增 config.example.json 模板。.gitignore 排除 config.json 防止泄露。配置文件支持 --config 参数 / BILI_CONFIG 环境变量 / 同目录 config.json 三级优先。GitHub token 存入 config.json 供上传使用。
- [x] 2026-06-14: **v5.4.3 视频下载 412 修复** — B站 playinfo API 对 `--add-header Cookie:` 方式返回 HTTP 412 (Precondition Failed), 导致所有视频下载失败、回复均为"视频下载失败"。根因: B站加强反爬, yt-dlp 通过 header 传递 Cookie 时缺少必要的 wbi 签名验证。修复: 改用 `--cookies` Netscape 文件方式传递 Cookie (脚本启动时自动生成 `/tmp/bili_monitor/bili_cookies.txt`)。同时 yt-dlp 从 2026.03.17 更新至 2026.06.09, 增加格式回退逻辑。
- [x] 2026-06-17: **v5.5 分P视频支持** — 分P视频(如 BV19aVp6dEe7 有2P) 只下载第一P, 导致 ASR 和视觉分析只覆盖小部分内容, GLM-5.1 因信息不足返回"信息不足,无法准确总结"。修复: `download_video()` 新增分P检测, 多P时逐个下载后用 ffmpeg concat demuxer 合并为单文件(concat copy 失败时自动降级重编码)。同步提升 extract_frames/extract_audio timeout 至 300 秒。
- [x] 2026-06-18: **v5.5.1 新版合集视频下载失败** — B站新版 anthology 格式视频(如 BV1fz421f7tk)只有分离的音视频流, 没有合并格式。格式链 `['30016+30216', 'best']` 均匹配失败, 返回 "Requested format is not available"。修复: 单P和多P下载的格式链均新增 `bestvideo+bestaudio` 作为中间回退, 新版分离流视频自动走 `40028+30280` 等组合下载后合并。

---

## GitHub 上传指南

项目支持通过 config.json 中的 `github` 字段管理 GitHub 仓库。

### config.json 中的 GitHub 配置

```json
{
    "github": {
        "token": "ghp_xxxxxxxxxxxx",
        "repo": "owner/repo-name",
        "note": "token 仅用于项目上传至 GitHub, 不参与脚本运行"
    }
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `github.token` | GitHub Personal Access Token, 需要 `repo` 权限(完整仓库读写) |
| `github.repo` | 目标仓库, 格式 `owner/repo-name`, 如 `user/bili-monitor` |
| `github.note` | 备注说明, 仅供阅读理解, 不参与任何逻辑 |

### 安全说明

- `config.json` 已被 `.gitignore` 排除, token 不会上传到仓库
- `config.example.json` 中仅包含占位符 `YOUR_GITHUB_TOKEN_HERE`
- token 仅用于大模型/脚本读取后调用 GitHub API 上传项目, **不参与 bili_monitor.py 的运行逻辑**

### 上传流程 (大模型参考)

1. 读取 config.json 中的 `github.token` 和 `github.repo`
2. 使用 GitHub API 创建仓库 (如不存在): `POST https://api.github.com/user/repos`
3. 添加远程仓库: `git remote add origin https://x-access-token:<token>@github.com/<repo>.git`
4. 推送: `git push -u origin master`

---

## ⚠️ 免责声明 (完整版)

本项目（Bili Monitor）的全部源代码均由 AI（大语言模型）在人工提示词引导下自动生成，**未经过任何专业人工代码审查或安全审计**。

### 1. 代码质量

- AI 生成的代码可能存在逻辑错误、安全漏洞、性能问题或未知的边界情况
- 代码中的注释和文档可能与实际行为不完全一致
- 不保证代码在所有环境下的稳定性和正确性

### 2. 使用风险

- 本脚本涉及 **B站账号的自动化操作**（登录态轮询、自动评论），可能违反哔哩哔哩的用户服务协议
- 使用本脚本可能导致 B站账号被 **限制、封禁或永久封号**
- 脚本中使用的第三方 API（智谱AI、阿里云百炼等）的免费额度可能随时变更
- B站 Cookie 中包含敏感的登录凭证，泄露后可能导致账号被盗

### 3. 法律与责任

- **使用者需自行承担使用本项目的全部风险和后果**
- 本项目作者不对因使用、修改、分发本项目造成的任何直接或间接损失负责
- 包括但不限于：账号损失、数据丢失、财产损失、法律纠纷等
- 使用者应当遵守所在地区的法律法规，不得将本工具用于任何违法用途

### 4. 知识产权

- 本项目采用 MIT License 开源
- 使用者应尊重相关第三方平台（哔哩哔哩、智谱AI、阿里云等）的服务条款和知识产权

### 5. 建议

- 在使用前 **逐行阅读并理解全部源代码**
- 在测试环境中充分验证后再投入实际使用
- 定期检查B站账号状态，如发现异常立即停止使用
- 不要在不信任的环境中运行本脚本
