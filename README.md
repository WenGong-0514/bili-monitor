# Bili Monitor - B站@消息AI自动监控

> ⚠️ **AI Generated Project** — 本项目的全部代码均由 AI（大语言模型）在人工提示词引导下自动生成，**未经过专业人工代码审核**。仅供学习交流使用，使用者需自行承担所有风险。详见下方免责声明。

---

## ⚠️ 免责声明

**本项目代码完全由 AI 生成，未经人工审核。** 克隆、使用、修改或分发本项目的任何行为所造成的直接或间接损失（包括但不限于账号封禁、数据丢失、财产损失、法律纠纷等），与项目作者无任何关系。使用者应当：

- 在使用前自行审查全部代码
- 了解本脚本涉及B站账号自动化操作，可能违反B站用户协议
- 自行承担使用本项目的所有风险和后果

---

B站自动监控脚本：检测到 `@Bot` 后自动下载视频 → 截帧分析 + ASR语音识别 → GLM总结 → 评论区回复，同时通过QQ Bot通知。

## 最近更新

- **2026-06-26 (v5.6.1)**: QQ 通知改为官方 Bot API 直连 — `channels.qqbot.{appId, clientSecret}` 直连, 脱离 OpenClaw 依赖; 老版 CLI 作为回退兼容
- **2026-06-26 (v5.6.0)**: ASR 改为本地推理优先 — SenseVoiceSmall + FSMN-VAD (funasr), 10x+ 实时速度; 云端链作为兜底
- **2026-06-19 (v5.5.2)**: 修复同线程重复发送总结 — B站拦截回复后去重失效,改为线程级去重
- **2026-06-18 (v5.5.1)**: 修复新版合集视频下载失败 — 新增 `bestvideo+bestaudio` 格式回退
- **2026-06-17 (v5.5)**: 分P视频支持 — 自动检测分P数量, 逐P下载后 ffmpeg concat 合并，确保完整分析
- **2026-06-14 (v5.4.3)**: 修复视频下载失败 — B站对 `--add-header Cookie` 返回 412，改用 `--cookies` 文件方式

## 功能

- 🎬 **视频自动总结** — 首次@自动下载、截帧、ASR识别、GLM总结
- 💬 **评论区对话** — 结合视频内容智能回复，支持追问视频细节
- 🎙️ **本地 ASR 优先** — SenseVoiceSmall + FSMN-VAD (funasr), 10x+ 实时速度; 无 funasr 时自动降级到云端
- 🔄 **多模型降级** — 视觉/语音模型链式降级，免费额度自动切换
- 📱 **QQ 通知 (官方 Bot API)** — 处理进度和结果实时推送到 QQ, 不依赖 OpenClaw CLI
- 🔄 **代理自适应** — 自动检测本地代理，有则走代理无则直连
- 📋 **动态总结** — 支持B站动态/专栏图文内容分析

## 快速开始

```bash
# 1. 安装依赖
pip install requests faster-whisper
# 本地 ASR (可选, 但强烈推荐): pip install funasr torch torchaudio
apt install ffmpeg
# 安装 yt-dlp: https://github.com/yt-dlp/yt-dlp

# 2. 配置
cp config.example.json config.json
nano config.json  # 填写 B站Cookie、API Key、QQ Bot 凭据等

# 3. 启动
nohup python3 -u bili_monitor.py >> /tmp/bili_monitor.log 2>&1 &
```

## 配置

所有敏感信息和可调参数均在 `config.json` 中配置（参考 `config.example.json`）。

配置文件查找优先级：
1. `--config /path/to/config.json` 命令行参数
2. `BILI_CONFIG` 环境变量
3. 脚本同目录下的 `config.json`

### 必填项

| 字段 | 说明 |
|------|------|
| `bilibili.sessdata` | B站登录Cookie |
| `bilibili.bili_jct` | B站CSRF Token |
| `bilibili.bot_mid` | Bot账号的B站UID |
| `bilibili.bot_name` | Bot在B站的昵称（用于识别@消息） |
| `zhipu.api_key` | 智谱AI API Key |
| `channels.qqbot.appId` | QQ 官方 Bot AppID |
| `channels.qqbot.clientSecret` | QQ 官方 Bot ClientSecret |
| `channels.qqbot.openid` | 接收通知的用户 OpenID |

### 可选项（有默认值）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dashscope.api_key` | 空 | 阿里云百炼ASR Key（不填则用本地Whisper） |
| `monitor.poll_interval` | 15 | 轮询间隔（秒） |
| `proxy.host` / `proxy.port` | 127.0.0.1:7890 | 代理地址 |
| `asr.local_first` | true | 优先用本地 SenseVoiceSmall, 失败降级云端 |
| `asr.local_model` | `iic/SenseVoiceSmall` | 本地 ASR 模型 |
| `asr.local_vad_model` | `iic/speech_fsmn_vad_...` | 本地 VAD 模型 |
| `asr.local_threads` | 8 | 本地 ASR 推理线程数 |
| `asr.model_chain` | qwen3-asr-flash系列 | ASR云端降级链（本地失败时使用） |
| `visual.model_chain` | glm-4.6v-flash系列 | 视觉模型降级链 |

## 详细文档

参见 [bili_monitor.md](./bili_monitor.md)。

## 依赖

- Python 3.8+
- yt-dlp（视频下载）
- ffmpeg（截帧 + 音频提取）
- requests（HTTP请求）
- funasr + torch（本地 ASR 推荐, 缺失时自动降级云端）
- faster-whisper（最深层级 ASR 兜底，可选）
- QQ 官方 Bot 账号（[q.qq.com](https://q.qq.com) 申请, 用于通知推送）

## 许可证

MIT License
