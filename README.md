# B站@消息自动监控 - AI视频总结 & 评论区对话

B站自动监控脚本：检测到 `@文共` 后自动下载视频 → 截帧分析 + ASR语音识别 → GLM总结 → 评论区回复，同时通过QQ Bot通知。

## 功能

- 🎬 **视频自动总结** — 首次@自动下载、截帧、ASR识别、GLM总结
- 💬 **评论区对话** — 结合视频内容智能回复，支持追问视频细节
- 🔄 **多模型降级** — 视觉/语音模型链式降级，免费额度自动切换
- 📱 **QQ通知** — 处理进度和结果实时推送到QQ
- 🔄 **代理自适应** — 自动检测本地代理，有则走代理无则直连
- 📋 **动态总结** — 支持B站动态/专栏图文内容分析

## 快速开始

```bash
# 1. 安装依赖
pip install requests faster-whisper
apt install ffmpeg
# 安装 yt-dlp: https://github.com/yt-dlp/yt-dlp

# 2. 配置
cp config.example.json config.json
nano config.json  # 填写 B站Cookie、API Key 等

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
| `bilibili.wengong_mid` | 被监控账号的B站UID |
| `zhipu.api_key` | 智谱AI API Key |
| `qq.openid` | QQ通知目标OpenID |

### 可选项（有默认值）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dashscope.api_key` | 空 | 阿里云百炼ASR Key（不填则用本地Whisper） |
| `monitor.poll_interval` | 15 | 轮询间隔（秒） |
| `proxy.host` / `proxy.port` | 127.0.0.1:7890 | 代理地址 |
| `asr.model_chain` | qwen3-asr-flash系列 | ASR模型降级链 |
| `visual.model_chain` | glm-4.6v-flash系列 | 视觉模型降级链 |

## 详细文档

参见 [bili_monitor.md](./bili_monitor.md)。

## 依赖

- Python 3.8+
- yt-dlp（视频下载）
- ffmpeg（截帧 + 音频提取）
- requests（HTTP请求）
- faster-whisper（本地ASR降级，可选）
- OpenClaw CLI（QQ消息通知）

## 许可证

MIT License
