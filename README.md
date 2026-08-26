# Bili Monitor - B站@消息AI自动监控

> ⚠️ **AI Generated Project** — 本项目的全部代码均由 AI（大语言模型）在人工提示词引导下自动生成，**未经过专业人工代码审核**。仅供学习交流使用，使用者需自行承担所有风险。详见下方免责声明。

---

## ⚠️ 免责声明

**本项目代码完全由 AI 生成，未经人工审核。** 克隆、使用、修改或分发本项目的任何行为所造成的直接或间接损失（包括但不限于账号封禁、数据丢失、财产损失、法律纠纷等），与项目作者无任何关系。使用者应当：

- 在使用前自行审查全部代码
- 了解本脚本涉及B站账号自动化操作，可能违反B站用户协议
- 自行承担使用本项目的所有风险和后果

---

B站自动监控脚本：检测到 `@Bot` 后自动下载视频 → 截帧分析 + ASR语音识别 → 本地视觉/文本总结（v5.11 起全本地流水线）→ 内嵌广告识别 → 评论区回复，同时通过QQ Bot通知。

## 最近更新

- **2026-08-26 (v5.13.1)**: 广告检测修复 — 视觉窗口按实际截帧时间戳合并相邻广告窗口（合并间隔=帧间隔，随截帧策略自适应），时长 <10s 的粗略提及不再标记；回复前广告提示改为逐段标记所有广告段（此前只提示第一段）
- **2026-08-26 (v5.13.0)**: 总结长度随时长动态调整 — 视频越短总结越精炼、越长越详细（时长→token 上限：30s→100 ~ 2h→2000，分段线性映射），本地/云端总结均透传该上限并动态生成字数目标；另加 B站 1000 字评论上限的句子边界截断保护
- **2026-08-26 (v5.12.0)**: 意图先行分流 — 意图分析最先用本地 LLM 判断，纯@/无明确意图才触发下载+ASR+视觉+文本总结；聊天/追问直接复用缓存（ASR转写+视觉描述+关键帧），不重复识别；已总结视频直接复用缓存总结
- **2026-08-26 (v5.11.1)**: 本地对话回复 — 意图分类与评论区对话回复改用本地 Qwen3-8B（不再依赖云端 GLM/DeepSeek）；本地 LLM 常驻复用（视频总结→意图→回复共用，视觉加载前自动释放显存）
- **2026-08-25 (v5.11.0)**: 本地全流程流水线 — 不做量化，ASR→视觉→文本 三段串行，每阶段只加载一个模型用完即释放显存（ASR=SenseVoiceSmall，视觉=Qwen2.5-VL-7B，文本=Qwen3-8B）。`local_pipeline.enabled` 开启后视觉/文本不再依赖云端，RTX 2080 Ti 22GB 实测十分钟视频全程约4分钟
- **2026-08-25 (v5.10.0)**: GTX1050 CUDA ASR — Docker 启用 `nvidia` runtime，安装 PyTorch 2.11.0+cu126；SenseVoiceSmall 自动 CUDA 优先/CPU 回退（22个30秒分块实测23.8s，约27.7x实时，较CPU约4.2倍）。同时广告检测复用阶段2关键帧与分块ASR，不再二次切帧/切音频/重复ASR
- **2026-08-21 (v5.9.1)**: 在线 ASR 全部过期下线 — 删除云端降级链，语音识别统一为本地 SenseVoiceSmall，仅保留本地 faster-whisper 兜底
- **2026-08-21 (v5.9.0)**: Docker 容器化部署 — 新增 `Dockerfile` + `docker-compose.yml`（CPU 版 PyTorch + funasr 本地 ASR）；文本模型新增 DeepSeek V4 Pro 降级链（GLM 不可用时自动切换）
- **2026-08-19 (v5.8.0)**: 内嵌广告识别上线 — 在已下载视频上复用稀疏多帧视觉AI + 30秒时间戳ASR + LLM语义分析 + `blacklist.txt` 黑名单；无广告回复保持原样式，有广告在总结前追加 `检测到xx广告，大约位于mm:ss-mm:ss，跳过空降坐标mm:ss。`
- **2026-08-19 (v5.7.0)**: 可靠消息轮询 — `unread` 15秒快速检测，@/回复列表每 3600 秒兜底；B站 API 连续超时/风控后自动退避 15 分钟并 QQ 告警；状态、总结、帧缓存、下载文件和日志全部持久化到项目内 `data/`

- **2026-06-26 (v5.6.3)**: prompt 约束 — 回复中不提及 ASR/视觉识别的失误, 即使识别有误也直接当成自己观察到的内容叙述
- **2026-06-26 (v5.6.2)**: 启动 dry-populate 防雪崩 — 进入主循环前把 unread 列表全部标记为已知, 防止进程重启/state file 不完整时历史未读被当成新消息挨个回复
- **2026-06-26 (v5.6.1)**: QQ 通知改为官方 Bot API 直连 — `channels.qqbot.{appId, clientSecret}` 直连, 脱离 OpenClaw 依赖; 老版 CLI 作为回退兼容
- **2026-06-26 (v5.6.0)**: ASR 改为本地推理优先 — SenseVoiceSmall + FSMN-VAD (funasr), 10x+ 实时速度; 云端链作为兜底
- **2026-06-19 (v5.5.2)**: 修复同线程重复发送总结 — B站拦截回复后去重失效,改为线程级去重
- **2026-06-18 (v5.5.1)**: 修复新版合集视频下载失败 — 新增 `bestvideo+bestaudio` 格式回退
- **2026-06-17 (v5.5)**: 分P视频支持 — 自动检测分P数量, 逐P下载后 ffmpeg concat 合并，确保完整分析
- **2026-06-14 (v5.4.3)**: 修复视频下载失败 — B站对 `--add-header Cookie` 返回 412，改用 `--cookies` 文件方式

## 功能

- 🎬 **视频自动总结** — 首次@自动下载、截帧、ASR识别、本地全流程总结（v5.11 起 ASR→视觉→文本 全本地）
- 🚫 **内嵌广告识别** — 视觉AI多帧联合判断 + 时间戳ASR语义分析 + 商家黑名单，输出广告品牌与空降坐标
- 💬 **评论区对话** — 结合视频内容智能回复，支持追问视频细节
- 🎙️ **本地 ASR 唯一路径** — SenseVoiceSmall + FSMN-VAD (funasr), 支持 CUDA 加速与 CPU 自动回退; 在线 ASR 已全部过期下线
- 🧠 **本地全流程 (v5.13.1)** — 不做量化，ASR→视觉→文本 三段串行 + 本地对话回复/意图分类（Qwen2.5-VL-7B 视觉 + Qwen3-8B 文本），全链路不依赖云端
- 🔄 **多模型降级** — 视觉模型链式降级 + 文本模型 GLM→DeepSeek V4 Pro 降级
- 📱 **QQ 通知 (官方 Bot API)** — 处理进度和结果实时推送到 QQ, 不依赖 OpenClaw CLI
- 🔄 **代理自适应** — 自动检测本地代理，有则走代理无则直连
- 🛡️ **列表兜底 + API退避** — unread 计数异常时由小时级列表兜底；连续超时/风控自动暂停请求
- 📋 **动态总结** — 支持B站动态/专栏图文内容分析

## 快速开始

```bash
# 1. 安装依赖
pip install requests funasr torch torchaudio
# 可选本地兜底: pip install faster-whisper
apt install ffmpeg
# 安装 yt-dlp: https://github.com/yt-dlp/yt-dlp

# 2. 配置
cp config.example.json config.json
nano config.json  # 填写 B站Cookie、API Key、QQ Bot 凭据等

# 3. 启动
mkdir -p data/logs
nohup python3 -u bili_monitor.py >> data/logs/bili_monitor.log 2>&1 &
```
## Docker 部署（推荐）

```bash
# 1. 构建镜像（CUDA 版 PyTorch；GPU不可用时程序自动回退CPU）
docker compose build

# 2. 配置（首次）
cp config.example.json config.json   # 填写 B站Cookie、API Key、QQ Bot 凭据、DeepSeek 降级密钥等
# config.yaml 广告检测参数 / blacklist.txt 黑名单 默认即可用

# 3. 启动
docker compose up -d
docker compose logs -f   # 查看启动横幅与轮询日志
```

容器说明：

- 数据持久化：`./data:/app/data`（状态、总结缓存、广告结果、工作目录全在项目自身目录内）
- ASR 模型缓存：命名卷 `modelscope_cache`，首次视频分析时自动下载 SenseVoiceSmall，之后复用不重复下载
- v5.11 本地全流程：视觉 Qwen2.5-VL-7B / 文本 Qwen3-8B 首次运行自动从 hf-mirror 下载（各约16GB），同样缓存于 `modelscope_cache` 卷；`config.json` 中 `local_pipeline.enabled` 开启
- ASR 模型缓存：命名卷 `modelscope_cache`，首次视频分析时自动下载 SenseVoiceSmall，之后复用不重复下载
- 真实配置以只读方式挂载：`config.json` / `config.yaml` / `blacklist.txt` 不入镜像、改动即时生效
- 重启策略 `unless-stopped`，Docker 服务重启后自动拉起
- 日志：json-file，单文件 20MB × 3 自动轮转

> 提示：文本模型降级链为 智谱 GLM（付费）→ DeepSeek V4 Pro。在 `config.json` 的 `deepseek` 段填入 `api_key` 即可启用；留空则仅使用 GLM。

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
| `monitor.poll_interval` | 15 | `unread` 快速轮询间隔（秒） |
| `monitor.at_fallback_interval` | 3600 | @消息列表兜底间隔（秒） |
| `monitor.reply_fallback_interval` | 3600 | 回复消息列表兜底间隔（秒） |
| `proxy.host` / `proxy.port` | 127.0.0.1:7890 | 代理地址 |
| `asr.local_first` | true | 本地 SenseVoiceSmall 唯一路径（在线 ASR 已下线） |
| `ad_detection.enabled` | true | 是否启用内嵌广告识别；详细参数在 `config.yaml`，黑名单在 `blacklist.txt` |
| `asr.local_model` | `iic/SenseVoiceSmall` | 本地 ASR 模型 |
| `asr.local_vad_model` | `iic/speech_fsmn_vad_...` | 本地 VAD 模型 |
| `asr.local_threads` | 8 | CPU回退时本地 ASR 推理线程数 |
| `asr.device` | `auto` | ASR设备：`auto`(有CUDA用CUDA) / `cuda` / `cpu` |
| `local_pipeline.enabled` | false | 是否启用本地全流程流水线（ASR→视觉→文本，全本地推理，不依赖云端 GLM） |
| `asr.device` | `auto` | ASR设备：`auto`(有CUDA用CUDA) / `cuda` / `cpu` |
| `visual.model_chain` | glm-4.6v-flash系列 | 视觉模型降级链 |
| `deepseek.api_key` | 空 | DeepSeek V4 Pro 降级链密钥（GLM 不可用时自动切换，OpenAI 兼容） |

## 详细文档

参见 [bili_monitor.md](./bili_monitor.md)。

## 依赖

- Python 3.8+
- yt-dlp（视频下载）
- ffmpeg（截帧 + 音频提取）
- requests（HTTP请求）
- funasr + torch（本地 ASR, 唯一路径）
- faster-whisper（本地 ASR 兜底，可选）
- transformers + accelerate + torchvision（v5.11 本地视觉/文本模型推理）
- faster-whisper（本地 ASR 兜底，可选）
- QQ 官方 Bot 账号（[q.qq.com](https://q.qq.com) 申请, 用于通知推送）

## 许可证

MIT License
