# bili-monitor 容器化部署
# 无 GPU 环境: ASR 使用 CPU 推理(SenseVoiceSmall), 视觉/文本保持云端(GLM + DeepSeek 降级)
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# CPU 版 PyTorch(体积小, 无 CUDA), 供 funasr 本地语音推理
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# yt-dlp 单独安装, 便于后续升级
RUN pip install -U yt-dlp

COPY . .

RUN mkdir -p data/logs data/state data/cache data/ad_detection data/work

VOLUME ["/app/data"]

CMD ["python", "-u", "bili_monitor.py"]
