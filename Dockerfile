# bili-monitor 容器化部署
# GPU环境: GTX1050 2GB 已验证兼容 CUDA 12.6 / sm_61; 无GPU时程序自动回退CPU
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

# CUDA 版 PyTorch: GTX1050(Pascal, sm_61)实测可运行 SenseVoiceSmall
# 运行时未暴露GPU时, bili_monitor 会自动回退CPU
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.11.0+cu126 torchaudio==2.11.0+cu126 torchvision==0.26.0+cu126

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# yt-dlp 单独安装, 便于后续升级
RUN pip install -U yt-dlp

COPY . .

RUN mkdir -p data/logs data/state data/cache data/ad_detection data/work

VOLUME ["/app/data"]

CMD ["python", "-u", "bili_monitor.py"]
