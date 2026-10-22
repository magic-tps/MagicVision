FROM python:3.11-slim-bookworm
ARG TORCH_FLAVOR=cu118
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics \
    MODEL_PATH=/opt/ml/model/best.pt MODEL_DEVICE=cuda \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/program
COPY requirements.txt constraints-cu118.txt ./
RUN test "$TORCH_FLAVOR" = cu118 \
    && pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt -c constraints-cu118.txt \
    && pip check \
    && python -c "import torch, torchvision; assert torch.__version__ == '2.7.1+cu118'; assert torchvision.__version__ == '0.22.1+cu118'; assert torch.version.cuda == '11.8'"
ENV YOLO_AUTOINSTALL=false
RUN mkdir -p /tmp/ultralytics
COPY serve.py verify_container.py ./
EXPOSE 8080
ENTRYPOINT ["python", "/opt/program/serve.py"]
CMD ["serve"]
