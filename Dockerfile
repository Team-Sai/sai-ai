FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY requirements.txt ./requirements.txt

# GPU가 없는 노드에서 돌리므로 CPU 전용 torch를 먼저 설치한다.
# (기본 torch는 CUDA 라이브러리가 함께 설치되어 이미지가 수 GB 커진다)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "sai_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
