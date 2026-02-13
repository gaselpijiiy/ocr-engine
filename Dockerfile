FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY PaddleOCR /app/PaddleOCR

WORKDIR /app/PaddleOCR

RUN pip install --upgrade pip \
    && pip install paddlepaddle \
    && pip install -r requirements.txt

CMD ["bash"]
