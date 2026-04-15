from fastapi import FastAPI, UploadFile, File, Request, HTTPException
import shutil
import pika
import json
import os
import uuid
import asyncio
import base64
import httpx
from pathlib import Path

app = FastAPI()

RABBITMQ_HOST = "localhost"
QUEUE_NAME    = "ocr_queue"
UPLOAD_DIR    = "data/uploads"
RESULT_DIR    = "data/results"

# Timeout maksimal menunggu worker selesai (detik).
# LibreChat menunggu response sebelum timeout — 120 detik cukup untuk
# dokumen PDF 1-2 halaman dengan DPI 200.
OCR_TIMEOUT_SECONDS = 120

# Interval polling: cek apakah result sudah ada setiap N detik
POLL_INTERVAL = 1.0

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# HELPER: publish job ke RabbitMQ
# ──────────────────────────────────────────────────────────────────────

def publish_job(job_id: str, file_path: str):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps({"job_id": job_id, "image_path": file_path}),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    connection.close()


# ──────────────────────────────────────────────────────────────────────
# HELPER: polling sampai result ada atau timeout
# ──────────────────────────────────────────────────────────────────────

async def wait_for_result(job_id: str) -> dict:
    result_path = os.path.join(RESULT_DIR, f"{job_id}.json")
    elapsed = 0.0
    while elapsed < OCR_TIMEOUT_SECONDS:
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"OCR job {job_id} timed out after {OCR_TIMEOUT_SECONDS}s")


# ──────────────────────────────────────────────────────────────────────
# ENDPOINT LAMA: POST /ocr (async, untuk client internal)
# ──────────────────────────────────────────────────────────────────────

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        job_id   = str(uuid.uuid4())
        ext      = os.path.splitext(file.filename)[1]
        filename = f"{job_id}{ext}"
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        publish_job(job_id, file_path)
        return {"status": "queued", "job_id": job_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/ocr/{job_id}")
async def get_result(job_id: str):
    result_path = os.path.join(RESULT_DIR, f"{job_id}.json")
    if not os.path.exists(result_path):
        return {"status": "processing"}
    with open(result_path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# ENDPOINT BARU: POST /v1/ocr  ← yang dipanggil LibreChat
#
# Meniru format Mistral OCR API sehingga LibreChat tidak perlu tahu
# bahwa di baliknya ada PaddleOCR.
#
# Request dari LibreChat (format Mistral):
#   {
#     "model": "mistral-ocr-latest",
#     "document": {
#       "type": "document_url" | "image_url" | "document_base64" | "image_base64",
#       "document_url": "...",   // jika URL
#       "image_url": "...",      // jika URL gambar
#       "data": "..."            // jika base64
#     }
#   }
#
# Response ke LibreChat (format Mistral):
#   {
#     "pages": [{"index": 0, "markdown": "teks...", "images": [], "dimensions": {...}}],
#     "model": "paddleocr",
#     "usage_info": {"pages_processed": 1, "doc_size_bytes": 12345}
#   }
# ──────────────────────────────────────────────────────────────────────

@app.post("/v1/ocr")
async def librechat_ocr(request: Request):
    body     = await request.json()
    document = body.get("document", {})
    doc_type = document.get("type", "")

    # ── 1. Ambil bytes file dari request ──────────────────────────────
    file_bytes = b""
    ext        = ".pdf"

    if doc_type in ("document_url", "image_url"):
        url = document.get("document_url") or document.get("image_url", "")
        if not url:
            raise HTTPException(status_code=400, detail="document_url / image_url kosong")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
        # Tentukan ekstensi dari URL atau Content-Type
        if "pdf" in url.lower() or "pdf" in resp.headers.get("content-type", ""):
            ext = ".pdf"
        else:
            ext = ".jpg"

    elif doc_type in ("document_base64", "image_base64"):
        raw = document.get("data", "")
        # Hapus prefix "data:...;base64," jika ada
        if "," in raw:
            raw = raw.split(",", 1)[1]
        file_bytes = base64.b64decode(raw)
        ext = ".pdf" if doc_type == "document_base64" else ".jpg"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe dokumen tidak didukung: {doc_type}"
        )

    # ── 2. Simpan file ke uploads ─────────────────────────────────────
    job_id    = str(uuid.uuid4())
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{job_id}{ext}"))
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # ── 3. Publish ke RabbitMQ ────────────────────────────────────────
    try:
        publish_job(job_id, file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RabbitMQ error: {e}")

    # ── 4. Polling sampai worker selesai ──────────────────────────────
    try:
        result = await wait_for_result(job_id)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    # ── 5. Kembalikan dalam format Mistral OCR ────────────────────────
    return {
        "pages": [
            {
                "index":      0,
                "markdown":   result.get("text", ""),
                "images":     [],
                "dimensions": {
                    "dpi":    200,
                    "height": 1754,
                    "width":  1240,
                },
            }
        ],
        "model":      "paddleocr",
        "usage_info": {
            "pages_processed": 1,
            "doc_size_bytes":  len(file_bytes),
        },
    }