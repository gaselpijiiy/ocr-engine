from fastapi import FastAPI, UploadFile, File
import shutil
import pika
import json
import os
import uuid

app = FastAPI()

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "ocr_queue"

UPLOAD_DIR = "data/uploads"
RESULT_DIR = "data/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        # buat job_id unik
        job_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{job_id}{file_extension}"

        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))

        # simpan file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # koneksi RabbitMQ
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)

        message = {
            "job_id": job_id,
            "image_path": file_path
        }

        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2  # persistent message
            )
        )

        connection.close()

        return {
            "status": "queued",
            "job_id": job_id
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/ocr/{job_id}")
async def get_result(job_id: str):
    result_path = os.path.join(RESULT_DIR, f"{job_id}.json")

    if not os.path.exists(result_path):
        return {"status": "processing"}

    with open(result_path, "r") as f:
        return json.load(f)