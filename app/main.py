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
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        # buat nama file unik
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_name))

        # simpan file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # koneksi ke RabbitMQ
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME)

        # queue
        message = {"image_path": file_path}
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message)
        )

        connection.close()

        return {
            "status": "queued",
            "filename": unique_name,
            "path": file_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
