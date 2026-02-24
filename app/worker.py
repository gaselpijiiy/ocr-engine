import os
import json
import pika
from app.services.ocr.ocr_service import OCRService

print("🚀 WORKER STARTED")

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "ocr_queue"
RESULT_DIR = "data/results"

os.makedirs(RESULT_DIR, exist_ok=True)

# Load OCR model saat worker start
ocr_service = OCRService()


def callback(ch, method, properties, body):
    try:
        print("🔥 CALLBACK TRIGGERED")

        data = json.loads(body)
        job_id = data["job_id"]
        file_path = os.path.abspath(data["image_path"])

        print(f"[WORKER] Processing job {job_id}")

        # Jalankan Smart OCR
        result = ocr_service.extract(file_path, job_id)

        # Simpan hasil
        result_path = os.path.join(RESULT_DIR, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[WORKER] Job {job_id} completed")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"[WORKER ERROR] {str(e)}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_worker():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

    print("📡 Subscribed to queue:", QUEUE_NAME)
    print("[WORKER] Waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    start_worker()