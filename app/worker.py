import pika
import json
from app.ocr_service import extract_text

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "ocr_queue"


def callback(ch, method, properties, body):
    data = json.loads(body)
    image_path = data["image_path"]

    print(f"[WORKER] Processing: {image_path}")

    text = extract_text(image_path)

    print("[WORKER] OCR RESULT:")
    print(text)

    ch.basic_ack(delivery_tag=method.delivery_tag)


connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST)
)
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME)

channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

print("[WORKER] Waiting for messages...")
channel.start_consuming()
