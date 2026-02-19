import pika
import json

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "ocr_queue"


def send_to_queue(data: dict):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME)

    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(data)
    )

    connection.close()
