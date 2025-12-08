from fastapi import APIRouter, HTTPException, Request
import logging

from src.schemas.kafka import KafkaMessage

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/produce")
async def produce_message(msg: KafkaMessage, request: Request):
    """Publish a message to a Kafka topic"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    success = ml_interface.produce_to_kafka(msg.topic, msg.message)

    if success:
        return {
            "status": "success",
            "topic": msg.topic,
            "message_length": len(msg.message)
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to produce message to Kafka")


@router.get("/messages/{topic}")
async def get_messages(topic: str, request: Request, limit: int = 10):
    """Retrieve consumed messages from a specific topic"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    messages = ml_interface.get_messages(topic, limit=limit)

    return {
        "topic": topic,
        "count": len(messages),
        "messages": messages
    }


@router.get("/topics")
async def list_topics(request: Request):
    """List all subscribed topics"""
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    topics = ml_interface.get_subscribed_topics()

    return {
        "status": "success",
        "count": len(topics),
        "topics": topics,
        "consumer_running": ml_interface.is_consumer_running()
    }
