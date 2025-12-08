from pydantic import BaseModel


class KafkaMessage(BaseModel):
    """Used for producing messages to kafka"""
    topic: str
    message: str
