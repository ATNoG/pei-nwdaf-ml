# Communication Interface for the ML Service component
#
# An abstraction layer which mediates messaging from
# different system components along with internal components
#
# Author: Miguel Neto

import fastapi
from kmw import PyKafBridge

import asyncio
import logging
from typing import Optional, List


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S",
                    handlers=[
                        # logging.FileHandler(f"./logs/student_{datetime.datetime.now().strftime("%d_%m_%y_at_%H_%M_%S")}.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


class MLInterface():
    def __init__(self, kafka_host: str = 'localhost', kafka_port: str = '9092'):
        self.bridge = PyKafBridge(hostname=kafka_host, port=kafka_port, debug_label='ML Interface Bridge')
        self.topics = ['ml.inference.request', 'network.data.processed', 'network.data.request']

        self.bridge.add_n_topics(self.topics)

    def is_connected(self) -> bool:
        return self.bridge is not None and self.bridge.consumer is not None

    # FIX: PyKaf should have functions to return true or false on these checks
    def is_consumer_running(self) -> bool:
        return self.bridge._running

    async def start_consumer(self) -> bool:
        """Starts the component-related consumer"""
        try:
            await self.bridge.start_consumer()
            logger.info("Started PyKaf Consumer")
            return True
        except e:
            logger.error(f"Failed to start PyKaf consumer: {e}")
            return False

    async def shutdown_bridge(self) -> None:
        if self.bridge:
            await self.bridge.close()
        logger.info("PyKaf bridge has shutdown")

    def produce_to_kafka(self, topic: str, message: str) -> bool:
        """
        Produce a message to a specific Kafka topic

        args:
            topic - target topic name
            message: message contents

        Returns True when successefully produced, otherwise False
        """

        ret = self.bridge.produce(topic, message)
        if ret:
            logger.debug(f"Produced message with success to {topic}")
        else:
            logger.error(f"Failed producing a message to {topic}")

        return ret

    def get_messages(self, topic: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve consumed messages from a topic

        Args:
            topic: Topic name
            limit: Maximum number of messages to return (None = all)

        Returns a list of messages (as dictionaries)
        """

        messages = self.bridge.get_topic(topic)
        if limit and messages:
            return messages[-limit:]

        return messages
