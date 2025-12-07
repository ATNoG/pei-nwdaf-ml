import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from src.interface.mlint import MLInterface
from src.routers.inference import router as inference_router
from src.routers.kafka import router as kafka_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
KAFKA_HOST = os.getenv("KAFKA_HOST", "localhost")
KAFKA_PORT = os.getenv("KAFKA_PORT", "9092")
KAFKA_TOPICS = os.getenv("KAFKA_TOPICS", "ml.inference.request,network.data.processed,network.data.request").split(",")

ml_interface: MLInterface = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_interface

    logger.info("Starting ML Service...")

    try:
        ml_interface = MLInterface(
            kafka_host=KAFKA_HOST,
            kafka_port=KAFKA_PORT,
        )

        app.state.ml_interface = ml_interface

        if KAFKA_TOPICS and KAFKA_TOPICS[0].strip():
            topics_to_subscribe = [t.strip() for t in KAFKA_TOPICS if t.strip()]

            for topic in topics_to_subscribe:
                ml_interface.subscribe_topic(topic)

            logger.info(f"Subscribed to topics: {topics_to_subscribe}")

        await ml_interface.start_consumer()

        logger.info("ML Service ready to accept requests")
        logger.info(f"Kafka: {KAFKA_HOST}:{KAFKA_PORT}")
        logger.info(f"Topics: {KAFKA_TOPICS}")

    except Exception as e:
        logger.error(f"Failed to initialize ML Service: {e}")

    yield

    logger.info("Shutting down ML Service...")

    if ml_interface:
        try:
            await ml_interface.shutdown()
            logger.info("ML Interface shut down")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="ML Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.include_router(kafka_router, prefix="/kafka", tags=["Kafka"])
app.include_router(inference_router, prefix="/ml", tags=["ML"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ML Service",
        "version": "1.0.0",
        "random": "hi",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns service health and Kafka connection status
    """
    ml_int = app.state.ml_interface

    if not ml_int:
        return {
            "status": "degraded",
            "kafka_connected": False,
            "subscribed_topics": [],
            "consumer_running": False
        }

    kafka_connected = ml_int.is_connected()
    consumer_running = ml_int.is_consumer_running()

    return {
        "status": "healthy" if kafka_connected else "degraded",
        "kafka_connected": kafka_connected,
        "consumer_running": consumer_running,
        "subscribed_topics": ml_int.get_subscribed_topics(),
        "kafka_broker": f"{KAFKA_HOST}:{KAFKA_PORT}"
    }


@app.get("/status", tags=["Health"])
async def detailed_status():
    ml_int = app.state.ml_interface

    if not ml_int:
        return {"error": "ML Interface not initialized"}

    return {
        "service": "ML Communication Interface",
        "kafka": {
            "connected": ml_int.is_connected(),
            "consumer_running": ml_int.is_consumer_running(),
            "broker": f"{KAFKA_HOST}:{KAFKA_PORT}",
            "subscribed_topics": ml_int.get_subscribed_topics(),
        },
        "config": {
            "auto_start_consumer": AUTO_START_CONSUMER,
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    }


if __name__ == "__main__":
    import uvicorn

    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    api_reload = os.getenv("API_RELOAD", "true").lower() == "true"

    logger.info(f"Starting server on {api_host}:{api_port}")

    uvicorn.run(
        "main:app",
        host=api_host,
        port=api_port,
        reload=api_reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
