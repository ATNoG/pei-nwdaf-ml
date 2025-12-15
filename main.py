import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from src.interface.mlint import MLInterface
from src.routers.inference import router as inference_router
from src.routers.kafka import router as kafka_router
from src.routers.data import router as data_router
from src.routers.v1 import v1_router
from src.routers.websocket import router as websocket_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
KAFKA_HOST = os.getenv("KAFKA_HOST", "localhost")
KAFKA_PORT = os.getenv("KAFKA_PORT", "9092")
KAFKA_TOPICS = os.getenv("KAFKA_TOPICS", "ml.inference.request,network.data.processed,network.data.request").split(",")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-service-default")
DATA_STORAGE_API_URL = os.getenv("DATA_STORAGE_API_URL", "http://localhost:8001")

ml_interface: MLInterface = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_interface

    logger.info("Starting ML Service...")

    try:
        ml_interface = MLInterface(
            kafka_host=KAFKA_HOST,
            kafka_port=KAFKA_PORT,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            data_storage_api_url=DATA_STORAGE_API_URL,
        )

        app.state.ml_interface = ml_interface

        # Initialize models (create missing models on startup)
        logger.info("Initializing models...")

        # Start Kafka consumer in background thread
        ml_interface.start_consumer_background()

        logger.info("ML Service ready to accept requests")
        logger.info(f"Kafka: {KAFKA_HOST}:{KAFKA_PORT}")
        logger.info(f"Topics: {KAFKA_TOPICS}")
        logger.info(f"MLFlow: {MLFLOW_TRACKING_URI}")
        logger.info(f"Data Storage API: {DATA_STORAGE_API_URL}")
        logger.info("Kafka consumer connecting in background")

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
app.include_router(data_router, prefix="/data", tags=["Data"])
app.include_router(v1_router, prefix="/api/v1", tags=["API v1"])
app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])


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
    mlflow_connected = ml_int.is_mlflow_connected()
    data_storage_connected = ml_int.check_data_storage_connection()

    return {
        "status": "healthy" if (kafka_connected and mlflow_connected and data_storage_connected) else "degraded",
        "kafka_connected": kafka_connected,
        "consumer_running": consumer_running,
        "mlflow_connected": mlflow_connected,
        "data_storage_connected": data_storage_connected,
        "subscribed_topics": ml_int.get_subscribed_topics(),
        "kafka_broker": f"{KAFKA_HOST}:{KAFKA_PORT}",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "data_storage_api": DATA_STORAGE_API_URL
    }


@app.get("/status", tags=["Health"])
async def detailed_status():
    ml_int = app.state.ml_interface

    if not ml_int:
        return {"error": "ML Interface not initialized"}

    return ml_int.get_system_health()


if __name__ == "__main__":
    import uvicorn

    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8060"))
    api_reload = os.getenv("API_RELOAD", "true").lower() == "true"

    logger.info(f"Starting server on {api_host}:{api_port}")

    uvicorn.run(
        "main:app",
        host=api_host,
        port=api_port,
        reload=api_reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
