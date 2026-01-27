from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging
import asyncio
import json
import time

logger = logging.getLogger(__name__)

router = APIRouter()


class TrainingStatusManager:
    """Manages WebSocket connections and training status updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.training_status: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        # Model-specific locks to prevent concurrent training of the same model
        self._model_locks: Dict[str, asyncio.Lock] = {}
        # Track which models are currently training
        self._active_trainings: Set[str] = set()

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

        # Send current status to new connection
        if self.training_status:
            try:
                await websocket.send_json({
                    "type": "initial_status",
                    "data": self.training_status
                })
            except Exception as e:
                logger.error(f"Error sending initial status: {e}")

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def update_training_status(self, model_name: str, status: dict):
        """Update training status for a model and broadcast to all connections"""
        async with self._lock:
            self.training_status[model_name] = status

        message = {
            "type": "training_update",
            "model_name": model_name,
            "data": status
        }

        await self._broadcast(message)

    async def _broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                self.active_connections -= disconnected

    def get_status(self, model_name: str = None) -> dict:
        """Get training status for a specific model or all models"""
        if model_name:
            return self.training_status.get(model_name, {})
        return self.training_status

    def is_training(self, model_name: str) -> bool:
        """Check if a model is currently being trained"""
        return model_name in self._active_trainings

    async def acquire_training_lock(self, model_name: str) -> bool:
        """
        Attempt to acquire a training lock for a specific model.

        Args:
            model_name: Name of the model to train

        Returns:
            True if lock was acquired (training can proceed),
            False if model is already being trained
        """
        async with self._lock:
            if model_name in self._active_trainings:
                return False

            # Get or create model-specific lock
            if model_name not in self._model_locks:
                self._model_locks[model_name] = asyncio.Lock()

            # Mark as training
            self._active_trainings.add(model_name)
            return True

    async def release_training_lock(self, model_name: str):
        """
        Release the training lock for a model.

        Args:
            model_name: Name of the model that finished training
        """
        async with self._lock:
            self._active_trainings.discard(model_name)
            # Clean up lock after a reasonable time to prevent memory leaks
            # We keep the lock dict entry in case another training starts soon

    async def get_model_lock(self, model_name: str) -> asyncio.Lock:
        """
        Get the lock object for a specific model.
        The caller must have already called acquire_training_lock successfully.

        Args:
            model_name: Name of the model

        Returns:
            The asyncio.Lock for this model
        """
        async with self._lock:
            if model_name not in self._model_locks:
                self._model_locks[model_name] = asyncio.Lock()
            return self._model_locks[model_name]


# Global instance
training_status_manager = TrainingStatusManager()


@router.websocket("/training/status")
async def websocket_training_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training status updates.

    Clients connect to this endpoint to receive live updates about model training progress.

    Message format:
    - Initial connection: {"type": "initial_status", "data": {model_name: status, ...}}
    - Training updates: {"type": "training_update", "model_name": str, "data": {...}}

    Status data includes:
    - current_epoch: Current training epoch
    - total_epochs: Total epochs to train
    - loss: Current training loss (optional)
    - status: "training", "completed", "error"
    - message: Status message (optional)
    """
    await training_status_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()

            # Handle ping/pong or status requests
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "get_status":
                    model_name = message.get("model_name")
                    status = training_status_manager.get_status(model_name)
                    await websocket.send_json({
                        "type": "status_response",
                        "data": status
                    })
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        await training_status_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await training_status_manager.disconnect(websocket)


def get_training_status_manager() -> TrainingStatusManager:
    """Get the global training status manager instance"""
    return training_status_manager


class PerformanceStatusManager:
    """Manages WebSocket connections and runtime-persistent performance updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.performance_status: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

        # Send all past info immediately
        await websocket.send_json({
            "type": "initial_status",
            "data": self.performance_status
        })

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)

        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def update_performance_status(
        self,
        window_size: int,
        cell_index: int,
        mse: float,
        timestamp: int | None = None
    ):
        """Persist performance info in memory and broadcast, keeping only last 50 entries"""

        if timestamp is None:
            timestamp = int(time.time())

        key = f"cell_{cell_index}_window_{window_size}"

        async with self._lock:
            if key not in self.performance_status:
                self.performance_status[key] = {
                    "cell_index": cell_index,
                    "window_size": window_size,
                    "history": [],
                    "latest_mse": None,
                }

            status = self.performance_status[key]
            status["latest_mse"] = mse
            status["history"].append({
                "timestamp": timestamp,
                "mse": mse
            })

            # Keep only the last 50 entries
            if len(status["history"]) > 50:
                status["history"] = status["history"][-50:]

        message = {
            "type": "performance_update",
            "model_key": key,
            "data": status
        }

        await self._broadcast(message)


    async def _broadcast(self, message: dict):
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        if disconnected:
            async with self._lock:
                self.active_connections -= disconnected

    def get_status(self, model_key: str | None = None) -> dict:
        if model_key:
            return self.performance_status.get(model_key, {})
        return self.performance_status

# Global instance
performance_status_manager = PerformanceStatusManager()

@router.websocket("/performance/status")
async def websocket_performance_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time performance status updates.

    Message format:
    - Initial connection:
        {"type": "initial_status", "data": {model_key: status, ...}}

    - Performance updates:
        {
          "type": "performance_update",
          "model_key": "cell_<cell>_window_<window>",
          "data": {
              "cell_index": int,
              "window_size": int,
              "latest_mse": float,
              "history": [
                  {"timestamp": int, "mse": float}
              ]
          }
        }
    """

    await performance_status_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif message.get("type") == "get_status":
                    model_key = message.get("model_key")
                    status = performance_status_manager.get_status(model_key)
                    await websocket.send_json({
                        "type": "status_response",
                        "data": status
                    })

                elif message.get("type") == "clear_all":
                    # Optional: allow clients to reset runtime persistence
                    async with performance_status_manager._lock:
                        performance_status_manager.performance_status.clear()

                    await performance_status_manager._broadcast({
                        "type": "clear_all",
                        "message": "All performance histories cleared"
                    })

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        await performance_status_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Performance WebSocket error: {e}")
        await performance_status_manager.disconnect(websocket)


def get_performance_status_manager() -> PerformanceStatusManager:
    """Get the global training status manager instance"""
    return performance_status_manager

def performance_monitor_callback(
    window_size: int,
    cell_index: int,
    mse: float
):
    """
    Callback used by PerformanceMonitor.eval
    Pushes performance updates to the PerformanceStatusManager
    """
    try:
        asyncio.get_running_loop()
        asyncio.create_task(
            performance_status_manager.update_performance_status(
                window_size=window_size,
                cell_index=cell_index,
                mse=mse
            )
        )
    except RuntimeError:
        # No running loop (e.g. sync context)
        asyncio.run(
            performance_status_manager.update_performance_status(
                window_size=window_size,
                cell_index=cell_index,
                mse=mse
            )
        )
