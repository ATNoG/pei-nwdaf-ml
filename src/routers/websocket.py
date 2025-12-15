from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging
import asyncio
import json

logger = logging.getLogger(__name__)

router = APIRouter()


class TrainingStatusManager:
    """Manages WebSocket connections and training status updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.training_status: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
    
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
