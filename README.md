# pei-nwdaf-ml

> Project for PEI evaluation 25/26

## Overview

ML inference and model management service for the NWDAF platform. Provides real-time ML predictions on network data, manages the complete ML lifecycle including training, versioning, and model selection, and integrates with MLflow for experiment tracking.

## Technologies

- **FastAPI** 0.124.0 - Web framework for REST APIs
- **PyTorch** 2.9.1 - Deep learning framework
- **XGBoost** 3.1.2 - Gradient boosting for predictions
- **scikit-learn** 1.7.2 - ML utilities
- **MLflow** 3.6.0 - Model registry and experiment tracking
- **Confluent Kafka** 2.12.2 - Event streaming
- **WebSockets** 14.1 - Real-time communication
- **boto3** 1.35.0 - AWS/S3 integration
- **PostgreSQL** - MLflow backend store
- **MinIO** - S3-compatible artifact storage
- **Docker & Docker Compose** - Containerization
- **Python** 3.13

## Key Features

- **Multiple inference modes**:
  - Single cell inference for specific network cells
  - Batch inference for multiple cells simultaneously
  - Auto-selection of best-performing models
  - Manual model selection by name/version/stage

- **Model management**:
  - Integration with MLflow model registry
  - Cell-specific models (e.g., `cell_12898855_xgboost`)
  - Model caching for performance
  - Support for XGBoost, ANN, LSTM model types

- **Kafka integration**:
  - Consumes from `ml.inference.request` topic
  - Processes network data from `network.data.processed` topic
  - Publishes results to `ml.inference.complete` topic
  - Background consumer thread for async message handling

- **Data storage integration**:
  - Fetches training data from Data Storage API
  - Queries latency data with time range and cell filtering
  - Supports dynamic cell discovery

- **Monitoring & health**:
  - System health checks (Kafka, MLflow, Data Storage)
  - Component status tracking (inference, training, registry, storage)
  - Performance metrics logging
  - WebSocket support for real-time monitoring

## API Endpoints

- `/ml/inference` - Trigger predictions
- `/ml/set-model` - Configure specific models
- `/ml/auto-mode` - Toggle auto-selection
- `/ml/models` - List registered models
- `/ml/best-model` - Get best performing model
- `/health` - Service health status
- `/kafka/*` - Kafka management
- `/data/*` - Data operations
- `/api/v1/*` - v1 API routes

## Directory Structure

```
ml/
├── main.py              # FastAPI entry point
├── src/
│   ├── interface/       # MLInterface - central communication hub
│   ├── models/          # Model implementations (ANN, LSTM, XGBoost)
│   ├── inference/       # InferenceMaker for predictions
│   ├── services/        # Business logic (inference, training, config, monitoring)
│   ├── routers/         # FastAPI route handlers
│   ├── schemas/         # Pydantic data models for validation
│   ├── mlflow/          # MLflow integration bridge
│   ├── config/          # Configuration and inference types
│   └── utils/           # Feature engineering utilities
└── tests/               # Test suite
```

## Quick Start

```bash
docker-compose up
```

## Infrastructure Dependencies

Requires running services:
- **PostgreSQL** - MLflow backend
- **MinIO** - Artifact storage
- **Kafka** - Message streaming
- **Data Storage** - Training data access

## Use Cases

- Network latency prediction for 5G cells
- QoS forecasting and analytics
- Model performance evaluation on streaming data
- ML model lifecycle management for network analytics
