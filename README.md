# pei-nwdaf-ml

ML service for training, inference, and anomaly detection on 5G network data.

## How it works

- **Forecast models**: ANN/LSTM models trained on windowed ClickHouse data, predict future metric values
- **Anomaly models**: Autoencoder-based models detect anomalous windows via reconstruction error
- Training jobs run as Kubernetes Jobs; model weights and scalers stored in MLflow/MinIO
- A background monitoring loop re-evaluates and retrains forecast models when performance degrades

## Running

```bash
cp .env.example .env
docker compose up -d
```

Port: `8060`

## API

Base path: `/v1`

### Forecast

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/fields` | List available metric fields |
| `POST` | `/models` | Create model config |
| `GET` | `/models` | List models |
| `GET` | `/models/{id}` | Model detail |
| `DELETE` | `/models/{id}` | Delete model |
| `POST` | `/training/train` | Queue training job |
| `GET` | `/training/jobs` | List jobs |
| `GET` | `/training/jobs/{id}` | Job detail |
| `DELETE` | `/training/jobs/{id}` | Cancel job |
| `POST` | `/inference` | Run inference |
| `POST` | `/performance/{field}/evaluate` | Score and elect best model |
| `GET` | `/performance/{field}/best` | Current best model |
| `GET` | `/performance/{field}/status` | Monitoring state machine status |

### Anomaly

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/anomaly/models` | Create anomaly model config |
| `GET` | `/anomaly/models` | List anomaly models |
| `GET` | `/anomaly/models/{id}` | Model detail |
| `DELETE` | `/anomaly/models/{id}` | Delete model |
| `POST` | `/anomaly/training/train` | Queue training job |
| `GET` | `/anomaly/training/jobs` | List jobs |
| `POST` | `/anomaly/detect` | Run anomaly detection (`explain=true` for SHAP attributions) |
| `POST` | `/anomaly/models/{id}/importance` | Compute permutation feature importance |
| `GET` | `/anomaly/models/{id}/importance` | Get cached importance |

## Environment

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `DATABASE_URL` | PostgreSQL connection string |
| `DATA_STORAGE_API_URL` | Data Storage base URL |
| `TRAIN_USE_KUBE` | Run training jobs on Kubernetes (`true`/`false`) |
| `TRAIN_KUBE_HOST` | Kubernetes API server URL |
| `TRAIN_KUBE_TOKEN` | Service account token |
| `TRAIN_KUBE_NAMESPACE` | Kubernetes namespace |
| `TRAIN_KUBE_IMAGE` | Training worker container image |
| `MONITORING_ENABLED` | Enable background monitoring loop (default: `true`) |
| `MONITORING_INTERVAL_SECONDS` | Monitor check interval (default: `300`) |
| `API_PORT` | Port (default: `8060`) |
