# NWDAF ML Service

FastAPI service for ML model lifecycle management and performance monitoring on 5G network data.

## How It Works

1. Models are registered with a configuration (architecture, input/output fields, window size, lookback/forecast steps)
2. Training jobs fetch historical windows from the Data Storage API, build sliding sequences and train a PyTorch model, logging results to MLflow
3. Inference fetches the latest cell data from ClickHouse and runs the elected best model for that field
4. A background monitoring loop re-scores the best model per output field on a configurable interval; if performance degrades past a threshold, all models for that field are retrained and re-evaluated automatically

## Databases & Integrations

| Service | Role |
|---|---|
| **MLflow** | Model registry, experiment tracking, per-field performance tags |
| **PostgreSQL** | Model configs, training job log, score history |
| **MinIO** | S3-compatible artifact store for trained model weights |
| **Data Storage API** | Source of processed ClickHouse windows for training, evaluation, and inference |

## API

Base path: `/v1`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/fields` | List available output fields. `?include_model_status=true` adds `has_models` flag per field |
| `GET` | `/models` | List all models. `?include_details=true` adds scores, best-for fields, training status |
| `POST` | `/models` | Create a new model config |
| `GET` | `/models/{model_id}` | Get model detail |
| `DELETE` | `/models/{model_id}` | Delete model from registry and config store |
| `POST` | `/training/train` | Queue a training job (202 Accepted) |
| `GET` | `/training/jobs` | List training jobs. `?model_id=` / `?status=` filters |
| `GET` | `/training/jobs/{job_id}` | Get job detail (status, timestamps, error) |
| `DELETE` | `/training/jobs/{job_id}` | Cancel a job |
| `POST` | `/inference` | Run inference for a cell. Omit `model_id` to use the best model |
| `POST` | `/performance/{field}/evaluate` | Score all models for a field, elect best. `?metric=rmse\|mae\|mape\|r2` |
| `GET` | `/performance/{field}` | Cached evaluation result (no live scoring) |
| `GET` | `/performance/{field}/best` | Current best model with baseline score and degradation threshold |
| `POST` | `/performance/{field}/set-best/{model_id}` | Override best model without re-evaluating |
| `POST` | `/performance/{field}/monitor` | Re-score best model only; does not change best designation |
| `GET` | `/performance/{field}/status` | State machine status: state, active jobs, next check time, thresholds |
| `GET` | `/performance/{field}/history` | Full score measurement history. `?model_id=` filter |

### Model Config Parameters

| Parameter | Type | Description |
|---|---|---|
| `architecture` | `ann` \| `lstm` | Model type |
| `input_fields` | `list[str]` | Fields used as input features |
| `output_fields` | `list[str]` | Fields to predict |
| `window_duration_seconds` | int | Time bucket size in seconds (e.g. 60, 300) |
| `lookback_steps` | int | Number of past windows fed as input |
| `forecast_steps` | int | Number of future windows predicted |
| `hidden_size` | int | Hidden layer width (default: 32) |

### Training Job Statuses

`queued -> running -> completed | failed | cancelled`

### Performance MLflow Tags

| Tag | Description |
|---|---|
| `best_for:{field}` | `"true"` on the elected best model |
| `baseline_score:{field}` | Score at election time - degradation reference |
| `score_for:{field}` | Latest score |
| `eval_metric:{field}` | Metric used (`rmse` / `mae` / `mape` / `r2`) |
| `eval_at:{field}` | ISO timestamp of last evaluation |

## Auto-Monitoring Loop

Runs as a background task on startup when `MONITORING_ENABLED=true`. Per-field state machine:

```
MONITORING --(degraded?)--> RETRAINING --(all jobs done)--> EVALUATING --> MONITORING
```

- **MONITORING** - re-scores the best model every `MONITORING_INTERVAL_SECONDS`. Triggers retraining if `score > baseline x MONITORING_DEGRADATION_FACTOR`
- **RETRAINING** - waits for all training jobs to reach a terminal state
- **EVALUATING** - re-ranks all models and elects a new best

On startup, stale `is_training` locks from crashed runs are automatically cleared.

## Configuration

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `DATABASE_URL` | PostgreSQL connection string |
| `DATA_STORAGE_API_URL` | Data Storage service base URL |
| `DATA_STORAGE_DATA_ENDPOINT` | Processed data endpoint (e.g. `/api/v1/processed`) |
| `DATA_STORAGE_EXAMPLE_ENDPOINT` | Example schema endpoint |
| `DATA_STORAGE_CELL_ENDPOINT` | Cell list endpoint |
| `DATA_STORAGE_EXCLUDED_FIELDS` | Comma-separated metadata fields to exclude from field discovery |
| `AWS_ACCESS_KEY_ID` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key |
| `AWS_S3_ENDPOINT_URL` | MinIO endpoint URL |
| `MONITORING_ENABLED` | Enable background monitoring loop (default: `true`) |
| `MONITORING_INTERVAL_SECONDS` | Seconds between monitor checks (default: `300`) |
| `MONITORING_DEGRADATION_FACTOR` | Score multiplier that triggers retraining (default: `1.5`) |
| `API_HOST` | Bind address (default: `0.0.0.0`) |
| `API_PORT` | Port (default: `8060`) |
| `ML_PORT` | Port (default: `8060`) |
| `LOG_LEVEL` | Log verbosity (default: `INFO`) |

## Running

```bash
cp .env.example .env
docker compose up
```
