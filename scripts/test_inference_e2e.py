#!/usr/bin/env python3
"""
End-to-end inference test script

This script tests the complete inference pipeline by:
1. Setting up MLflow connection and registering a test model
2. Sending inference requests via the API
3. Validating the responses

Usage (inside Docker container - RECOMMENDED):
    docker exec -it pei-ml python scripts/test_inference_e2e.py
    docker exec -it pei-ml python scripts/test_inference_e2e.py --cells 12898855 12898856 --batch

Usage (outside Docker - requires proper port mappings):
    python scripts/test_inference_e2e.py --external
    python scripts/test_inference_e2e.py --external --cells 12898855 12898856 --batch

Prerequisites:
    - Docker compose is up and running
    - MLflow is accessible
    - The ML service (pei-ml) is running
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import requests
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also add /app for when running inside container
sys.path.insert(0, '/app')

# Suppress verbose logging
import logging
logging.getLogger('mlflow').setLevel(logging.WARNING)
logging.getLogger('git.cmd').setLevel(logging.WARNING)
logging.getLogger('git.util').setLevel(logging.WARNING)

# Feature names in the order expected by the model
FEATURE_NAMES = [
    'rsrp_mean', 'rsrp_max', 'rsrp_min', 'rsrp_std',
    'sinr_mean', 'sinr_max', 'sinr_min', 'sinr_std',
    'rsrq_mean', 'rsrq_max', 'rsrq_min', 'rsrq_std',
    'cqi_mean', 'cqi_max', 'cqi_min', 'cqi_std'
]


def is_running_in_docker():
    """Check if we're running inside a Docker container"""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)


def get_default_urls(external: bool = False):
    """Get default URLs based on environment"""
    if external or not is_running_in_docker():
        # Running outside Docker - use localhost with mapped ports
        return {
            'api_url': os.getenv('API_URL', 'http://localhost:8060'),
            'mlflow_url': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'minio_url': os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000'),
        }
    else:
        # Running inside Docker - use container names
        return {
            'api_url': os.getenv('API_URL', 'http://localhost:8060'),
            'mlflow_url': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
            'minio_url': os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
        }


def setup_mlflow(mlflow_url: str, minio_url: str):
    """Setup MLflow connection with MinIO artifact store"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Configure S3/MinIO credentials
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = minio_url
    os.environ.setdefault('AWS_ACCESS_KEY_ID', os.getenv('MINIO_ROOT_USER', 'minio'))
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', os.getenv('MINIO_ROOT_PASSWORD', 'minio123'))
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("inference_e2e_test")
    
    print(f"[+] Connected to MLflow at {mlflow_url}")
    print(f"[+] Using MinIO at {minio_url}")
    return MlflowClient()


def create_xgboost_model():
    """Create a simple XGBoost classifier for testing"""
    import xgboost as xgb
    
    # Generate dummy training data matching expected features
    # 16 features: rsrp (4), sinr (4), rsrq (4), cqi (4)
    np.random.seed(42)
    X = np.random.rand(200, 16)
    y = np.random.randint(0, 2, 200)
    
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X, y)
    
    return model


def create_randomforest_model():
    """Create a simple Random Forest classifier for testing"""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X = np.random.rand(200, 16)
    y = np.random.randint(0, 2, 200)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model


def register_cell_model(client, cell_index: str, model_type: str = 'xgboost', force: bool = False):
    """Register a model for a specific cell in MLflow"""
    import mlflow
    import mlflow.sklearn
    
    cell_str = str(cell_index).replace('.', '_')
    model_name = f"cell_{cell_str}_{model_type}"
    
    # Check if model already exists
    if not force:
        try:
            client.get_registered_model(model_name)
            print(f"[*] Model {model_name} already exists, skipping registration")
            return model_name
        except Exception:
            pass
    
    print(f"[+] Creating {model_type} model for cell {cell_index}...")
    
    # Create appropriate model
    if model_type == 'xgboost':
        model = create_xgboost_model()
    else:
        model = create_randomforest_model()
    
    # Log model to MLflow
    with mlflow.start_run(run_name=f"{model_name}_e2e_test") as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.85 + np.random.uniform(-0.05, 0.05))
        mlflow.log_metric("f1_score", 0.82 + np.random.uniform(-0.05, 0.05))
        mlflow.log_metric("precision", 0.84 + np.random.uniform(-0.05, 0.05))
        mlflow.log_metric("recall", 0.81 + np.random.uniform(-0.05, 0.05))
        
        # Set tags
        mlflow.set_tag("cell_index", str(cell_index))
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("test_model", "true")
        mlflow.set_tag("created_by", "test_inference_e2e.py")
        
        print(f"    Run ID: {run.info.run_id}")
    
    # Transition to Production stage
    try:
        # Wait for model version to be available
        time.sleep(1)
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            print(f"    Transitioned {model_name} v{version} to Production")
    except Exception as e:
        print(f"    Warning: Could not transition to Production: {e}")
    
    return model_name


def generate_test_data():
    """
    Generate sample network metrics data for inference.
    Returns data as a dictionary (as the API schema expects).
    The inference layer will convert this to a DataFrame for the model.
    """
    return {
        'rsrp_mean': -86.0 + np.random.uniform(-5, 5),
        'rsrp_max': -81.0 + np.random.uniform(-3, 3),
        'rsrp_min': -92.0 + np.random.uniform(-3, 3),
        'rsrp_std': 3.27 + np.random.uniform(-0.5, 0.5),
        'sinr_mean': 1.88 + np.random.uniform(-1, 1),
        'sinr_max': 6.0 + np.random.uniform(-1, 1),
        'sinr_min': -2.0 + np.random.uniform(-1, 1),
        'sinr_std': 2.52 + np.random.uniform(-0.3, 0.3),
        'rsrq_mean': -11.15 + np.random.uniform(-2, 2),
        'rsrq_max': -8.0 + np.random.uniform(-1, 1),
        'rsrq_min': -14.0 + np.random.uniform(-1, 1),
        'rsrq_std': 1.65 + np.random.uniform(-0.2, 0.2),
        'cqi_mean': 6.16 + np.random.uniform(-1, 1),
        'cqi_max': 9.0 + np.random.uniform(-1, 1),
        'cqi_min': 3.0 + np.random.uniform(-1, 1),
        'cqi_std': 1.60 + np.random.uniform(-0.2, 0.2)
    }


def test_health_check(api_url: str) -> bool:
    """Check if the ML service is healthy"""
    print("\n[*] Checking service health...")
    
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        
        print(f"    Status: {health.get('status')}")
        print(f"    MLflow connected: {health.get('mlflow_connected')}")
        print(f"    Kafka connected: {health.get('kafka_connected')}")
        
        if health.get('status') != 'healthy':
            print("    [!] Service is not fully healthy, but continuing...")
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"    [!] Cannot connect to API at {api_url}")
        print("    Make sure the ML service is running (docker compose up)")
        return False
    except Exception as e:
        print(f"    [!] Health check failed: {e}")
        return False


def test_single_cell_inference(api_url: str, cell_index: str, model_type: str = 'xgboost') -> bool:
    """Test inference for a single cell"""
    print(f"\n[*] Testing single cell inference for cell {cell_index}...")
    
    # Generate data as dict (API schema format)
    data = generate_test_data()
    
    payload = {
        "data": data,
        "cell_index": cell_index,
        "model_type": model_type,
        "publish_result": False
    }
    
    print(f"    Sending data with {len(data)} features")
    
    try:
        response = requests.post(
            f"{api_url}/ml/inference",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    Status: {result.get('status')}")
            print(f"    Model used: {result.get('model_used')}")
            print(f"    Prediction: {result.get('predictions')}")
            return True
        else:
            print(f"    [!] Inference failed with status {response.status_code}")
            print(f"    Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"    [!] Inference request failed: {e}")
        return False


def test_batch_inference(api_url: str, cell_indices: list, model_type: str = 'xgboost') -> bool:
    """Test batch inference for multiple cells"""
    print(f"\n[*] Testing batch inference for {len(cell_indices)} cells...")
    
    # Generate data for each cell as list of dicts (API schema format)
    data_list = [generate_test_data() for _ in cell_indices]
    
    payload = {
        "data": data_list,
        "cell_indices": cell_indices,
        "model_type": model_type,
        "publish_result": False
    }
    
    print(f"    Sending {len(data_list)} samples, each with {len(data_list[0])} features")
    
    try:
        response = requests.post(
            f"{api_url}/ml/inference",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    Status: {result.get('status')}")
            print(f"    Models used: {json.dumps(result.get('model_used'), indent=6)}")
            predictions = result.get('predictions')
            if isinstance(predictions, dict):
                for cell, pred in predictions.items():
                    print(f"    Cell {cell}: {pred}")
            else:
                print(f"    Predictions: {predictions}")
            return True
        else:
            print(f"    [!] Batch inference failed with status {response.status_code}")
            print(f"    Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"    [!] Batch inference request failed: {e}")
        return False


def test_list_models(api_url: str) -> bool:
    """Test listing registered models"""
    print("\n[*] Listing registered models...")
    
    try:
        response = requests.get(f"{api_url}/ml/models", timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"    Found {len(models)} registered models:")
            for model in models[:10]:  # Show first 10
                print(f"      - {model.get('name')}")
            if len(models) > 10:
                print(f"      ... and {len(models) - 10} more")
            return True
        else:
            print(f"    [!] Failed to list models: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"    [!] List models request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end inference test script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (inside Docker container - RECOMMENDED):
    docker exec -it pei-ml python scripts/test_inference_e2e.py
    docker exec -it pei-ml python scripts/test_inference_e2e.py --cells 12898855 12898856 --batch

Examples (outside Docker):
    python scripts/test_inference_e2e.py --external
    python scripts/test_inference_e2e.py --external --cells 12898855 12898856 --batch
        """
    )
    
    parser.add_argument(
        '--external',
        action='store_true',
        help='Running outside Docker (uses localhost URLs)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='ML service API URL (auto-detected if not provided)'
    )
    parser.add_argument(
        '--mlflow-url',
        type=str,
        default=None,
        help='MLflow tracking URI (auto-detected if not provided)'
    )
    parser.add_argument(
        '--minio-url',
        type=str,
        default=None,
        help='MinIO endpoint URL (auto-detected if not provided)'
    )
    parser.add_argument(
        '--cells',
        type=str,
        nargs='+',
        default=['12898855'],
        help='Cell IDs to test (default: 12898855)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'randomforest'],
        help='Model type to use (default: xgboost)'
    )
    parser.add_argument(
        '--skip-register',
        action='store_true',
        help='Skip model registration (assume models exist)'
    )
    parser.add_argument(
        '--force-register',
        action='store_true',
        help='Force re-registration of models even if they exist'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Test batch inference (requires multiple cells)'
    )
    parser.add_argument(
        '--skip-health',
        action='store_true',
        help='Skip health check'
    )
    
    args = parser.parse_args()
    
    # Get default URLs based on environment
    defaults = get_default_urls(args.external)
    api_url = args.api_url or defaults['api_url']
    mlflow_url = args.mlflow_url or defaults['mlflow_url']
    minio_url = args.minio_url or defaults['minio_url']
    
    in_docker = is_running_in_docker()
    
    print("=" * 60)
    print("  End-to-End Inference Test")
    print("=" * 60)
    print(f"  Environment: {'Docker container' if in_docker else 'Host machine'}")
    print(f"  API URL:     {api_url}")
    print(f"  MLflow URL:  {mlflow_url}")
    print(f"  MinIO URL:   {minio_url}")
    print(f"  Cells:       {args.cells}")
    print(f"  Model Type:  {args.model_type}")
    print("=" * 60)
    
    if not in_docker and not args.external:
        print("\n[!] Warning: Running outside Docker without --external flag.")
        print("    If you encounter connection issues, try:")
        print("    1. Run inside container: docker exec -it pei-ml python scripts/test_inference_e2e.py")
        print("    2. Or use --external flag: python scripts/test_inference_e2e.py --external")
        print("")
    
    results = {
        'health_check': None,
        'model_registration': None,
        'single_inference': None,
        'batch_inference': None,
        'list_models': None
    }
    
    # Health check
    if not args.skip_health:
        results['health_check'] = test_health_check(api_url)
        if not results['health_check']:
            print("\n[!] Health check failed. Exiting.")
            sys.exit(1)
    
    # Model registration
    if not args.skip_register:
        print("\n[*] Registering models in MLflow...")
        try:
            client = setup_mlflow(mlflow_url, minio_url)
            
            for cell in args.cells:
                register_cell_model(
                    client,
                    cell,
                    args.model_type,
                    force=args.force_register
                )
            
            results['model_registration'] = True
            print("[+] Model registration complete")
            
            # Give MLflow a moment to sync
            time.sleep(2)
            
        except Exception as e:
            print(f"[!] Model registration failed: {e}")
            results['model_registration'] = False
            import traceback
            traceback.print_exc()
    else:
        print("\n[*] Skipping model registration")
    
    # Test listing models
    results['list_models'] = test_list_models(api_url)
    
    # Single cell inference tests
    print("\n" + "-" * 60)
    print("  Single Cell Inference Tests")
    print("-" * 60)
    
    single_results = []
    for cell in args.cells:
        success = test_single_cell_inference(api_url, cell, args.model_type)
        single_results.append(success)
    
    results['single_inference'] = all(single_results)
    
    # Batch inference test
    if args.batch and len(args.cells) > 1:
        print("\n" + "-" * 60)
        print("  Batch Inference Test")
        print("-" * 60)
        results['batch_inference'] = test_batch_inference(
            api_url,
            args.cells,
            args.model_type
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED ✓"
        else:
            status = "FAILED ✗"
            all_passed = False
        
        print(f"  {test_name.replace('_', ' ').title():.<40} {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n[+] All tests passed!")
        sys.exit(0)
    else:
        print("\n[!] Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()