#!/bin/bash
#
# Test Kafka-based inference for the NWDAF ML Service
#
# This script tests the end-to-end Kafka inference flow:
# 1. Produces an inference request to ml.inference.request topic
# 2. Waits for the result on ml.inference.complete topic
# 3. Displays the inference result
#
# Usage:
#   ./scripts/test_kafka_inference.sh                           # Default test
#   ./scripts/test_kafka_inference.sh --cell 12898855          # Specific cell
#   ./scripts/test_kafka_inference.sh --cells 12898855,12898856 # Batch inference
#   ./scripts/test_kafka_inference.sh --help                    # Show help
#
# Prerequisites:
#   - Docker compose is up and running
#   - Kafka is accessible
#   - Models are registered in MLflow (use test_inference_e2e.py first)
#

set -e

# Configuration
KAFKA_CONTAINER="kafka"
ML_CONTAINER="pei-ml"
REQUEST_TOPIC="ml.inference.request"
RESPONSE_TOPIC="ml.inference.complete"
KAFKA_BOOTSTRAP="localhost:9092"
KAFKA_BIN="/opt/kafka/bin"

# Default values
CELL_INDEX="12898855"
CELL_INDICES=""
MODEL_TYPE="xgboost"
TIMEOUT=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test Kafka-based inference for the NWDAF ML Service"
    echo ""
    echo "Options:"
    echo "  --cell CELL_ID       Single cell ID for inference (default: 12898855)"
    echo "  --cells CELL_IDS     Comma-separated cell IDs for batch inference"
    echo "  --model-type TYPE    Model type: xgboost or randomforest (default: xgboost)"
    echo "  --timeout SECONDS    Timeout waiting for response (default: 30)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Test with default cell"
    echo "  $0 --cell 12898857                   # Test specific cell"
    echo "  $0 --cells 12898855,12898856,12898857  # Batch inference"
    echo "  $0 --model-type randomforest         # Use RandomForest model"
    echo ""
    echo "Prerequisites:"
    echo "  1. Docker compose must be running"
    echo "  2. Models must be registered (run test_inference_e2e.py first)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cell)
            CELL_INDEX="$2"
            shift 2
            ;;
        --cells)
            CELL_INDICES="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  NWDAF Kafka Inference Test${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if ML container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${ML_CONTAINER}$"; then
    echo -e "${RED}[!] Container '${ML_CONTAINER}' is not running.${NC}"
    echo "Please start the services first: docker compose up -d"
    exit 1
fi

# Check if Kafka container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${KAFKA_CONTAINER}$"; then
    echo -e "${RED}[!] Container '${KAFKA_CONTAINER}' is not running.${NC}"
    echo "Please start Kafka first"
    exit 1
fi

# Ensure required Kafka topics exist
ensure_topics() {
    echo -e "${YELLOW}[*] Checking Kafka topics...${NC}"

    local existing_topics=$(docker exec ${KAFKA_CONTAINER} ${KAFKA_BIN}/kafka-topics.sh --list --bootstrap-server ${KAFKA_BOOTSTRAP} 2>/dev/null)

    for topic in "$REQUEST_TOPIC" "$RESPONSE_TOPIC"; do
        if echo "$existing_topics" | grep -q "^${topic}$"; then
            echo -e "${GREEN}    ✓ Topic '${topic}' exists${NC}"
        else
            echo -e "${YELLOW}    Creating topic '${topic}'...${NC}"
            docker exec ${KAFKA_CONTAINER} ${KAFKA_BIN}/kafka-topics.sh \
                --create --topic "$topic" \
                --bootstrap-server ${KAFKA_BOOTSTRAP} \
                --partitions 1 --replication-factor 1 2>/dev/null || true
            echo -e "${GREEN}    ✓ Topic '${topic}' created${NC}"
        fi
    done
}

ensure_topics

# Generate a unique request ID
REQUEST_ID="test-$(date +%s)-$$"

# Generate sample data with some randomness
generate_data() {
    python3 -c "
import json
import random

data = {
    'rsrp_mean': -86.0 + random.uniform(-5, 5),
    'rsrp_max': -81.0 + random.uniform(-3, 3),
    'rsrp_min': -92.0 + random.uniform(-3, 3),
    'rsrp_std': 3.27 + random.uniform(-0.5, 0.5),
    'sinr_mean': 1.88 + random.uniform(-1, 1),
    'sinr_max': 6.0 + random.uniform(-1, 1),
    'sinr_min': -2.0 + random.uniform(-1, 1),
    'sinr_std': 2.52 + random.uniform(-0.3, 0.3),
    'rsrq_mean': -11.15 + random.uniform(-2, 2),
    'rsrq_max': -8.0 + random.uniform(-1, 1),
    'rsrq_min': -14.0 + random.uniform(-1, 1),
    'rsrq_std': 1.65 + random.uniform(-0.2, 0.2),
    'cqi_mean': 6.16 + random.uniform(-1, 1),
    'cqi_max': 9.0 + random.uniform(-1, 1),
    'cqi_min': 3.0 + random.uniform(-1, 1),
    'cqi_std': 1.60 + random.uniform(-0.2, 0.2)
}
print(json.dumps(data))
"
}

# Build the request message
build_request() {
    local data=$(generate_data)

    if [ -n "$CELL_INDICES" ]; then
        # Batch inference - generate data for each cell
        IFS=',' read -ra CELLS <<< "$CELL_INDICES"
        local data_list="["
        local first=true
        for cell in "${CELLS[@]}"; do
            if [ "$first" = true ]; then
                first=false
            else
                data_list+=","
            fi
            data_list+=$(generate_data)
        done
        data_list+="]"

        # Build cell_indices array
        local cells_json=$(python3 -c "import json; print(json.dumps('$CELL_INDICES'.split(',')))")

        echo "{\"request_id\": \"$REQUEST_ID\", \"cell_indices\": $cells_json, \"model_type\": \"$MODEL_TYPE\", \"data\": $data_list}"
    else
        # Single cell inference
        echo "{\"request_id\": \"$REQUEST_ID\", \"cell_index\": \"$CELL_INDEX\", \"model_type\": \"$MODEL_TYPE\", \"data\": $data}"
    fi
}

# Send message via the ML service API (which uses the Kafka producer)
send_via_api() {
    local message="$1"

    echo -e "${YELLOW}[*] Sending inference request via Kafka...${NC}"
    echo -e "${BLUE}    Topic: ${REQUEST_TOPIC}${NC}"
    echo -e "${BLUE}    Request ID: ${REQUEST_ID}${NC}"

    if [ -n "$CELL_INDICES" ]; then
        echo -e "${BLUE}    Cells: ${CELL_INDICES}${NC}"
    else
        echo -e "${BLUE}    Cell: ${CELL_INDEX}${NC}"
    fi
    echo -e "${BLUE}    Model Type: ${MODEL_TYPE}${NC}"

    # Use the ML service's Kafka produce endpoint
    response=$(curl -s -X POST "http://localhost:8060/kafka/produce" \
        -H "Content-Type: application/json" \
        -d "{\"topic\": \"$REQUEST_TOPIC\", \"message\": $(echo "$message" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}")

    if echo "$response" | grep -q '"status":"success"'; then
        echo -e "${GREEN}[+] Request sent successfully${NC}"
        return 0
    else
        echo -e "${RED}[!] Failed to send request: $response${NC}"
        return 1
    fi
}

# Poll for the response
wait_for_response() {
    echo -e "${YELLOW}[*] Waiting for inference result (timeout: ${TIMEOUT}s)...${NC}"

    local start_time=$(date +%s)
    local found=false

    while [ $(($(date +%s) - start_time)) -lt $TIMEOUT ]; do
        # Get messages from the response topic via API
        response=$(curl -s "http://localhost:8060/kafka/messages/${RESPONSE_TOPIC}?limit=20")

        # Check if our request ID is in the response
        if echo "$response" | grep -q "$REQUEST_ID"; then
            found=true
            break
        fi

        sleep 1
        echo -n "."
    done
    echo ""

    if [ "$found" = true ]; then
        echo -e "${GREEN}[+] Response received!${NC}"

        # Extract and display the result
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}  Inference Result${NC}"
        echo -e "${GREEN}======================================${NC}"

        # Parse and pretty-print the response
        echo "$response" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)
    messages = data.get('messages', [])

    for msg in messages:
        content = msg.get('content', '')
        if '$REQUEST_ID' in content:
            result = json.loads(content)
            print(json.dumps(result, indent=2))
            break
except Exception as e:
    print(f'Error parsing response: {e}')
    sys.exit(1)
"
        return 0
    else
        echo -e "${RED}[!] Timeout waiting for response${NC}"
        return 1
    fi
}

# Alternative: Direct test via API (for comparison)
test_via_api() {
    echo -e "${YELLOW}[*] Testing direct API inference for comparison...${NC}"

    local data=$(generate_data)

    if [ -n "$CELL_INDICES" ]; then
        IFS=',' read -ra CELLS <<< "$CELL_INDICES"
        local data_list="["
        local first=true
        for cell in "${CELLS[@]}"; do
            if [ "$first" = true ]; then
                first=false
            else
                data_list+=","
            fi
            data_list+=$(generate_data)
        done
        data_list+="]"

        local cells_json=$(python3 -c "import json; print(json.dumps('$CELL_INDICES'.split(',')))")

        response=$(curl -s -X POST "http://localhost:8060/ml/inference" \
            -H "Content-Type: application/json" \
            -d "{\"data\": $data_list, \"cell_indices\": $cells_json, \"model_type\": \"$MODEL_TYPE\", \"publish_result\": true}")
    else
        response=$(curl -s -X POST "http://localhost:8060/ml/inference" \
            -H "Content-Type: application/json" \
            -d "{\"data\": $data, \"cell_index\": \"$CELL_INDEX\", \"model_type\": \"$MODEL_TYPE\", \"publish_result\": true}")
    fi

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Direct API Result${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
}

# Main execution
main() {
    # Build the request
    REQUEST_MESSAGE=$(build_request)

    echo ""
    echo -e "${BLUE}Request payload:${NC}"
    echo "$REQUEST_MESSAGE" | python3 -m json.tool 2>/dev/null || echo "$REQUEST_MESSAGE"
    echo ""

    # Method 1: Send via Kafka and wait for response
    if send_via_api "$REQUEST_MESSAGE"; then
        # Give the ML service time to process
        sleep 2

        if wait_for_response; then
            echo ""
            echo -e "${GREEN}[+] Kafka inference test completed successfully!${NC}"
        else
            echo ""
            echo -e "${YELLOW}[*] Kafka response not found. Testing direct API...${NC}"
            test_via_api
        fi
    else
        echo -e "${RED}[!] Failed to send Kafka request${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Test Complete${NC}"
    echo -e "${GREEN}======================================${NC}"
}

main
