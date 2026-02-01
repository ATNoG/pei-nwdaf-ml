#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    -- Create mlflow database if it doesn't exist
    SELECT 'CREATE DATABASE mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

    -- Create ml_service database if it doesn't exist
    SELECT 'CREATE DATABASE ml_service'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ml_service')\gexec

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE ml_service TO $POSTGRES_USER;
EOSQL
