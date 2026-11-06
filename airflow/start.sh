#!/usr/bin/env bash
# Starts all Airflow components (api-server, scheduler, dag-processor, triggerer) in one container.
set -euo pipefail

# Simple auth manager login: user "admin", password from AIRFLOW_ADMIN_PASSWORD in .env
echo "{\"admin\": \"${AIRFLOW_ADMIN_PASSWORD:-admin}\"}" > /opt/airflow/passwords.json

airflow db migrate
exec airflow standalone
