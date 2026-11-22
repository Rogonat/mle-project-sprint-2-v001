set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
elif [[ -f vm-emulation/local-emulation.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source vm-emulation/local-emulation.env
  set +a
fi

: "${MLFLOW_BACKEND_STORE_URI:?Заполни MLFLOW_BACKEND_STORE_URI в .env}"
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:?Заполни MLFLOW_DEFAULT_ARTIFACT_ROOT в .env}"
: "${AWS_ACCESS_KEY_ID:?Заполни AWS_ACCESS_KEY_ID в .env}"
: "${AWS_SECRET_ACCESS_KEY:?Заполни AWS_SECRET_ACCESS_KEY в .env}"

export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-https://storage.yandexcloud.net}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ru-central1}"

docker compose -f vm-emulation/docker-compose.yml up -d --wait

exec mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT:-5000}"
