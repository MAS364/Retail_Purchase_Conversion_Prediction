# Purchase Predictor API — FastAPI + GCP Cloud Run

Production-ready ML API serving 4 trained models for retail purchase conversion prediction.

## Project Structure

```
retail_conversion_api/
├── app.py                          # FastAPI application (main entry point)
├── src/
│   ├── schemas.py                  # Pydantic request/response models
│   ├── utils.py                    # Shared constants and utilities
│   ├── predict.py                  # CLI prediction script
│   ├── train.py                    # Model training pipeline
│   ├── evaluate.py                 # Model evaluation
│   ├── feature_engineering.py      # Feature engineering
│   └── data_preprocessing.py       # Data cleaning/validation
├── models/
│   ├── decision_tree_pipeline.joblib
│   ├── naive_bayes_pipeline.joblib
│   ├── svm_pipeline.joblib
│   └── mlp_pipeline.joblib
├── tests/
│   └── test_api.py                 # API tests
├── .github/workflows/
│   └── deploy.yml                  # CI/CD pipeline
├── Dockerfile                      # Multi-stage production build
├── .dockerignore
└── requirements.txt
```

## Why FastAPI over Flask

| Aspect | Flask (original) | FastAPI (new) |
|---|---|---|
| Request validation | Manual `if` checks | Automatic via Pydantic |
| API docs | None (must build manually) | Auto-generated Swagger + ReDoc |
| Error responses | Custom `jsonify` | Structured 422 with field details |
| Async support | Limited | Native |
| Type safety | None | Full Python type hints |
| Performance | WSGI | ASGI (2-3× faster) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirects to Swagger docs |
| `GET` | `/health` | Health check (Cloud Run) |
| `GET` | `/models` | List available models |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch (up to 1000) |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app:app --reload --port 8080

# Run tests
pip install pytest httpx
pytest tests/ -v

# Open docs
open http://localhost:8080/docs
```

## Docker Build & Run

```bash
# Build
docker build -t purchase-predictor-api .

# Run
docker run -p 8080:8080 purchase-predictor-api

# Test
curl http://localhost:8080/health
```

## GCP Cloud Run Deployment (Manual)

### 1. Prerequisites

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### 3. Build & Push Container

```bash
PROJECT_ID=$(gcloud config get-value project)
IMAGE=gcr.io/$PROJECT_ID/purchase-predictor-api

# Build with Cloud Build (no local Docker needed)
gcloud builds submit --tag $IMAGE

# OR build locally and push
docker build -t $IMAGE .
docker push $IMAGE
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy purchase-predictor-api \
  --image $IMAGE \
  --region europe-west2 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 120 \
  --set-env-vars MODEL_NAME=mlp \
  --port 8080
```

### 5. Get Your URL

```bash
gcloud run services describe purchase-predictor-api \
  --region europe-west2 \
  --format 'value(status.url)'
```

## CI/CD Pipeline (GitHub Actions)

The `.github/workflows/deploy.yml` runs automatically on push to `main`:

1. **Test** — runs `pytest` on every push and PR
2. **Deploy** — builds Docker image, pushes to GCR, deploys to Cloud Run (main only)

### Setup

Add these GitHub Secrets (Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Base64-encoded service account JSON key |
| `GCP_REGION` | `europe-west2` (or your preferred region) |

Create the service account:

```bash
# Create SA
gcloud iam service-accounts create github-deploy \
  --display-name "GitHub Actions Deploy"

# Grant roles
PROJECT_ID=$(gcloud config get-value project)
SA=github-deploy@$PROJECT_ID.iam.gserviceaccount.com

for role in run.admin cloudbuild.builds.builder storage.admin iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member "serviceAccount:$SA" --role "roles/$role"
done

# Create and download key
gcloud iam service-accounts keys create key.json --iam-account $SA
cat key.json | base64  # paste this as GCP_SA_KEY secret
rm key.json
```

## Monitoring & Logging

Cloud Run automatically integrates with Google Cloud Operations:

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=purchase-predictor-api" \
  --limit 50 --format json

# Set up an uptime check (optional)
gcloud monitoring uptime-checks create http purchase-predictor-uptime \
  --resource-type cloud-run-revision \
  --host YOUR_SERVICE_URL \
  --path /health
```

The API logs every request with method, path, status code, and latency via the built-in middleware.

## Security Best Practices

The deployment already implements these:

- **Non-root container** — Dockerfile runs as `appuser`, not root
- **Multi-stage build** — build dependencies don't ship in production
- **Input validation** — Pydantic enforces types, ranges, and required fields
- **CORS middleware** — configured (restrict `allow_origins` for production)
- **No secrets in code** — config via environment variables
- **Global exception handler** — internal errors don't leak stack traces

To further harden for production:

```bash
# Require authentication (remove --allow-unauthenticated)
gcloud run deploy purchase-predictor-api \
  --no-allow-unauthenticated ...

# Add IAM invoker role to specific users/services
gcloud run services add-iam-policy-binding purchase-predictor-api \
  --region europe-west2 \
  --member "user:you@example.com" \
  --role "roles/run.invoker"
```

## Estimated Costs & Scaling

Cloud Run bills per-request with generous free tier:

| Resource | Free Tier (monthly) | Beyond Free Tier |
|---|---|---|
| Requests | 2 million | $0.40 per million |
| CPU | 180,000 vCPU-seconds | $0.00002400/vCPU-sec |
| Memory | 360,000 GiB-seconds | $0.00000250/GiB-sec |
| Networking | 1 GB egress | $0.12/GB |

With `min-instances: 0`, you pay nothing when idle. A typical portfolio project with moderate traffic stays well within free tier. At ~1000 predictions/day, expect roughly **$0–2/month**.

Scaling behaviour: Cloud Run auto-scales 0→10 instances based on concurrent requests (default 80 per instance). Cold starts take ~3-5 seconds for this image size; set `min-instances: 1` (~$5/month) to eliminate them.

## Quick Test

```bash
curl -X POST https://YOUR-URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 150, "total_time_spent": 120,
    "session_length": 6, "interaction_count": 6,
    "view_count": 3, "click_count": 2,
    "wishlist_count": 0, "add_to_cart_count": 1,
    "avg_time_per_interaction": 20.0,
    "cart_to_view_ratio": 0.33, "click_to_view_ratio": 0.67,
    "has_cart_action": 1, "has_wishlist_action": 0,
    "hour": 14, "day_of_week": 2, "month": 3, "is_weekend": 0,
    "category": "electronics", "brand": "apple",
    "channel": "web", "device_type": "desktop",
    "region": "uk", "traffic_source": "organic"
  }'
```
