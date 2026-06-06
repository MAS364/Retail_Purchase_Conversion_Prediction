# ⚙️ Setup & Deployment Guide

Complete instructions for running locally, testing, and deploying to production.

---

## Prerequisites

- Python 3.10+
- pip
- Git
- Docker (optional, for containerised deployment)

---

## 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/MAS364/Retail_Purchase_Conversion_Prediction.git
cd Retail_Purchase_Conversion_Prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install API dependencies
pip install -r requirements.txt

# Install Streamlit dependencies
pip install -r requirements-ui.txt
```

---

## 2. Run Locally

### Start the API (Terminal 1)

```bash
source .venv/bin/activate
uvicorn app:app --reload --port 8080
```

You should see all 4 models load:
```
Loaded model: decision_tree
Loaded model: naive_bayes
Loaded model: svm
Loaded model: mlp
Application startup complete.
```

Open http://localhost:8080/docs for Swagger UI.

### Start the Dashboard (Terminal 2)

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Opens at http://localhost:8501

### One-Command Launch (Both Services)

```bash
chmod +x run.sh
./run.sh
```

---

## 3. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Single prediction
curl -X POST "http://localhost:8080/predict?model=mlp" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 150.0, "total_time_spent": 120.0, "session_length": 6,
    "interaction_count": 6, "view_count": 3, "click_count": 2,
    "wishlist_count": 0, "add_to_cart_count": 1,
    "avg_time_per_interaction": 20.0, "cart_to_view_ratio": 0.33,
    "click_to_view_ratio": 0.67, "has_cart_action": 1, "has_wishlist_action": 0,
    "hour": 14, "day_of_week": 2, "month": 3, "is_weekend": 0,
    "category": "electronics", "brand": "apple", "channel": "web",
    "device_type": "desktop", "region": "uk", "traffic_source": "organic"
  }'

# CSV upload (raw event data — auto-detected)
curl -X POST "http://localhost:8080/predict/csv?model=mlp" \
  -F "file=@your_data.csv"
```

### Run Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 4. Docker

```bash
# Build
docker build -t purchase-predictor-api .

# Run
docker run -p 8080:8080 purchase-predictor-api

# Test
curl http://localhost:8080/health
```

---

## 5. Deploy to Render.com (Free)

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → sign up with GitHub
3. **New → Web Service** → connect your repo
4. Settings:
   - **Name:** `retail-purchase-conversion-prediction`
   - **Region:** Frankfurt
   - **Runtime:** Docker
   - **Instance Type:** Free
5. Click **Deploy**

You'll get a URL like: `https://retail-purchase-conversion-prediction.onrender.com`

---

## 6. Deploy Dashboard to Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Select `streamlit_app.py` as the main file
4. Go to **Settings → Secrets** and add:
   ```
   API_URL = "https://retail-purchase-conversion-prediction.onrender.com"
   ```
5. Deploy

---

## 7. Deploy to GCP Cloud Run (Alternative)

```bash
# Login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Deploy (builds in the cloud)
gcloud run deploy purchase-predictor-api \
  --source . \
  --region europe-west2 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 120 \
  --set-env-vars MODEL_NAME=mlp \
  --port 8080
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `mlp` | Default model when none specified |
| `PORT` | `8080` | API port |
| `API_URL` | `http://localhost:8080` | Backend URL (used by Streamlit) |

---

## Troubleshooting

### scikit-learn version mismatch warnings
The models were trained with scikit-learn 1.8.0. If you see `InconsistentVersionWarning`, install a compatible version:
```bash
pip install scikit-learn>=1.7
```

### `python-multipart` not installed
Required for CSV file upload. If you see `RuntimeError: Form data requires "python-multipart"`:
```bash
pip install python-multipart
```

### Wrong Python environment
If you see imports from `/opt/anaconda3/` instead of `.venv/`, your conda base is overriding the venv:
```bash
conda deactivate
source .venv/bin/activate
which python  # Should show .venv path
```

### Render free tier sleeping
Free tier sleeps after 15 minutes of inactivity. First request takes ~30 seconds. Visit the `/docs` URL to wake it up before using the Streamlit dashboard.
