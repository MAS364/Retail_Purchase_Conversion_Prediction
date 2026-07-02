# Retail Purchase Conversion Prediction API

Production-ready machine learning API for predicting whether an e-commerce customer session will result in a purchase using behavioural and contextual features.

**🌐 Live Demo:** https://retail-purchase-conversion-prediction.streamlit.app/

---

# Problem

Online retailers receive thousands of customer sessions every day, but only a small proportion lead to a purchase. Identifying high-intent customers in real time enables businesses to personalise offers, optimise marketing spend, reduce cart abandonment, and improve conversion rates.

This project builds an end-to-end machine learning pipeline and production-ready REST API that predicts purchase likelihood from customer session behaviour, making the model suitable for integration into modern retail applications. :contentReference[oaicite:0]{index=0}

---

# Business Impact

- 📈 Increase conversion rates by identifying customers most likely to purchase.
- 🎯 Improve marketing ROI through targeted campaigns instead of blanket promotions.
- 🛒 Support cart abandonment strategies with real-time purchase scoring.
- 💰 Reduce customer acquisition costs by prioritising high-value sessions.
- ⚡ Deploy scalable real-time predictions through a FastAPI REST API on Google Cloud Run.

---

# Result

**MLP achieved 0.8936 ROC-AUC and 82.25% accuracy, while the Decision Tree achieved 97.27% recall for identifying purchasing customers.** :contentReference[oaicite:1]{index=1}

---

# Approach

## Data

- Retail e-commerce user behaviour dataset
- 108,584 customer events
- Aggregated into 18,000 customer sessions
- Binary classification problem
- 23.35% purchase conversion rate :contentReference[oaicite:2]{index=2}

## Feature Engineering

Behavioural and contextual features including:

- View count
- Click count
- Add-to-cart count
- Wishlist count
- Session duration
- Interaction count
- Cart-to-view ratio
- Click-to-view ratio
- Time-based features
- Product category
- Device type
- Traffic source
- Region

Target leakage features were removed before model training to ensure realistic performance. :contentReference[oaicite:3]{index=3}

## Models Evaluated

- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

The API supports all four trained models, with the MLP selected as the strongest overall model based on ROC-AUC.

## Evaluation

Models were evaluated using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score
- Stratified 5-Fold Cross Validation

---

# Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `FastAPI` `Pydantic` `Pytest` `Docker` `GitHub Actions` `Google Cloud Run`

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Redirect to Swagger documentation |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc documentation |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000 records) |

---

# Project Structure

```text
retail_conversion_api/
├── app.py
├── src/
│   ├── schemas.py
│   ├── utils.py
│   ├── predict.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   └── data_preprocessing.py
├── models/
│   ├── decision_tree_pipeline.joblib
│   ├── naive_bayes_pipeline.joblib
│   ├── svm_pipeline.joblib
│   └── mlp_pipeline.joblib
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── deploy.yml
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

# Why FastAPI?

| Flask | FastAPI |
|--------|----------|
| Manual request validation | Automatic validation using Pydantic |
| No built-in API documentation | Swagger UI & ReDoc generated automatically |
| Limited async support | Native asynchronous support |
| Minimal type checking | Full Python type hints |
| WSGI | High-performance ASGI |

---

# Running Locally

## Clone the repository

```bash
git clone https://github.com/your-username/retail-purchase-conversion-api.git

cd retail-purchase-conversion-api
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload --port 8080
```

Open:

```
http://localhost:8080/docs
```

## Run Tests

```bash
pip install pytest httpx

pytest tests/ -v
```

---

# Run with Docker

```bash
docker build -t purchase-predictor-api .

docker run -p 8080:8080 purchase-predictor-api
```

Verify deployment:

```bash
curl http://localhost:8080/health
```

---

# Deploy to Google Cloud Run

```bash
gcloud run deploy purchase-predictor-api \
  --image gcr.io/YOUR_PROJECT_ID/purchase-predictor-api \
  --region europe-west2 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080
```

---

# CI/CD

GitHub Actions automatically:

- Runs unit tests
- Builds Docker image
- Pushes the container image
- Deploys to Google Cloud Run

---

# Example API Request

```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{
  "price":150,
  "total_time_spent":120,
  "session_length":6,
  "interaction_count":6,
  "view_count":3,
  "click_count":2,
  "wishlist_count":0,
  "add_to_cart_count":1,
  "avg_time_per_interaction":20.0,
  "cart_to_view_ratio":0.33,
  "click_to_view_ratio":0.67,
  "has_cart_action":1,
  "has_wishlist_action":0,
  "hour":14,
  "day_of_week":2,
  "month":3,
  "is_weekend":0,
  "category":"electronics",
  "brand":"apple",
  "channel":"web",
  "device_type":"desktop",
  "region":"uk",
  "traffic_source":"organic"
}'
```

---

# Future Improvements

- Model monitoring and drift detection
- Explainable AI using SHAP
- Model versioning with MLflow
- Authentication using API keys or OAuth
- Continuous retraining pipeline
- Kubernetes deployment
- Feature store integration

---

# Author

**Mohammad Arshad Siddique**

Machine Learning • Data Science • MLOps • Cloud Deployment
