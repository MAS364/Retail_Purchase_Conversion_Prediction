"""
streamlit_app.py — Purchase Predictor Dashboard
─────────────────────────────────────────────────
CSV upload is the primary interface.
All preprocessing happens in the FastAPI backend.
"""

import os
import json
import time
from typing import Optional

import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")

st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 12px; padding: 20px 24px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] span { color: #e2e8f0 !important; }
.result-purchase {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius: 14px;
    padding: 28px 32px; text-align: center; margin: 16px 0;
}
.result-no-purchase {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-radius: 14px;
    padding: 28px 32px; text-align: center; margin: 16px 0;
}
.result-label { font-size: 32px; font-weight: 700; color: white; margin: 0; }
.result-prob { font-size: 16px; color: #d1d5db; margin-top: 4px; }
.status-ok { color: #10b981; font-weight: 600; }
.status-err { color: #ef4444; font-weight: 600; }
#MainMenu, footer, header { visibility: hidden; }

/* Compact "model badge" shown next to the main-page model selector */
.model-active-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px 14px;
    color: #e2e8f0;
    font-size: 14px;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

SAMPLE_SESSION = {
    "price": 150.0, "total_time_spent": 120.0, "session_length": 6,
    "interaction_count": 6, "view_count": 3, "click_count": 2,
    "wishlist_count": 0, "add_to_cart_count": 1,
    "avg_time_per_interaction": 20.0, "cart_to_view_ratio": 0.33,
    "click_to_view_ratio": 0.67, "has_cart_action": 1, "has_wishlist_action": 0,
    "hour": 14, "day_of_week": 2, "month": 3, "is_weekend": 0,
    "category": "electronics", "brand": "apple", "channel": "web",
    "device_type": "desktop", "region": "uk", "traffic_source": "organic",
}
ALL_MODELS_FALLBACK = ["mlp", "decision_tree", "naive_bayes", "svm"]


# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(path, timeout=5):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path, payload, timeout=30):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    if r.status_code == 422:
        raise ValueError(f"Validation error: {r.json().get('detail', r.text)}")
    r.raise_for_status()
    return r.json()


def api_upload_csv(file_bytes, filename, model, timeout=120):
    r = requests.post(
        f"{API_URL}/predict/csv?model={model}",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=timeout,
    )
    if r.status_code == 422:
        raise ValueError(f"Validation error: {r.json().get('detail', r.text)}")
    r.raise_for_status()
    return r.json()


def render_result(result):
    is_purchase = result.get("prediction") == 1
    css = "result-purchase" if is_purchase else "result-no-purchase"
    icon = "✅" if is_purchase else "❌"
    label = result.get("label", "unknown").replace("_", " ").title()
    prob = result.get("probability")
    prob_str = f"Confidence: {prob:.1%}" if prob is not None else ""
    model_str = result.get("model", "")
    st.markdown(
        f'<div class="{css}">'
        f'<p class="result-label">{icon} {label}</p>'
        f'<p class="result-prob">{prob_str} · Model: {model_str}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Health check + model list (computed once, before any selector renders) ─────

health = api_get("/health")
if health:
    available_models = health["models_loaded"]
    default_model = health["default_model"]
else:
    available_models = ALL_MODELS_FALLBACK
    default_model = "mlp"

# Keep a single source of truth for the selected model across reruns,
# so the sidebar selector and the main-page selector always agree —
# this matters most on mobile, where the sidebar may be collapsed
# and the user only ever sees the main-page control.
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = (
        default_model if default_model in available_models else available_models[0]
    )
elif st.session_state["selected_model"] not in available_models:
    st.session_state["selected_model"] = (
        default_model if default_model in available_models else available_models[0]
    )


def _sync_from_sidebar():
    st.session_state["selected_model"] = st.session_state["sidebar_model_select"]


def _sync_from_main():
    st.session_state["selected_model"] = st.session_state["main_model_select"]


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 Purchase Predictor")
    st.caption("ML-powered conversion prediction")
    st.divider()

    if health:
        st.markdown(
            f'<span class="status-ok">● API Online</span> — {len(health["models_loaded"])} models loaded',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-err">● API Offline</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(st.session_state["selected_model"]),
        key="sidebar_model_select",
        on_change=_sync_from_sidebar,
    )
    st.divider()
    st.caption(f"Endpoint: `{API_URL}`")
    st.markdown(f"[📄 Swagger Docs]({API_URL}/docs) · [📘 ReDoc]({API_URL}/redoc)")


# ── Main ───────────────────────────────────────────────────────────────────────

st.markdown("# Purchase Prediction Dashboard")
st.caption("Predict whether an e-commerce session will convert to a purchase")

# Main-page model selector + status — always visible regardless of sidebar
# state, so it works on mobile where the sidebar starts collapsed.
status_col, model_col = st.columns([3, 2])
with status_col:
    if health:
        st.markdown(
            f'<div class="model-active-badge">'
            f'<span class="status-ok">● API Online</span> — {len(health["models_loaded"])} models loaded'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="model-active-badge"><span class="status-err">● API Offline</span></div>',
            unsafe_allow_html=True,
        )
with model_col:
    st.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(st.session_state["selected_model"]),
        key="main_model_select",
        on_change=_sync_from_main,
        label_visibility="collapsed",
    )

selected_model = st.session_state["selected_model"]

tab_csv, tab_single, tab_compare, tab_api = st.tabs(
    ["📊 CSV Upload", "🎯 Single Prediction", "⚖️ Model Comparison", "🔌 API Reference"]
)


# ━━ Tab 1: CSV Upload (PRIMARY) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_csv:
    st.subheader("Upload CSV — Batch Prediction")
    st.caption(
        "Upload a CSV that matches the provided sample schema. "
        "Download the sample CSV below or use your own dataset with the same columns."
    )

    st.info(
        "📄 Tip: Use the sample CSV or upload your own dataset with the same schema."
    )

    with open("sample_data.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample CSV",
            data=f,
            file_name="sample_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_up",
    )

    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            st.info(f"**{len(preview_df):,} rows** · {len(preview_df.columns)} columns · `{uploaded.name}`")
            st.dataframe(preview_df.head(5), use_container_width=True)
            uploaded.seek(0)
            file_bytes = uploaded.read()
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            file_bytes = None

        if file_bytes and st.button("🚀 Run Prediction", type="primary", use_container_width=True, key="btn_csv"):
            with st.spinner("Uploading and processing (all in backend)…"):
                try:
                    result = api_upload_csv(file_bytes, uploaded.name, selected_model)

                    st.markdown("### Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sessions", f"{result['session_count']:,}")
                    m2.metric("Purchases", f"{result['purchases']:,}")
                    m3.metric("Conversion Rate", f"{result['conversion_rate']:.1%}")
                    m4.metric("Latency", f"{result['latency_ms']:,} ms")

                    fmt = result.get("data_format", "unknown")
                    if fmt == "raw_events":
                        st.success(
                            f"✅ Auto-detected **raw event data** → "
                            f"engineered **{result['session_count']:,} sessions** "
                            f"from **{result['raw_event_count']:,} events** (server-side)"
                        )
                    else:
                        st.success(
                            f"✅ Session-level data — **{result['session_count']:,} sessions** predicted directly"
                        )

                    pred_df = pd.DataFrame(result["predictions"])
                    st.dataframe(pred_df, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Results CSV",
                        pred_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except ValueError as exc:
                    st.error(f"Validation error: {exc}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API — is uvicorn running?")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ━━ Tab 2: Single Prediction (secondary — calls backend) ━━━━━━━━━━━━━━━━━━━━

with tab_single:
    st.subheader("Single Session Prediction")
    st.caption("Enter 23 session features manually. Prediction runs via the API backend.")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown("##### Numerical Features")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            price = st.number_input("Price (£)", min_value=0.0, value=150.0, step=10.0, key="s_price")
            total_time = st.number_input("Total Time (sec)", min_value=0.0, value=120.0, step=10.0, key="s_time")
            session_len = st.number_input("Session Length", min_value=1, value=6, key="s_len")
            interactions = st.number_input("Interactions", min_value=0, value=6, key="s_inter")
            views = st.number_input("Views", min_value=0, value=3, key="s_views")
            clicks = st.number_input("Clicks", min_value=0, value=2, key="s_clicks")
        with r1c2:
            wishlists = st.number_input("Wishlists", min_value=0, value=0, key="s_wish")
            carts = st.number_input("Add to Carts", min_value=0, value=1, key="s_carts")
            avg_time_int = st.number_input("Avg Time/Interaction", min_value=0.0, value=20.0, key="s_avgti")
            cart_view = st.slider("Cart-to-View Ratio", 0.0, 5.0, 0.33, key="s_cvr")
            click_view = st.slider("Click-to-View Ratio", 0.0, 5.0, 0.67, key="s_clvr")
        with r1c3:
            has_cart = st.selectbox("Has Cart Action", [0, 1], index=1, key="s_hcart")
            has_wish = st.selectbox("Has Wishlist Action", [0, 1], index=0, key="s_hwish")
            hour = st.slider("Hour", 0, 23, 14, key="s_hour")
            dow = st.slider("Day of Week (0=Mon)", 0, 6, 2, key="s_dow")
            month = st.slider("Month", 1, 12, 3, key="s_month")
            is_wknd = st.selectbox("Is Weekend", [0, 1], index=0, key="s_wknd")

    with col_right:
        st.markdown("##### Categorical Features")
        category = st.selectbox(
            "Category",
            ["electronics", "clothing", "groceries", "home", "beauty", "sports", "toys", "books"],
            key="s_cat",
        )
        brand = st.selectbox(
            "Brand",
            ["apple", "samsung", "nike", "adidas", "sony", "lg", "organicco", "other"],
            key="s_brand",
        )
        channel = st.selectbox("Channel", ["web", "mobile_app", "social", "email"], key="s_chan")
        device = st.selectbox("Device", ["desktop", "mobile", "tablet"], key="s_dev")
        region = st.selectbox("Region", ["uk", "us", "eu", "jp", "in", "au", "other"], key="s_reg")
        traffic = st.selectbox(
            "Traffic Source",
            ["organic", "paid", "direct", "referral", "social"],
            key="s_traf",
        )

    session_data = {
        "price": price, "total_time_spent": total_time,
        "session_length": session_len, "interaction_count": interactions,
        "view_count": views, "click_count": clicks,
        "wishlist_count": wishlists, "add_to_cart_count": carts,
        "avg_time_per_interaction": avg_time_int,
        "cart_to_view_ratio": cart_view, "click_to_view_ratio": click_view,
        "has_cart_action": has_cart, "has_wishlist_action": has_wish,
        "hour": hour, "day_of_week": dow, "month": month, "is_weekend": is_wknd,
        "category": category, "brand": brand, "channel": channel,
        "device_type": device, "region": region, "traffic_source": traffic,
    }

    st.divider()
    if st.button("🔮 Predict", type="primary", use_container_width=True, key="btn_predict"):
        if not health:
            st.error("API is offline.")
        else:
            with st.spinner("Running prediction…"):
                try:
                    t0 = time.time()
                    result = api_post(f"/predict?model={selected_model}", session_data)
                    ms = round((time.time() - t0) * 1000)
                    render_result(result)
                    st.caption(f"⏱ {ms} ms")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ━━ Tab 3: Model Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_compare:
    st.subheader("Compare All Models")
    st.caption("Sends the same session (from Single Prediction tab) to every loaded model.")

    if st.button("⚖️ Compare All Models", type="primary", key="btn_compare"):
        if not health:
            st.error("API is offline.")
        else:
            rows = []
            for m in available_models:
                try:
                    t0 = time.time()
                    r = api_post(f"/predict?model={m}", session_data)
                    ms = round((time.time() - t0) * 1000)
                    rows.append({
                        "Model": m,
                        "Prediction": r.get("label", "?").replace("_", " ").title(),
                        "Probability": r.get("probability"),
                        "Latency (ms)": ms,
                    })
                except Exception as exc:
                    rows.append({
                        "Model": m,
                        "Prediction": "Error",
                        "Probability": None,
                        "Latency (ms)": None,
                    })
            if rows:
                cdf = pd.DataFrame(rows)
                st.dataframe(cdf, use_container_width=True, hide_index=True)
                chart_rows = [r for r in rows if r["Probability"] is not None]
                if chart_rows:
                    chart_df = pd.DataFrame(chart_rows)[["Model", "Probability"]].set_index("Model")
                    st.bar_chart(chart_df, color="#10b981")


# ━━ Tab 4: API Reference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_api:
    st.subheader("API Reference")

    st.markdown("**Health check**")
    st.code(f"curl {API_URL}/health", language="bash")

    st.markdown("**CSV upload (raw or session-level — auto-detected)**")
    st.code(
        f'curl -X POST "{API_URL}/predict/csv?model=mlp" \\\n'
        f'  -F "file=@your_data.csv"',
        language="bash",
    )

    st.markdown("**Single prediction**")
    st.code(
        f'curl -X POST "{API_URL}/predict?model=mlp" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{json.dumps(SAMPLE_SESSION)}'",
        language="bash",
    )

    st.markdown("**Interactive docs**")
    st.markdown(f"[Swagger UI]({API_URL}/docs) · [ReDoc]({API_URL}/redoc)")