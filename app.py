# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from google import genai
from PIL import Image

# Set up clean layout with a wide canvas structure
st.set_page_config(page_title="Multi-Modal AI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# =========================================================
# CUSTOM PREMIUM UI STYLING (CSS INJECTION)
# =========================================================
st.markdown("""
    <style>
        /* Main App Background and Typography */
        .main {
            background-color: #0d1117;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Modern Card Containers */
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Beautiful Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #2f7fff 0%, #1756ff 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(23, 86, 255, 0.4);
        }
        
        /* Metrics Redesign */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #58a6ff !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: #8b949e !important;
        }
        
        /* Markdown Blockquote / Copywriter Output Styling */
        blockquote {
            background-color: #21262d !important;
            border-left: 4px solid #58a6ff !important;
            color: #c9d1d9 !important;
            padding: 1rem !important;
            border-radius: 0 8px 8px 0;
            font-size: 1.05rem;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LAYER 1: Tabular Machine Learning Pipeline Loading
# =========================================================
@st.cache_resource
def load_pricing_pipeline():
    with open("models/pricing_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    pricing_model, scaler = load_pricing_pipeline()
except FileNotFoundError:
    st.error("⚠️ Scikit-Learn binaries not found! Make sure to run 'python train_model.py' first.")

# Header Layout Elements
st.title("🛍️ Multi-Modal AI E-Commerce Product Auditor")
st.caption("Native Mac Pipeline: Scaled Linear Regression Pricing + Gemini Multi-Modal Vision Classifier & AI Price Auditor")
st.markdown("<hr style='border-color: #30363d;' />", unsafe_allow_html=True)

# Welcome Guide
st.markdown("""
### Welcome to the Smart Reseller Auditor! 🚀
This advanced dashboard helps e-commerce sellers evaluate their inventory, verify market value pricing accuracy with dual-stage AI validation, and generate instant marketing copy in seconds. 
""")

st.markdown("---")

# Split screen into 2 core columns (Left Side: Inputs, Right Side: AI Analytics Engine)
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown("### 📸 Workspace Asset Ingestion")
    
    with st.container():
        uploaded_files = st.file_uploader(
            "Upload product photos (Select multiple angles)...", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown("<br>", unsafe_allow_html=True)
            t_cols = st.columns(min(len(uploaded_files), 4))
            for idx, file in enumerate(uploaded_files):
                with t_cols[idx % 4]:
                    img_preview = Image.open(file)
                    st.image(img_preview, caption=f"Angle {idx+1}", use_container_width=True)

    st.markdown("### 📊 Metadata Configuration")
    with st.container():
        user_price = st.number_input("Your Proposed Listing Price ($)", min_value=1.0, value=15.0, step=5.0)
        st.markdown("<br>", unsafe_allow_html=True)
        condition = st.slider("Product Structural Rating / Condition", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

with col2:
    st.markdown("### 🤖 Intelligence Evaluation Hub")
    
    run_audit = st.button("🚀 EXECUTE MULTI-STAGE ANALYSIS")
    
    if run_audit:
        if not uploaded_files:
            st.warning("Please upload at least one image asset to trigger the computer vision network.")
        else:
            with st.spinner("Processing multi-image pipeline via Google GenAI Engine..."):
                
                classification_prompt = (
                    "Look at all these uploaded angles of the same single product. "
                    "Categorize it into exactly one of these seven labels: "
                    "Electronics, Clothing, Footwear, Books, Video Game Discs, Gaming Consoles, or Health Care. "
                    "Return ONLY the category name as a single word or space-separated phrase exactly as listed."
                )
                
                try:
                    client = genai.Client()
                    vision_payload = [classification_prompt]
                    temp_paths = []
                    
                    for idx, file in enumerate(uploaded_files):
                        temp_path = f"temp_prod_image_{idx}.jpg"
                        img = Image.open(file)
                        img.save(temp_path)
                        temp_paths.append(temp_path)
                        
                        uploaded_vision_file = client.files.upload(file=temp_path)
                        vision_payload.append(uploaded_vision_file)
                    
                    # 1. Ask Gemini to classify the item based on ALL images combined
                    vision_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=vision_payload
                    )
                    
                    detected_category = vision_response.text.strip().replace(".", "")
                    
                    categories_map = {
                        'Electronics': 1.0, 
                        'Clothing': 2.0, 
                        'Footwear': 3.0,
                        'Books': 4.0,
                        'Health Care': 4.0,       
                        'Video Game Discs': 5.0,
                        'Gaming Consoles': 5.0  
                    }
                    detected_category_id = categories_map.get(detected_category, 1.0)

                    # --- STAGE 1: REGRESSION PREDICTION ---
                    raw_features = np.array([[condition, detected_category_id]])
                    scaled_features = scaler.transform(raw_features)
                    raw_model_price = float(pricing_model.predict(scaled_features)[0])
                    
                    # --- STAGE 2: GENERATIVE AI VALUE SANITY AUDIT ---
                    # Instead of manual code multipliers, we ask Gemini to look at the photo and correct the model
                    audit_prompt = (
                        f"You are an expert e-commerce price auditor. A linear regression model looked at this product "
                        f"and estimated its resale value to be ${raw_model_price:.2f}. "
                        f"Look closely at the image assets provided. If the model's estimate is wildly unrealistic for this "
                        f"specific item (e.g., a simple hand sanitizer or paperback book showing $100), adjust the price "
                        f"downward or upward to a realistic e-commerce market value. "
                        f"Return ONLY a valid decimal number representing the corrected fair price. Do not include a dollar sign or any text."
                    )
                    
                    audit_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[audit_prompt] + vision_payload[1:]
                    )
                    
                    # Parse Gemini's clean numeric correction safely
                    try:
                        predicted_fair_price = float(audit_response.text.strip())
                    except ValueError:
                        # Fallback to model price if text parsing fails
                        predicted_fair_price = raw_model_price

                    st.success("✅ Multi-Stage Analysis Complete!")

                    # Profile Summary Card
                    st.markdown("#### 📦 Product Profile Summary")
                    with st.container():
                        st.markdown(f"""
                        * **Identified Asset Type:** `{detected_category}`
                        * **Inspected Physical Quality:** `{condition} / 5.0` 
                        * **Seller Target Valuation:** `${user_price:.2f}`
                        
                        This product profile has been extracted from your visual assets and successfully verified by our dual-engine network.
                        """)

                    # Render Metrics Dashboard Inside Clean Cards
                    st.markdown("#### 📈 Model Metrics Reconciliation")
                    with st.container():
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric(label="AI Vision Categorization", value=detected_category)
                        with m_col2:
                            st.metric(label="AI-Audited Value Estimation", value=f"${predicted_fair_price:.2f}")
                    
                    # Auditing Variance Check System Block
                    price_gap = user_price - predicted_fair_price
                    percentage_gap = (price_gap / predicted_fair_price) * 100
                    
                    st.markdown("#### ⚖️ Risk Assessment Verdict")
                    if percentage_gap > 10.0:
                        st.warning(f"📈 Overpriced variant (Listed {percentage_gap:.1f}% higher than standard market target expectations).")
                    elif percentage_gap < -10.0:
                        st.error(f"📉 Underpriced variant (Listed {abs(percentage_gap):.1f}% lower than standard market target expectations).")
                    else:
                        st.info("⚖️ Fair market baseline distribution matched (Within ±10% acceptable tolerance threshold).")

                    # --- GENERATIVE AI MARKETING COPYWRITER ---
                    st.markdown("<br>✍️ Engine Optimized Ad Copy", unsafe_allow_html=True)
                    
                    marketing_prompt = (
                        f"Write a short, engaging e-commerce platform product listing description for this item. "
                        f"It is confirmed to be an item of '{detected_category}' with a condition score of {condition}/5.0 "
                        f"and an attractive price tag of ${user_price:.2f}. Detail its characteristics, value proposition "
                        f"based on the provided visual angles, and provide trendy hashtags."
                    )
                    
                    marketing_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[marketing_prompt] + vision_payload[1:]
                    )
                    
                    with st.container():
                        st.markdown(f"> {marketing_response.text}")
                    
                    # Clean up local image asset caches
                    for path in temp_paths:
                        if os.path.exists(path):
                            os.remove(path)
                            
                except Exception as e:
                    st.error("Generative layer encountered an error. Verify your API key variable configuration.")
                    st.caption(f"Traceback tracking block: {e}")
