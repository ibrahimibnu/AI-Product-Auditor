# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from google import genai
from PIL import Image

# Set up clean layout on your Mac browser
st.set_page_config(page_title="Multi-Modal AI Dashboard", layout="wide")

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
st.caption("Native Mac Pipeline: Scaled Linear Regression Pricing + Gemini Multi-Modal Vision Classifier & Copywriter")
st.markdown("---")

# =========================================================
# 📖 NEW: USER DESCRIPTION & PLATFORM GUIDE
# =========================================================
st.markdown("""
### Welcome to the Smart Reseller Auditor! 🚀
This advanced dashboard helps e-commerce sellers evaluate their inventory, verify market value pricing accuracy, and generate instant marketing copy in seconds. 

#### 📈 How to use the tool:
1. **Upload an image** of your item in the left panel.
2. **Set your expected price** and use the slider to rate the item's **structural condition** (1.0 = heavily worn, 5.0 = pristine).
3. Click the **Execute Multi-Stage Analysis** button.
4. Watch our hybrid AI engine categorize your item, calculate a fair-market price based on historical data, and draft your ad copy!
""")

# Expandable Technical Background for users who want to know more
with st.expander("🔍 See how the AI models process your data behind the scenes"):
    st.markdown("""
    This website runs an advanced, multi-modal pipeline to protect and optimize your listings:
    * **Cloud Vision (Gemini 2.5 Flash):** Evaluates your raw image and instantly extracts the domain classification (`Electronics`, `Clothing`, or `Footwear`).
    * **Predictive Pricing (Scikit-Learn Regression):** Your item's condition rating and the AI-detected category are normalized using a standard Z-score scaler and run through a trained Linear Regression model to find its true financial baseline.
    * **Automated Copywriter (Generative AI):** Synthesizes all parameters to output a platform-ready, hashtag-optimized ad caption.
    """)

st.markdown("---")

# Split screen into 2 columns (Left Side: Inputs, Right Side: AI Analytics Engine)
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Product Upload & Parameters")
    uploaded_file = st.file_uploader("Upload product photo...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img_preview = Image.open(uploaded_file)
        st.image(img_preview, caption="Uploaded Product Preview", use_container_width=True)
    
    user_price = st.number_input("Your Proposed Listing Price ($)", min_value=1.0, value=150.0, step=5.0)
    condition = st.slider("Product Structural Rating / Condition", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

with col2:
    st.header("🤖 Multi-Modal Evaluation Engine")
    run_audit = st.button("🚀 EXECUTE MULTI-STAGE ANALYSIS")
    
    if run_audit:
        if uploaded_file is None:
            st.warning("Please upload an image asset first to trigger the computer vision network.")
        else:
            with st.spinner("Processing pipeline via Google GenAI Engine..."):
                
                # Save uploaded Streamlit file to a temporary file path for cloud processing
                temp_path = "temp_prod_image.jpg"
                img_preview.save(temp_path)
                
                # Instruction setup for the zero-shot cloud vision task
                classification_prompt = (
                    "Look at this product photo. Categorize it into exactly one of these three labels: "
                    "Electronics, Clothing, or Footwear. Return ONLY the category name as a single word."
                )
                
                try:
                    # Initializes the client via system environment token safely
                    client = genai.Client()
                    
                    # Upload the binary image asset using SDK cloud utilities
                    uploaded_vision_file = client.files.upload(file=temp_path)
                    
                    # 1. Ask Gemini to classify the image text label
                    vision_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[classification_prompt, uploaded_vision_file]
                    )
                    
                    detected_category = vision_response.text.strip().replace(".", "")
                    
                    # Map the categorical label back to your Scikit-Learn tabular numerical model indexes
                    categories_map = {'Electronics': 1.0, 'Clothing': 2.0, 'Footwear': 3.0}
                    detected_category_id = categories_map.get(detected_category, 1.0)

                    # --- REGRESSION PREDICTION ENGINE LOOP ---
                    raw_features = np.array([[condition, detected_category_id]])
                    scaled_features = scaler.transform(raw_features)
                    predicted_fair_price = float(pricing_model.predict(scaled_features)[0])
                    
                    # Calculation metrics display
                    st.success("✅ Multi-Stage Analysis Complete!")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="AI Vision Categorization", value=detected_category)
                    with m_col2:
                        st.metric(label="Regression Value Estimation", value=f"${predicted_fair_price:.2f}")
                    
                    # Alert calculation variance check
                    price_gap = user_price - predicted_fair_price
                    if price_gap > 15:
                        st.warning(f"📈 Overpriced variant (Listed ${price_gap:.2f} higher than standard target expectations).")
                    elif price_gap < -15:
                        st.error(f"📉 Underpriced variant (Listed ${abs(price_gap):.2f} lower than standard target expectations).")
                    else:
                        st.info("⚖️ Fair market baseline distribution matched.")

                    st.markdown("---")

                    # --- GENERATIVE AI MARKETING COPYWRITER ---
                    st.subheader("✍️ Automated Copywriter Output")
                    
                    marketing_prompt = (
                        f"Write a short, engaging e-commerce platform product listing description for this item. "
                        f"It is confirmed to be an item of '{detected_category}' with a condition score of {condition}/5.0 "
                        f"and an attractive price tag of ${user_price:.2f}. Detail its characteristics, value proposition, "
                        f"and provide trendy hashtags."
                    )
                    
                    # 2. Ask Gemini to output the marketing ad layout
                    marketing_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[marketing_prompt, uploaded_vision_file]
                    )
                    
                    # Render response matching markdown blockquote layout structure
                    st.markdown(f"> {marketing_response.text}")
                    
                    # Clean up local asset cache from workspace disk space
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error("Generative layer encountered an error. Verify your API key variable configuration.")
                    st.caption(f"Traceback tracking block: {e}")

#.
