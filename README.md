# 🛍️ Multi-Modal AI E-Commerce Product Auditor

An advanced, production-grade e-commerce inventory auditing dashboard designed to help online resellers optimize their listings. This system runs a hybrid, dual-stage artificial intelligence pipeline that cross-examines traditional tabular machine learning logic against generative computer vision networks to accurately evaluate and price marketplace inventory.

🌐 **Live App Link:** (https://ibrahimibnu-ai-product-auditor-app-ft7cmh.streamlit.app/)

---

## 🚀 Key Features

* **Dual-Engine Valuation Pipeline:** Combines traditional predictive algorithms with a real-time Generative AI sanity audit to eliminate category pricing skews.
* **Multi-Angle Visual Ingestion:** Allows users to drop multiple image files simultaneously for comprehensive computer vision analysis.
* **Automated Product Profiling:** Instantly extracts the product's domain category and quality metrics using zero-shot classification.
* **E-Commerce Copywriter:** Synthesizes pricing parameters, item condition, and visual characteristics to generate highly engaging, platform-ready marketing descriptions with trending hashtags.
* **Premium Dark UI Layout:** Features a fully customized Streamlit frontend with smooth transitions, responsive card containers, and dynamic color-coded visual metrics.

---

## 🧠 Architectural Overview

The application processes data through a modern, multi-stage reconciliation pipeline:

1. **Tabular Predictive Layer:** Uses a Scikit-Learn `LinearRegression` model. Product structural condition scores and categorical data strings are passed through a `StandardScaler` Z-score matrix to predict a foundational baseline value.
2. **Generative Sanity Check:** The raw mathematical prediction is passed alongside the visual image binaries directly to the `gemini-2.5-flash` model. The AI functions as an online appraiser, dynamically correcting linear skew vectors (e.g., separating mass-market commodities like hand sanitizers or paperbacks from high-value collectibles or textbooks).
3. **Risk Assessment Output:** Calculates the variance percentage between the seller's target price and the final AI-audited valuation, delivering real-time flags for overpriced or underpriced inventory variance thresholds.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Dashboard:** Streamlit (Custom CSS-Injected Layout)
* **Core ML Layer:** Python 3.9+, Scikit-Learn, NumPy, Pandas
* **Generative Engine:** Google GenAI SDK (`gemini-2.5-flash`)
* **Image Processing:** Pillow (PIL)

---

