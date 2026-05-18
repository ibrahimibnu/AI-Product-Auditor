# 🛍️ Multi-Modal AI E-Commerce Product Auditor

A live, cloud-deployed web application that automates and optimizes product listings for online resale marketplaces (like eBay or Poshmark). It combines cloud-native AI vision, a machine learning pricing model, and a generative copywriting engine into a seamless, one-click experience.

🌐 **Live App Link:** (https://ibrahimibnu-ai-product-auditor-app-ft7cmh.streamlit.app/)

---

## 🚀 Key Features & Workflow
When a user uploads a product photo, sets the item's condition rating, and enters a target price, the app executes a multi-stage pipeline:

1. **AI Vision Categorization:** Automatically identifies the product category (`Electronics`, `Clothing`, or `Footwear`) directly from the uploaded image.
2. **Predictive Machine Learning Pricing:** Feeds the item's condition score and the mapped category data into a custom-trained **Scikit-Learn Regression Model** to calculate an objective fair market value.
3. **Market Delta Analysis:** Compares the user's proposed price to the calculated market baseline and instantly flags if the item is overpriced or underpriced.
4. **Automated Copywriter Output:** Uses **Gemini 2.5 Flash** to evaluate the product's visual attributes and parameters, generating a platform-ready marketing description complete with trending hashtags.

---

## 📦 Tech Stack
* **UI & Hosting:** Streamlit (Streamlit Community Cloud)
* **Generative Core & Vision:** Google GenAI SDK (`gemini-2.5-flash`)
* **Predictive Analytical Engine:** Scikit-Learn (Linear Regression Core & StandardScaler)
* **Data Pipelines:** NumPy, Pandas, Pillow (PIL)

---
