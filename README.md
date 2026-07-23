# -Intelligent-Credit-Card-Fraud-Detection-System-Using-Machine-Learning-and-Anomaly-Detection

An end-to-end cybersecurity solution combining Vision-Language Models (VLM OCR) and hybrid Machine Learning algorithms (Random Forest + Isolation Forest) to detect fraud in real-time transactions.

## 🚀 Key Features
* **VLM Data Extraction:** Uses Gemini VLM API to perform OCR on transaction images/receipts and extract structured data.
* **Hybrid Model Architecture:** 
  * **Supervised Learning (Random Forest):** Classifies known fraud patterns.
  * **Anomaly Detection (Isolation Forest):** Identifies novel deviations and outliers.
* **Interactive UI:** Built with PyQt6 using asynchronous multi-threading (`QThread`) to prevent UI freeze during API and inference calls.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn, Pillow, Joblib
* **API:** Google Generative AI (Gemini Flash)
* **GUI Framework:** PyQt6
