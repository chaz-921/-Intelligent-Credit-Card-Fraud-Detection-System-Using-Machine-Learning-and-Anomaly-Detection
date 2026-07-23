import sys
import json
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv  


load_dotenv()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt

import google.generativeai as genai

class FraudDetectionWorker(QThread):
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)

    def __init__(self, image_path, api_key):
        super().__init__()
        self.image_path = image_path
        self.api_key = api_key

    def run(self):
        try:
            
            genai.configure(api_key=self.api_key)
            vlm_model = genai.GenerativeModel('gemini-2.5-flash')

            rf_model = joblib.load('rf_model.pkl')
            iso_forest = joblib.load('iso_forest.pkl')
            scaler = joblib.load('scaler.pkl')

            img = Image.open(self.image_path)
            prompt = """
            Analyze this receipt/transaction image and extract details into JSON format:
            {"Amount": float, "Merchant": "string"}
            Return strictly valid JSON without markdown wrapping or extra text.
            """
            response = vlm_model.generate_content([prompt, img])
            
            raw_json = response.text.strip().replace('```json', '').replace('```', '')
            extracted_data = json.loads(raw_json)
            amount = float(extracted_data.get("Amount", 0.0))

            live_features = pd.DataFrame({
                'Amount': [amount],
                'Location_Score': [0.5],
                'PCA1': [np.random.normal(0, 1)],
                'PCA2': [np.random.normal(0, 1)],
                'PCA3': [np.random.normal(0, 1)]
            })
            live_scaled = scaler.transform(live_features)

            rf_pred = rf_model.predict(live_scaled)[0]
            iso_pred = iso_forest.predict(live_scaled)[0]

            if rf_pred == 1 or iso_pred == -1:
                status = "🚨 FRAUD ALERT: Anomaly or High Risk detected!"
            else:
                status = "✅ APPROVED: Legitimate transaction."

            self.finished.emit(extracted_data, status)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("credit card Fraud Detection System")
        self.resize(700, 600)
        
        self.selected_image_path = None
        self.api_key = os.getenv("GEMINI_API_KEY") 

        self.init_ui()