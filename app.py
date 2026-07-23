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


from google import genai


class FraudDetectionWorker(QThread):
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)

    def __init__(self, image_path, api_key):
        super().__init__()
        self.image_path = image_path
        self.api_key = api_key

    def run(self):
        try:
        
            client = genai.Client(api_key=self.api_key)

            rf_model = joblib.load('rf_model.pkl')
            iso_forest = joblib.load('iso_forest.pkl')
            scaler = joblib.load('scaler.pkl')

            img = Image.open(self.image_path)
            prompt = """
            Analyze this receipt/transaction image and extract details into JSON format:
            {"Amount": float, "Merchant": "string"}
            Return strictly valid JSON without markdown wrapping or extra text.
            """
        
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[img, prompt]
            )
            
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
        self.setWindowTitle("Cybersecurity Fraud Detection System")
        self.resize(700, 600)
        
        self.selected_image_path = None
        self.api_key = os.getenv("GEMINI_API_KEY")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel("No Transaction Image Selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; border-radius: 8px; padding: 20px;")
        self.image_label.setFixedHeight(250)
        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Receipt / Transaction Image")
        self.upload_btn.clicked.connect(self.select_image)
        btn_layout.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton("Analyze Transaction")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.analyze_btn)

        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Analysis results will appear here...")
        layout.addWidget(self.result_box)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path).scaled(
                300, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.analyze_btn.setEnabled(True)

    def run_analysis(self):
        if not self.selected_image_path:
            return

        self.result_box.setText("Extracting transaction data via VLM & running Anomaly Detection...")
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.show()

        self.worker = FraudDetectionWorker(self.selected_image_path, self.api_key)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_complete(self, extracted_data, status):
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        result_text = "=== EXTRACTED DATA (VLM OCR) ===\n"
        result_text += json.dumps(extracted_data, indent=4) + "\n\n"
        result_text += f"=== MODEL EVALUATION ===\n{status}"
        
        self.result_box.setText(result_text)

    def on_analysis_error(self, err):
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.result_box.setText(f"Error occurred: {err}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
