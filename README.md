



# Credit Card Fraud Detection

Credit Card Fraud Detection is a machine learning-based project aimed at identifying and classifying fraudulent transactions from credit card data. The model is trained to distinguish between legitimate and fraudulent activities, providing early detection and minimizing financial loss.

---

## Features

### 🔐 Secure and Intelligent Fraud Detection
- Detects unusual transaction patterns using AI.
- Identifies fraudulent activities in real-time based on transaction behavior.
- Works effectively even with imbalanced datasets using advanced sampling techniques.

### 📊 Multiple ML Models Implemented
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest & XGBoost (Optional Advanced)

### 📈 Evaluation Metrics
- Precision, Recall, F1-Score
- Confusion Matrix & ROC-AUC Curve
- Cross-validation for model reliability

---

## Dataset

- **Name**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (0.172%)
- **Features**: 28 anonymized PCA features + Time + Amount + Class

---

## Technology Stack

- **Programming**: Python 3.10+
- **Libraries**: 
  - `pandas`, `numpy` – Data handling
  - `scikit-learn` – ML algorithms and metrics
  - `matplotlib`, `seaborn` – Visualization
  - `imblearn` – SMOTE oversampling
- **Platform**: Jupyter Notebook / Google Colab

---

## Implementation Steps

1. **Data Loading**: Import and preview the dataset.
2. **Data Preprocessing**:
   - Normalize the `Amount` feature.
   - Apply **SMOTE** for class imbalance handling.
3. **Model Training**:
   - Train multiple classifiers: Logistic Regression, Decision Tree, SVM, KNN.
4. **Model Evaluation**:
   - Evaluate using Precision, Recall, F1-Score, and ROC-AUC.
   - Select the best-performing model.
5. **Visualization**:
   - Plot confusion matrix and ROC curves for interpretability.

---

## File Structure

credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              # Dataset file
│
├── notebooks/
│   └── model_training.ipynb        # Main notebook for training and evaluation
│
├── models/
│   └── best_model.pkl              # Saved trained model (optional)
│
├── utils/
│   └── preprocessing.py            # Helper functions for preprocessing
│
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies

---

## How to Run the Project

### 🖥️ Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

	2.	Install dependencies:

pip install -r requirements.txt

	3.	Run the Jupyter Notebook or Python script to train and evaluate models.

⸻

Results & Performance
	•	Achieved over 98% AUC-ROC on test data.
	•	High recall for fraud class after using SMOTE.
	•	Able to detect rare fraudulent cases with minimal false positives.

⸻

Future Enhancements
	•	🚀 Deploy model as an API (Flask/FastAPI)
	•	🌐 Add web UI for real-time transaction prediction
	•	📉 Integrate streaming data for live fraud detection
	•	🧠 Add Explainable AI tools like SHAP or LIME

⸻

Credits
	•	Dataset: Kaggle – Credit Card Fraud Detection
	•	ML Libraries: Scikit-learn, imbalanced-learn
	•	Visualizations: Matplotlib, Seaborn

⸻

🔐 Stay secure. Detect fraud before it hurts.

Built with ❤️ by Maruthisundar | B.Tech CSE | KL University

---

Let me know if you’d like this exported as a real `.md` file, styled with badges, or linked to a live demo!