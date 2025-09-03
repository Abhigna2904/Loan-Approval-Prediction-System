# 🏦 Loan Approval Prediction System

## 📌 Overview
This project predicts whether a loan application will be **approved or rejected** based on applicant details such as income, credit history, employment, and more.  
It uses **Machine Learning models (Logistic Regression, Random Forest)** for training and is deployed as an interactive **Streamlit web application**.

---

## 🚀 Features
- Loan approval prediction using trained ML models.
- Interactive **Streamlit web app** for real-time user inputs.
- Pre-trained model (`loan_model.pkl`) for instant predictions.
- Simple UI with success/error messages for predictions.

---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **NumPy / Pandas**

---

## 📂 Project Structure
- train.py # Model training script
- app.py # Streamlit web application
- loan_model.pkl # Pre-trained loan prediction model
- README.md # Project documentation
- requirements.txt

---

## ⚙️ Installation & Usage

1. Clone the repository:
  - git clone https://github.com/your-username/loan-prediction.git
  - cd loan-prediction

2. Install dependencies:
- pip install -r requirements.txt

3. Run the Streamlit app:
- streamlit run app.py

