# Term-Deposit-Subscription-Prediction
* Developed a Term Deposit Subscription Prediction model using machine learning to identify high-potential banking customers. Deployed a Streamlit web application enabling real-time predictions with automated   preprocessing and probability-based decision support.
* A machine learning project that predicts whether a bank customer will subscribe to a term deposit, helping financial institutions optimize marketing campaigns.

## 🔍 Objective
To classify customers as likely or unlikely to subscribe using historical banking and campaign data.

## 🧠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Streamlit
- Pickle

## ⚙️ Model Pipeline
- Data preprocessing & encoding
- Feature alignment using trained model schema
- Classification with probability output
- End-to-end deployment via Streamlit

## 🌐 Web App Features
- Interactive customer input form
- Real-time subscription prediction
- Confidence score display

## 📁 Project Structure
- `main.py` – Streamlit deployment
- `*.pkl` – Trained ML model
- `*.ipynb` – Model training notebook

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run main.py
