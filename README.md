# Customer Churn Prediction

A machine learning web app built with **Streamlit** that predicts whether a customer is likely to churn, helping businesses take proactive retention actions.

🔗 **Live Demo**: [customer-churn-prediction-ishika.streamlit.app](https://customer-churn-prediction-ishika.streamlit.app)

---

## Features

- Interactive UI for entering customer data
- Real-time churn probability prediction
- Visual insights using Plotly, Matplotlib, and Seaborn
- Built on scikit-learn's ML pipeline

---

## Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/Ishika-guptaa25/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run churn_prediction_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Frontend | Streamlit |
| ML | scikit-learn, NumPy, pandas |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## Project Structure

```
Customer-Churn-Prediction/
├── churn_prediction_app.py   # Main Streamlit app
└── requirements.txt          # Python dependencies
```
