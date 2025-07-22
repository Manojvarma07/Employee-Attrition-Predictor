🧠 Employee Attrition Predictor
This project is an AI-powered web application built with Streamlit that predicts the likelihood of employee attrition in a company using a trained CatBoost Classifier. It offers real-time insights, visualizations, and interactive analytics to support HR decision-making.

📌 Features
🚀 Streamlit-based Web UI – Easy-to-use, responsive interface.

📊 Real-time Data Upload – Accepts custom HR datasets in CSV format.

🧠 CatBoost Classifier – Highly accurate ML model trained for binary classification.

📈 Live Metrics & Visuals – Interactive charts, confusion matrix, ROC curve, and more.

🔍 Employee-wise Prediction – Understand attrition likelihood for individual records.

🧰 Tech Stack
Frontend/UI: Streamlit, HTML/CSS
Backend: Python
ML Model: CatBoostClassifier
Visualization: Plotly

📁 Employee-Attrition-Predictor
├── app.py              
├── data.csv            
├── model.pkl           
├── requirements.txt    
├── .env                
└── README.md      

🔄 How It Works
Upload an HR dataset (CSV format)
App preprocesses the data
Trained CatBoost model predicts attrition probability
Metrics and charts help interpret the predictions
