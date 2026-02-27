📌 Credit Default Risk Prediction — Production MLOps System

A production-oriented machine learning system for credit default risk prediction, designed with reproducibility, experiment tracking, model versioning and containerized deployment in mind.

This project demonstrates how to move beyond notebook experimentation and build a deployable, versioned ML system.


---

🚀 Architecture Overview

The system consists of the following components:

Component	Role

Data Pipeline	Data cleaning and feature engineering
MLflow	Experiment tracking + Model Registry
Backend Store	SQLite (for metadata tracking)
Artifact Store	Model artifact persistence
FastAPI	Real-time inference service
Docker	Containerized production environment
GitHub Actions	Automated CI build validation


Workflow

1. Raw data is preprocessed and validated


2. Feature engineering pipeline is applied


3. Model is trained and evaluated


4. Experiments logged to MLflow


5. Best model registered in Model Registry


6. FastAPI loads the production model


7. Docker ensures reproducible deployment


8. CI validates build integrity




---

🧠 Key Engineering Decisions

Separation of backend store and artifact store

Modular feature engineering

Registry-based model promotion

Environment-based configuration

Containerized reproducibility

Graceful handling when model is not loaded



---

🛠️ Tech Stack

Python
Scikit-learn
MLflow
FastAPI
Docker
GitHub Actions
Pydantic


---

📂 Project Structure

credit-default-mlops/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── scripts/
│   ├── train.py
│   ├── utils.py
│   ├── schema.py
│   └── config.py
│
├── models/
├── docker-compose.yml
├── .env
└── README.md


---

⚙️ How to Run Locally

Prerequisites

Docker Desktop

Git


Steps

git clone https://github.com/fatimaammara180-ML/credit-default-mlops.git
cd credit-default-mlops
docker compose up --build

FastAPI: http://localhost:8000
MLflow UI: http://localhost:5000


---

🔌 API Endpoint

POST /predict

Example:

{
  "feature1": 1.5,
  "feature2": 45.0,
  "feature3": 22.3
}

Response:

{
  "prediction": 1,
  "default_probability": 0.78
}


---

📌 Lessons Learned

Accuracy is not the main challenge in ML systems

Reproducibility matters more than experimentation speed

Model versioning is essential for safe deployment

Containerization reduces environment inconsistencies



---

🔮 Future Improvements

Add monitoring layer

Add drift detection integration

Add automated model promotion workflow

Deploy to cloud infrastructure



👩‍💻 Author

Ammara Fatima



