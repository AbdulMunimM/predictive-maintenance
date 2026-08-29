# ⚙️ AI Predictive Maintenance & Machine Condition Monitoring

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Open-FF4B4B?style=for-the-badge\&logo=streamlit)](https://predictive-maintenance-am.streamlit.app/)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge)](https://predictive-maintenance-3p11.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-85EA2D?style=for-the-badge\&logo=swagger)](https://predictive-maintenance-3p11.onrender.com/docs)


An end-to-end machine learning system for **predictive maintenance, machine failure prediction, failure-mode identification, explainable AI, and maintenance decision support** using multivariate machine operating data from the AI4I 2020 Predictive Maintenance Dataset.

The project combines **XGBoost, multi-label classification, SHAP explainability, FastAPI, and Streamlit** into a deployable predictive-maintenance application.

---

## 🚀 Project Overview

Unexpected machine failures can lead to production downtime, equipment damage, maintenance costs, and operational disruption.

This project develops an intelligent predictive-maintenance system that answers three practical questions:

> **1. Is the machine likely to fail?**
> **2. What failure mode is most likely?**
> **3. Why did the model make that prediction?**

The final application supports two data-input workflows:

### ✋ Manual Analysis

An operator can enter a specific machine reading and immediately receive a failure-risk assessment.

### 📡 Sensor Stream Replay

Historical AI4I machine records are automatically replayed as a **simulated sensor stream**, allowing the system to demonstrate continuous monitoring without requiring physical IoT hardware.

Both workflows send the same machine measurements to the same FastAPI backend and use the same trained models.

---

# 🎯 Objectives

The project was designed to:

* Predict machine failure before it occurs.
* Address the strong class imbalance in machine-failure data.
* Identify one or multiple possible failure modes.
* Engineer useful relationships between machine operating variables.
* Explain model predictions using SHAP.
* Convert model predictions into actionable maintenance recommendations.
* Expose the ML system through REST APIs.
* Provide an operator-friendly monitoring dashboard.
* Demonstrate both manual diagnosis and automated sensor-stream monitoring.
* Deploy the system as independent backend and frontend services.

---

# 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains **10,000 machine operating observations** covering machine type, temperature, rotational speed, torque, tool wear, machine failure, and failure-mode indicators.

## Original predictive variables

| Feature                   | Description                   |
| ------------------------- | ----------------------------- |
| `Type`                    | Machine quality/type category |
| `Air temperature [K]`     | Air temperature               |
| `Process temperature [K]` | Process temperature           |
| `Rotational speed [rpm]`  | Machine rotational speed      |
| `Torque [Nm]`             | Machine torque                |
| `Tool wear [min]`         | Accumulated tool wear         |

## Primary target

`Machine failure`

The failure class is highly imbalanced, so evaluation focuses on metrics such as **precision, recall, F1-score, ROC-AUC, and PR-AUC**, rather than relying on accuracy alone.

## Failure-mode targets

The final failure-mode model uses:

* `TWF` — Tool Wear Failure
* `HDF` — Heat Dissipation Failure
* `PWF` — Power Failure
* `OSF` — Overstrain Failure

The failure-mode problem is treated as **multi-label classification**, since a single observation can contain multiple failure-mode labels.

---

# 🧠 Feature Engineering

The final models use the original machine measurements together with three engineered features.

## 1. Temperature Difference

```python
Temp_Diff = Process temperature [K] - Air temperature [K]
```

This represents the temperature difference between the process and surrounding air.

## 2. Torque-Speed Interaction

The original project implementation used:

```python
Power_W = Rotational speed [rpm] * Torque [Nm]
```

Despite the original variable name `Power_W`, this calculation is **not physical power in watts** because rotational speed has not been converted from RPM to angular velocity.

Therefore, it is more accurately interpreted as a:

> **Torque-speed interaction / power proxy**

It captures the combined operating effect of torque and rotational speed and was retained because the trained models were built using this exact definition.

## 3. Overstrain Index

```python
Overstrain_Index = Torque [Nm] * Tool wear [min]
```

This feature captures the combined relationship between mechanical load and accumulated tool wear.

## Final feature set

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Temp_Diff
Power_W
Overstrain_Index
```

The engineered features are created automatically during inference. Users do not need to calculate them manually.

---

# 🤖 Machine Learning Architecture

The final system contains two supervised predictive models.

```text
Machine Operating Data
          ↓
   Feature Engineering
          ↓
    ┌─────┴─────┐
    ↓           ↓
 Model 1       Model 2
    ↓           ↓
Failure      Failure Modes
Prediction   TWF/HDF/PWF/OSF
    │           │
    └─────┬─────┘
          ↓
Maintenance Decision
          ↓
       SHAP
```

---

# 1️⃣ Model 1 — Machine Failure Prediction

## Objective

Predict whether a machine is likely to experience failure.

## Algorithm

**XGBoost Classifier**

## Decision threshold

The final threshold is:

```text
0.80
```

A machine is classified as a failure risk when:

```text
Failure probability ≥ 0.80
```

## Final test performance

The final evaluation was performed on a held-out test set containing **2,000 observations**.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **99.15%** |
| Precision | **91.80%** |
| Recall    | **82.35%** |
| F1-score  | **86.82%** |
| ROC-AUC   | **98.64%** |
| PR-AUC    | **89.14%** |

## Confusion matrix

```text
                 Predicted
                No Failure  Failure

Actual
No Failure        1927        5
Failure             12       56
```

The selected threshold detected:

```text
56 / 68 = 82.35%
```

of the failure cases in the final test set while producing only **5 false positives**.

---

# 2️⃣ Model 2 — Failure Mode Classification

Once Model 1 identifies a machine as a failure risk, Model 2 determines which labeled failure mode or combination of failure modes is likely.

## Model structure

The system uses a multi-output architecture consisting of four XGBoost classifiers:

```text
                  Model 2
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
     TWF           HDF           PWF           OSF
```

The independent outputs allow combinations such as:

```text
PWF + OSF
HDF + OSF
TWF + PWF + OSF
```

## Failure-mode dataset

Only observations containing at least one labeled failure mode were used:

```text
330 observations
```

| Failure Mode | Positive Cases |
| ------------ | -------------: |
| TWF          |             46 |
| HDF          |            115 |
| PWF          |             95 |
| OSF          |             98 |

## Data split

| Split      | Observations |
| ---------- | -----------: |
| Training   |          235 |
| Validation |           44 |
| Test       |           51 |

Iterative multi-label stratification was used to preserve failure-mode distributions across the splits.

## Decision thresholds

The final threshold for each mode is:

```text
TWF = 0.50
HDF = 0.50
PWF = 0.50
OSF = 0.50
```

## Final test performance

| Metric       |      Result |
| ------------ | ----------: |
| Hamming Loss | **0.00490** |
| Micro F1     |  **99.07%** |
| Macro F1     |  **99.14%** |

### Individual failure-mode performance

| Failure Mode | Precision | Recall |       F1 |
| ------------ | --------: | -----: | -------: |
| TWF          |      1.00 |   1.00 | **1.00** |
| HDF          |      1.00 |   1.00 | **1.00** |
| PWF          |      1.00 |   1.00 | **1.00** |
| OSF          |      1.00 |   0.93 | **0.97** |

The only error in the test set was one missed `OSF` case.

Because the Model 2 test set is small, these results should be interpreted as performance on this specific held-out split rather than as a guarantee of production performance.

---

# 🔍 Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to make machine-failure predictions interpretable.

SHAP is used to answer:

> **Why did the model flag this machine?**

## Global Model 1 importance

The final Model 1 produced the following ranking based on mean absolute SHAP values:

| Rank | Feature                  | Mean |SHAP| |
| ---: | ------------------------ | ----------: |
|    1 | Tool wear                |   **1.432** |
|    2 | Rotational speed         |   **1.286** |
|    3 | Torque-speed interaction |   **1.194** |
|    4 | Torque                   |   **0.911** |
|    5 | Overstrain Index         |   **0.752** |
|    6 | Temperature Difference   |   **0.663** |

These values indicate how strongly the trained model uses each feature across the dataset. They describe **model behavior**, not physical causality.

## Failure-mode explainability

The failure-mode models showed distinct dominant features:

| Failure Mode | Dominant Feature         |
| ------------ | ------------------------ |
| TWF          | Tool wear                |
| HDF          | Temperature Difference   |
| PWF          | Torque-speed interaction |
| OSF          | Overstrain Index         |

This suggests that the different failure-mode classifiers rely on different machine-condition patterns.

## Local explanations

For an individual high-risk machine, the application can request a SHAP explanation through:

```http
POST /explain
```

The Streamlit interface exposes this through:

```text
🔎 Why was this machine flagged?
```

The explanation identifies features pushing the model prediction toward or away from failure.

---

# 🏗️ End-to-End Architecture

The production system is structured as two independent applications:

```text
                    ┌────────────────────────┐
                    │    Streamlit Cloud      │
                    │                        │
                    │  Manual Analysis       │
                    │  Sensor Stream Replay  │
                    └───────────┬────────────┘
                                │
                              HTTPS
                                │
                                ▼
                    ┌────────────────────────┐
                    │        Render          │
                    │       FastAPI          │
                    │                        │
                    │       /health          │
                    │       /predict         │
                    │       /explain         │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Feature Engineering    │
                    │                        │
                    │ Temp_Diff              │
                    │ Torque-Speed Proxy     │
                    │ Overstrain_Index       │
                    └───────────┬────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              ┌─────────────┐      ┌──────────────┐
              │   Model 1   │      │   Model 2    │
              │   XGBoost    │      │ Multi-label  │
              │             │      │  XGBoost     │
              │ Machine     │      │ TWF/HDF/PWF/ │
              │ Failure     │      │ OSF          │
              └──────┬──────┘      └──────┬───────┘
                     │                    │
                     └──────────┬─────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Maintenance Decision   │
                    └───────────┬────────────┘
                                │
                                ▼
                         SHAP Explanation
```

---

# 🖥️ Dashboard

The Streamlit dashboard provides two operating modes.

## Manual Analysis

An operator can enter:

```text
Machine Type
Air Temperature
Process Temperature
Rotational Speed
Torque
Tool Wear
```

The dashboard sends those measurements to:

```http
POST /predict
```

and displays:

* Failure probability
* Machine status
* Likely failure mode(s)
* Failure-mode probabilities
* Maintenance recommendation
* Manual analysis history
* SHAP explanation for flagged machines

Every manual analysis performed during the current Streamlit session is stored in a **Manual Analysis History** table.

---

## Sensor Stream Replay

The application can automatically replay the AI4I dataset:

```text
ai4i2020.csv
```

as a simulated sensor stream.

The workflow is:

```text
AI4I Record
    ↓
Streamlit
    ↓
POST /predict
    ↓
FastAPI
    ↓
Model 1 + Model 2
    ↓
Prediction
    ↓
Dashboard update
    ↓
Next record
```

The dashboard shows the current sensor readings, current prediction, failure modes, maintenance recommendation, and recent prediction history.

> **Important:** Sensor Stream Replay is a simulation based on historical AI4I records. It does not represent a live physical IoT sensor connection.

---

# 🌐 API

The backend is implemented using **FastAPI**.

## Health check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model1": "loaded",
  "model2": "loaded"
}
```

## Machine prediction

```http
POST /predict
```

Example request:

```json
{
  "Type": "L",
  "air_temperature": 300.8,
  "process_temperature": 309.9,
  "rotational_speed": 1312,
  "torque": 65.3,
  "tool_wear": 192
}
```

Example response:

```json
{
  "failure_probability": 0.9995548,
  "machine_failure": true,
  "failure_modes": {
    "TWF": {
      "probability": 0.001234,
      "predicted": false
    },
    "HDF": {
      "probability": 0.003230,
      "predicted": false
    },
    "PWF": {
      "probability": 0.042811,
      "predicted": false
    },
    "OSF": {
      "probability": 0.989143,
      "predicted": true
    }
  },
  "maintenance_status": "CRITICAL",
  "recommended_action": "Inspect machine immediately. Likely failure mode(s): OSF."
}
```

## SHAP explanation

```http
POST /explain
```

Returns the base model score and the most influential feature contributions for the submitted machine.

## Interactive API documentation

```text
https://predictive-maintenance-3p11.onrender.com/docs
```

---

# 📁 Project Structure

```text
predictive-maintenance/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── dashboard.py
│   └── requirements.txt
│
├── artifacts/
│   ├── model1_xgboost.ubj
│   ├── model1_preprocessor.joblib
│   ├── model2_TWF.ubj
│   ├── model2_HDF.ubj
│   ├── model2_PWF.ubj
│   ├── model2_OSF.ubj
│   ├── model2_preprocessor.joblib
│   └── project_config.joblib
│
├── notebooks/
│   └── Predictive_Maintenance.ipynb
│
├── ai4i2020.csv
├── .gitignore
├── .python-version
└── README.md
```

---

# 🛠️ Technology Stack

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP

## Backend

* FastAPI
* Uvicorn

## Frontend

* Streamlit

## Development & Deployment

* Git
* GitHub
* Render
* Streamlit Community Cloud

---

# ▶️ Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/AbdulMunimM/predictive-maintenance.git
cd predictive-maintenance
```

## 2. Create the Python environment

The backend uses Python 3.13.

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

## 4. Configure the local Streamlit secret

Create:

```text
frontend/.streamlit/secrets.toml
```

with:

```toml
API_BASE_URL = "http://127.0.0.1:8000"
```

Do not commit this file.

## 5. Start the FastAPI backend

From the repository root:

```bash
python -m uvicorn backend.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 6. Start the Streamlit dashboard

In another terminal:

```bash
pip install -r frontend/requirements.txt
python -m streamlit run frontend/dashboard.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

The application uses separate cloud deployments for the backend and frontend.

## FastAPI Backend — Render

The FastAPI service is deployed from GitHub to Render.

Build command:

```text
pip install -r backend/requirements.txt
```

Start command:

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

Production API:

```text
https://predictive-maintenance-3p11.onrender.com
```

## Streamlit Frontend — Streamlit Community Cloud

The frontend is deployed from:

```text
frontend/dashboard.py
```

The production backend URL is provided through Streamlit Secrets:

```toml
API_BASE_URL = "https://predictive-maintenance-3p11.onrender.com"
```

This keeps the API configuration separate from the source code.

---

# 🔬 Machine Learning Development Workflow

The project followed a structured ML workflow:

```text
Dataset Understanding
        ↓
Data Quality Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Validation / Test Strategy
        ↓
Model Comparison
        ↓
Threshold Selection
        ↓
Final Test Evaluation
        ↓
SHAP Explainability
        ↓
Model Serialization
        ↓
Inference Pipeline
        ↓
FastAPI
        ↓
Streamlit
        ↓
Cloud Deployment
```

---

# 🧪 Approaches Evaluated but Not Included in the Final Product

The project considered additional techniques during development, but they were not retained in the final operator-facing system.

## Anomaly Detection

Isolation Forest was evaluated as an additional abnormal-behavior detector.

It showed that known failures were more likely to be flagged as anomalous than non-failures. However, anomaly predictions could disagree with the supervised failure model and did not provide enough additional actionable value for the final application.

Therefore:

> **Anomaly detection was evaluated but intentionally excluded from the final product.**

The final application focuses on the two primary maintenance questions:

```text
Will the machine fail?
What failure mode is likely?
```

## Remaining Useful Life

RUL prediction was also investigated.

An initially very strong RUL result was identified as suffering from information leakage because `Tool wear` was directly related to the constructed RUL target.

After removing leakage-prone information, RUL performance became weak.

Therefore:

> **RUL prediction was not included in the final system because the available dataset did not support a sufficiently reliable RUL model under a clean evaluation setup.**

This avoids presenting an apparently strong but methodologically unreliable model.

---

# ⚠️ Limitations

## Dataset

The AI4I 2020 dataset is a benchmark dataset rather than a real production fleet.

The Sensor Stream Replay therefore demonstrates automated monitoring using historical records rather than actual physical sensor telemetry.

## Failure-mode test size

Model 2 was evaluated on only:

```text
51 test observations
```

with relatively few positive samples for some modes. Its very high metrics should therefore be interpreted cautiously.

## Generalization

The reported performance reflects the held-out dataset used in this project. Real industrial deployment would require validation on data from real machines, different operating environments, unseen assets, and longer operational histories.

## Explainability

SHAP describes how features influenced the trained model's prediction. It does not prove that those features are physical causes of failure.

## Simulation vs. real-time sensors

The current application uses historical dataset replay to simulate sensor ingestion. A true production system would require an actual telemetry pipeline such as MQTT, HTTP sensor ingestion, OPC UA, or another industrial communication protocol.

---

# 🔮 Future Improvements

Potential extensions include:

* Real-time IoT sensor ingestion.
* MQTT or industrial protocol integration.
* Persistent machine-level prediction history.
* Multi-machine monitoring.
* Automated alerts and maintenance notifications.
* Model monitoring and concept-drift detection.
* Automated retraining pipelines.
* Real-world RUL prediction using true degradation trajectories.
* Integration with CMMS or industrial asset-management systems.
* User authentication and role-based dashboards.
* Historical trend visualization for machine health.

---

# 📌 Example End-to-End Prediction

For the following machine reading:

```text
Type: L
Air Temperature: 300.8 K
Process Temperature: 309.9 K
Rotational Speed: 1312 rpm
Torque: 65.3 Nm
Tool Wear: 192 min
```

the production system predicts approximately:

```text
Failure Probability: 99.96%

Machine Failure: YES

TWF: 0.12%
HDF: 0.32%
PWF: 4.28%
OSF: 98.91%

Likely Failure Mode: OSF

Maintenance Status: CRITICAL
```

The resulting recommendation is:

```text
Inspect machine immediately.
Likely failure mode(s): OSF.
```

The operator can then request a SHAP explanation to understand the main feature contributions behind the prediction.

---

# 🎓 Project Outcome

This project demonstrates a complete machine-learning engineering workflow rather than a standalone model or notebook.

It integrates:

```text
Data Analysis
     +
Feature Engineering
     +
Imbalanced Classification
     +
Multi-label Learning
     +
Explainable AI
     +
Model Serialization
     +
REST API
     +
Interactive Dashboard
     +
Automated Sensor Replay
     +
Cloud Deployment
```

The final system can:

> **Receive machine operating conditions, estimate failure risk, identify likely failure modes, explain the prediction, and translate the result into an actionable maintenance recommendation.**

The two frontend workflows make the system useful both for **individual machine diagnosis** and for demonstrating **automated sensor-driven monitoring**.

---

## Author

**Abdul Munim**

GitHub:
https://github.com/AbdulMunimM/predictive-maintenance
