# 🧠 Student Burnout Predictor Dashboard

An automated, end-to-end Machine Learning pipeline and interactive web application that predicts student burnout risk based on lifestyle factors and academic pressures. The system benchmarks multiple ML algorithms side-by-side and dynamically serves predictions using the highest-performing model.

🚀 **Live Demo:** [Insert your Streamlit Share link here if you deployed it!]

---

## 📊 Project Overview

Student burnout is a critical issue influenced by complex interactions between study habits, environment, and personal health. This project provides a data-driven approach to tracking risk variables by analyzing **1 Million student profiles**. 

Instead of hardcoding a single predictive model, the core pipeline independently trains and cross-evaluates multiple regression algorithms to automatically select the optimal back-end engine for the user dashboard.

### Key Features
* **Automated Model Selection:** Evaluates multiple algorithms and saves the winner based on $R^2$ Score.
* **Dynamic Performance Leaderboard:** Displays real-time evaluation metrics directly on the web interface.
* **Interactive Frontend:** Built with Streamlit, allowing users to manipulate lifestyle variables and get instant risk assessments.
* **Big Data Foundation:** Core models are optimized for execution against a 1,000,000-row tabular dataset.

---

## 🔬 Dataset & Features

The models are trained on the **Student Mental Health and Burnout** dataset from Kaggle. The following key predictors are engineered into the pipeline:

* `study_hours_per_day`: Average duration dedicated daily to academic preparation.
* `exam_pressure`: Quantitative rank of structural testing stress (Scale 1–5).
* `academic_performance`: Relative index of current performance metrics (Scale 1–5).
* `sleep_hours`: Baseline duration of daily rest.
* `physical_activity`: Total weekly hours allocated to physical fitness.
* `screen_time`: Average daily leisure screen engagement duration.

**Target Variable:** `burnout_score` (Quantitative index representing systematic exhaustion risk).

---

## 📈 Model Evaluation & Benchmarking

The background automated pipeline cross-evaluates three structural variations of regressors. Below are the execution metrics calculated from the 1-million-row dataset:

| Model | RMSE | $R^2$ Score | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **1.3446** | **0.3465** | 🏆 **Selected (Best)** |
| Decision Tree Regressor | 1.3525 | 0.3388 | Defeated |
| Linear Regression | 1.3535 | 0.3378 | Defeated |

*Note: The Random Forest Regressor was configured with constraints (`max_depth=10`, `n_estimators=50`) to optimize memory footprint and speed while mitigating overfitting constraints during large-scale operations.*

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.13
* **Machine Learning Library:** `scikit-learn`
* **Data Processing:** `pandas`, `numpy`
* **Model Serialization:** `joblib`
* **User Interface:** `Streamlit`

---

## 🏃‍♂️ Local Installation & Execution

Follow these steps to spin up the pipeline and local dashboard on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/kapoorhimanshu079-sudo/student-burnout-predictor.git](https://github.com/kapoorhimanshu079-sudo/student-burnout-predictor.git)
cd student-burnout-predictor
