import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the best pre-trained model
model = joblib.load('best_burnout_model.pkl')

st.title("🧠 Student Burnout Predictor Dashboard")
st.write("Predict your burnout risk using the optimal Machine Learning algorithm trained on 1 Million student profiles.")

# --- Model Leaderboard Section ---
st.subheader("📊 Model Performance Leaderboard")
try:
    results_df = pd.read_csv('model_results.csv')
    st.dataframe(results_df.style.highlight_max(
        axis=0, subset=['R2 Score'], color='#d4edda'))
    st.caption(
        "The algorithm with the highest R² Score was automatically selected to power this dashboard.")
except FileNotFoundError:
    st.info("Run your training script first to see the model comparison chart here!")

st.markdown("---")

# --- User Inputs based on Dataset Columns ---
st.subheader("🔮 Input Student Habits & Stressors")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Daily Study Hours", 0.0, 16.0, 6.0, step=0.5)
    exam_pressure = st.slider("Exam Pressure Level (1-5)", 1, 5, 3)
    academic_performance = st.slider(
        "Academic Performance Level (1-5)", 1, 5, 3)

with col2:
    sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    physical_activity = st.slider(
        "Weekly Physical Activity Hours", 0.0, 15.0, 3.0, step=0.5)
    screen_time = st.slider(
        "Daily Leisure Screen Time Hours", 0.0, 12.0, 4.0, step=0.5)

# --- Prediction Action ---
if st.button("Calculate Burnout Risk"):
    # Group inputs in the exact shape the model expects
    features = np.array([[study_hours, exam_pressure, academic_performance,
                        sleep_hours, physical_activity, screen_time]])
    prediction = model.predict(features)[0]

    # Format the display based on the typical dataset range (assuming 0-1 or 0-100 scales)
    st.subheader(f"Predicted Burnout Score: {prediction:.2f}")

    if prediction > 0.7 or prediction > 70:  # Adapts gracefully to whether your data is percentage or decimal based
        st.error(
            "🚨 High risk of burnout! Time to step back, adjust priorities, and rest.")
    elif prediction > 0.4 or prediction > 40:
        st.warning(
            "⚠️ Moderate risk. Check your balance between work and rest hours.")
    else:
        st.success(
            "✅ Low risk! You're managing expectations and boundaries well.")
