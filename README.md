# Fitness Prediction & Analysis Dashboard

A **Streamlit web application** that provides:  
1. **Exploratory Data Analysis (EDA)** of fitness-related features.  
2. **Predictive Modeling** to estimate whether a person is fit or unfit based on lifestyle and health parameters.

---

## Features

- **Data Analysis**
  - View dataset samples and summary statistics.
  - Visualize distributions, bar plots, and correlations.
  - Explore fitness trends across gender, age, and health indicators.

- **Prediction**
  - Input personal details (age, height, weight, heart rate, etc.).
  - Get **real-time fitness predictions** with probability scores.
  - Interactive UI with sliders and selection boxes.

---

## Tech Stack

- **Frontend/UI:** [Streamlit](https://fitnesspredictionmodel.streamlit.app/)
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn (trained model in `fitness_model.pkl`)  
- **Serialization:** Pickle, Joblib  

---

## Project Structure
├── app.py # Main Streamlit app
├── fitness.ipynb # Jupyter Notebook (EDA & Model Training)
├── df.pkl # Dataset (serialized)
├── fitness_model.pkl # Trained ML model
├── requirements.txt # Dependencies
└── README.md # Project Documentation

---

## 📂 Project Structure

