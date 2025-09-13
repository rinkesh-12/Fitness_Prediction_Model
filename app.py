import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib

# Load pickled objects
with open("df.pkl", "rb") as f:
    df = pickle.load(f)
# df = joblib.load("df.joblib")

# Load the pickled pipeline (model)
with open("fitness_model.pkl", "rb") as fm:
    model = pickle.load(fm)
# model = joblib.load("fitness_model.joblib")


# Page Config
st.set_page_config(page_title="Fitness Prediction App", layout="wide")
st.title("Fitness Prediction & Analysis Dashboard")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Prediction"])

# Data Analysis
if page == "Data Analysis":
    st.header("Dataset Insights & Statistical Summary")

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Distributions")
    col1, col2= st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["age"], bins=20, kde=True, ax=ax, palette="Set1")
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x=df['is_fit'], y=df['age'], hue=df['gender'], palette="Set2")
        ax.set_title('Average Age by Fitness Status and Gender')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.barplot(x=df['gender'], y=df['age'], palette='pastel')
        ax.set_title("Average Age by Gender")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        sns.countplot(x="is_fit", data=df, ax=ax, palette="Set3")
        ax.set_title("Fitness Class Distribution (Target)")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("This analysis helps understand data balance, feature distribution, and key patterns before modeling.")


# Prediction
elif page == "Prediction":
    st.header("Fitness Prediction")

    # Use min/max/mean from dataset for input sliders
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
        height_cm = st.slider("Height (cm)", int(df['height_cm'].min()), int(df['height_cm'].max()), int(df['height_cm'].mean()))
        weight_kg = st.slider("Weight (kg)", int(df['weight_kg'].min()), int(df['weight_kg'].max()), int(df['weight_kg'].mean()))
        heart_rate = st.slider("Heart Rate (bpm)", int(df['heart_rate'].min()), int(df['heart_rate'].max()), int(df['heart_rate'].mean()))
        blood_pressure = st.slider("Blood Pressure (Systolic)", int(df['blood_pressure'].min()), int(df['blood_pressure'].max()), int(df['blood_pressure'].mean()))

    with col2:
        sleep_hours = st.slider("Average Sleep Hours", int(df['sleep_hours'].min()), int(df['sleep_hours'].max()), int(df['sleep_hours'].mean()))
        nutrition_quality = st.slider("Nutrition Quality (0=Poor, 10=Excellent)", int(df['nutrition_quality'].min()), int(df['nutrition_quality'].max()), int(df['nutrition_quality'].mean()))
        activity_index = st.slider("Activity Index", int(df['activity_index'].min()), int(df['activity_index'].max()), int(df['activity_index'].mean()))
        smokes = st.selectbox("Do you smoke?", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    # Create input dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "heart_rate": heart_rate,
        "blood_pressure": blood_pressure,
        "sleep_hours": sleep_hours,
        "nutrition_quality": nutrition_quality,
        "activity_index": activity_index,
        "smokes": smokes,
        "gender": gender
    }])

    st.write("### Your Input Data")
    st.dataframe(input_data)

    if st.button("🔍 Predict Fitness"):
        # prediction = model.predict(input_data)[0]
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]  # probability of being fit

        if prediction == 1:
            st.success(f"You are likely to be **FIT**! (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"You might be **Unfit**. (Confidence: {(1-prob)*100:.1f}%)")
