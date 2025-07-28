import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Title
st.title("⛽ Fuel Consumption Prediction")
st.write("Predict fuel consumption based on engine size or number of cylinders using Linear Regression")

# Try to load the file
file_path = r"C:\Users\saish\Downloads\FuelConsumption.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Show sample data
    st.subheader("📊 Sample Data")
    st.dataframe(df.head())

    # Sidebar feature selection
    st.sidebar.header("🔧 Select Feature for Prediction")
    feature_choice = st.sidebar.selectbox("Choose input feature", ["ENGINE SIZE", "CYLINDERS"])

    # Prepare data
    X = df[[feature_choice]]
    y = df[['FUEL CONSUMPTION']]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model info
    a = model.coef_[0][0]
    b = model.intercept_[0]
    st.subheader("📈 Model Parameters")
    st.write(f"**Feature Used:** {feature_choice}")
    st.write(f"**Slope (a):** {a:.4f}")
    st.write(f"**Intercept (b):** {b:.4f}")

    # Sidebar input for prediction
    st.sidebar.subheader("🎯 Predict Fuel Consumption")
    min_val = float(X[feature_choice].min())
    max_val = float(X[feature_choice].max())
    default_val = float(X[feature_choice].mean())
    input_val = st.sidebar.slider(f"{feature_choice}", min_val, max_val, default_val, step=0.1)

    predicted_fuel = model.predict([[input_val]])[0][0]
    st.sidebar.success(f"Predicted Fuel Consumption: **{predicted_fuel:.2f}**")

    # Plot 1
    st.subheader(f"📉 {feature_choice} vs Fuel Consumption with Prediction")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=feature_choice, y='FUEL CONSUMPTION', data=df, ax=ax1)
    sns.lineplot(x=X_test[feature_choice], y=y_pred.flatten(), color='red', label='Regression Line', ax=ax1)
    ax1.scatter(input_val, predicted_fuel, color='green', s=100, label='Your Prediction')
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2
    st.subheader("🟡 Scatter Plot: Engine Size & Cylinders vs Fuel Consumption")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='ENGINE SIZE', y='FUEL CONSUMPTION', data=df, label='Engine Size', color='blue', ax=ax2)
    sns.scatterplot(x='CYLINDERS', y='FUEL CONSUMPTION', data=df, label='Cylinders', color='orange', ax=ax2)
    ax2.set_title("Combined Scatter Plot")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.error("❌ File not found. Please make sure `FuelConsumption.csv` exists at this path:\n"
             f"`{file_path}`")
