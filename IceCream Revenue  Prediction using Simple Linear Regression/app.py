import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import io

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Title of the Streamlit App
st.title("Revenue Prediction - Simple Linear Regression")

# STEP 1: Problem Statement
st.header("Problem Statement")
st.write('''You own an ice cream business and you would like to create a model that could predict the daily revenue in dollars based on the outside air temperature (degC). You decide that a Linear Regression model might be a good candidate to solve this problem.
Data set:

Independant variable X: Outside Air Temperature
Dependant variable Y: Overall daily revenue generated in dollars''')

# STEP 2: Libraries Import (Streamlit automatically handles library imports)

# STEP 3: Import Dataset
st.header("Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    IceCream = pd.read_csv(uploaded_file)
    st.write("Dataset Head", IceCream.head())
    st.write("Dataset Tail", IceCream.tail())
    st.write("Dataset Description", IceCream.describe())
    st.write("Dataset Info")
    buffer = io.StringIO()
    IceCream.info(buf=buffer)
    st.text(buffer.getvalue())

    # STEP 4: Visualize Dataset
    st.header("Data Visualization")
    st.subheader("Jointplot")
    fig1 = sns.jointplot(x='Temperature', y='Revenue', data=IceCream)
    st.pyplot(fig1)
    
    st.subheader("Pairplot")
    fig2 = sns.pairplot(IceCream)
    st.pyplot(fig2)
    
    st.subheader("Linear Regression Plot")
    fig3 = sns.lmplot(x='Temperature', y='Revenue', data=IceCream)
    st.pyplot(fig3)

    # STEP 5: Create Testing and Training Dataset
    st.header("Training and Testing Dataset")
    y = IceCream['Revenue']
    X = IceCream[['Temperature']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    st.write(f"X_train Shape: {X_train.shape}")

    # STEP 6: Train the Model
    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(X_train, y_train)
    st.write(f"Linear Model Coefficient (m): {regressor.coef_[0]}")
    st.write(f"Linear Model Intercept (b): {regressor.intercept_}")

    # Save the trained model
    with open('icecream_revenue_model.pkl', 'wb') as file:
        pickle.dump(regressor, file)
    st.write('Model saved to icecream_revenue_model.pkl')

    # STEP 7: Test the Model
    y_predict = regressor.predict(X_test)
    st.write("Predicted Values", y_predict)
    st.write("Actual Values", y_test.values)

    # Visualize Train Set Results
    st.subheader("Train Set Results")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.xlabel('Temperature [degC]')
    plt.ylabel('Revenue [dollars]')
    plt.title('Revenue Generated vs. Temperature (Training dataset)')
    st.pyplot(plt)

    # Visualize Test Set Results
    st.subheader("Test Set Results")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, regressor.predict(X_test), color='blue')
    plt.xlabel('Temperature [degC]')
    plt.ylabel('Revenue [dollars]')
    plt.title('Revenue Generated vs. Temperature (Test dataset)')
    st.pyplot(plt)

    # Prediction
    st.header("Make a Prediction")
    temp_input = st.number_input("Enter the Temperature (degC)", min_value=-30.0, max_value=50.0, value=30.0)
    prediction = regressor.predict(np.array([[temp_input]]))
    st.write(f"Predicted Revenue: ${prediction[0]:.2f}")

# Load the trained model from the pickle file and make a prediction
else:
    st.header("Load Pre-trained Model and Make a Prediction")
    pickle_file = st.file_uploader("Choose a pickle file", type="pkl")
    if pickle_file is not None:
        regressor = pickle.load(pickle_file)
        st.write("Model loaded successfully!")
        
        temp_input = st.number_input("Enter the Temperature (degC)", min_value=-30.0, max_value=50.0, value=30.0)
        prediction = regressor.predict(np.array([[temp_input]]))
        st.write(f"Predicted Revenue: ${prediction[0]:.2f}")
