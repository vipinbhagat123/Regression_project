import streamlit as st
import joblib
from PIL import Image
from spam_emails import spam_emails
from non_spam_emails import non_spam_emails


def classify_email(test_input):
    input_data_features = tfidf_vectorizer.transform([test_input])
    prediction = model.predict(input_data_features)
    return prediction[0]

# Navigation Bar
nav_selection = st.sidebar.radio("Navigation", ["Home", "Spam Email List", "Not Spam Email List"])


if nav_selection == "Spam Email List":
    st.header("Spam Email List")
    for email in spam_emails:
        st.code(email)
elif nav_selection == "Not Spam Email List":
    st.header("Not Spam Email List")
    for email in non_spam_emails:
        st.code(email)
else:
    st.write('<h3>Email Spam Classification <span style="color:#EE7214;">(Logistic Regression)</span></h3>', unsafe_allow_html=True)
    st.caption('Empower Your Inbox: Effortlessly Distinguish Spam from Legitimate Emails with our Email Spam Classification Tool')
    # Loading the trained Logistic Regression model
    model = joblib.load('models/email_model.pkl')
    # Loading the TF-IDF vectorizer from the file
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    # Getting the user input
    input_text = st.text_area("Enter an email text to classify:", max_chars=500)

    if st.button("Classify"):
        if input_text:
            result = classify_email(input_text)
            if result == 1:
                background = Image.open("assets/alert.png")
                col1, col2, col3 = st.columns(3)
                col2.image(background, use_column_width=True, width=10)
                st.write("Prediction:", "<b style='color:#FC4100;'>Spam Eamil</b>", unsafe_allow_html=True)
            else:
                background = Image.open("assets/ok.png")
                col1, col2, col3 = st.columns(3)
                col2.image(background, use_column_width=True, width=10)
                st.write("Prediction:", "<b style='color:#65B741;'>None Spam Email</b>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to classify.")
