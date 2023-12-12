import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Load the pre-trained model and vectorizer
with open("model.pkl", "rb") as file:
    model_and_vectorizer = pickle.load(file)
    final_model = model_and_vectorizer['model']
    vectorizer = model_and_vectorizer['vectorizer']

# Function to preprocess input text and vectorize
def preprocess_and_vectorize(text):
    preprocessed_text = text.lower()  # Example: Convert to lowercase
    tfidf_features = vectorizer.transform([preprocessed_text])
    return tfidf_features

# Function to predict
def predict(text):
    tfidf_features = preprocess_and_vectorize(text)
    prediction = final_model.predict(tfidf_features)[0]
    return prediction

# Streamlit app
def main():
    st.title("Spam Detection App")

    # Input text box
    user_input = st.text_area("Enter the text to classify")

    if st.button("Predict"):
        if user_input:
            prediction = predict(user_input)
            st.write(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
        else:
            st.warning("Please enter text for prediction.")

if __name__ == "__main__":
    main()
