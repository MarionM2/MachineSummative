import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Load the pre-trained model and vectorizer
try:
    with open("model.pkl", "rb") as file:
        model_and_vectorizer = pickle.load(file)
        final_model = model_and_vectorizer['model']  # Load the trained machine learning model
        vectorizer = model_and_vectorizer['vectorizer']  # Load the text vectorizer
except FileNotFoundError:
    st.error("Error: Model file not found. Please make sure 'model.pkl' is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Function to preprocess input text and vectorize
def preprocess_and_vectorize(text):
    preprocessed_text = text.lower()  # Convert to lowercase for consistency
    tfidf_features = vectorizer.transform([preprocessed_text])  # Use the pre-trained vectorizer
    return tfidf_features

# Function to predict
def predict(text):
    try:
        tfidf_features = preprocess_and_vectorize(text)
        prediction = final_model.predict(tfidf_features)[0]  # Make a prediction using the loaded model
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit app
def main():
    st.title("Spam Detection App")

    # Input text box
    user_input = st.text_area("Enter the text to classify")

    if st.button("Predict"):
        if user_input and isinstance(user_input, str):  # Additional input validation
            try:
                prediction = predict(user_input)
                if prediction is not None:
                    st.write(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter valid text for prediction.")

if __name__ == "__main__":
    main()
