import pickle
from text_preprocessing import preprocess_text

def load_model():
    with open('model (1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_tfidf():
    # Load the TF-IDF vectorizer from the saved file
    with open('tfidf_vectorizer (1).pkl', 'rb') as file:
            loaded_tfidf = pickle.load(file)
    return loaded_tfidf

def predict_news(model, tfidf, text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf.transform([processed_text])
    prediction = model.predict(vectorized_text)
    probability = model.predict_proba(vectorized_text)[0]
    return prediction[0], probability
