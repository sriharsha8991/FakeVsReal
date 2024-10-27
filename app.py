import streamlit as st
from utils import load_model, predict_news, load_tfidf
from visualisations import plot_distribution, plot_prediction_confidence, plot_wordcloud

# Load the model and TF-IDF vectorizer
model, tfidf = load_model(), load_tfidf()

def main():
    st.set_page_config(page_title='Fake News Detection Dashboard', layout='wide')
    st.title("Fake News Detection Dashboard")
    st.markdown("This dashboard analyzes text to determine whether news is likely to be real or fake. Enter text below and hit predict to see results.")
    
    # Sidebar for user input
    st.sidebar.header("User Input Features")
    text = st.sidebar.text_area("Enter the news text here:", height=10)
    button_pressed = st.sidebar.button("Predict")

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Column 1: Prediction and Confidence
    with col1:
        if button_pressed:
            prediction, probability = predict_news(model, tfidf, text)
            if prediction == 1:
                st.success("This news is predicted as **Real**.")
                st.metric(label="Confidence", value=f"{probability[1]*100:.2f}%")
            else:
                st.error("This news is predicted as **Fake**.")
                st.metric(label="Confidence", value=f"{probability[0]*100:.2f}%")
        
            st.write("## Prediction Confidence")
            st.pyplot(plot_prediction_confidence(probability))
    
    # Column 2: Visualizations
    with col2:
        st.write("## News Type Distribution")
        st.pyplot(plot_distribution(11599, 11785))  # Replace these with actual counts from your dataset
        
        if text:
            st.write("## Word Cloud for Input Text")
            st.pyplot(plot_wordcloud(text))

if __name__ == "__main__":
    main()
