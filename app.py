import streamlit as st
import joblib

# Page configuration
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# Custom CSS with background graphics and styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body {
            font-family: 'Poppins', sans-serif;
            background-color: #f9fafc;
            
            background-size: cover;
        }

     

        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        p.subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #ced6e0;
            padding: 1rem;
            font-size: 1rem;
        }

        .stButton button {
            background-color: #1E90FF;
            color: white;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border: none;
            border-radius: 10px;
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #F0F8FF;  /* Light blue on hover */
            color: #27ae60;             /* Optional: green text on hover */
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 15px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }

        .spam {
            background-color: #ffe6e6;
            color: #c0392b;
            border: 2px solid #e74c3c;
        }

        .not-spam {
            background-color: #eaffea;
            color: #27ae60;
            border: 2px solid #2ecc71;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('spam_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Main container
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown("<h1>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quickly check whether a message is spam or not using AI!</p>', unsafe_allow_html=True)

sms = st.text_area("✍️ Enter your SMS message below:")

if st.button("🔍 Predict"):
    if sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        vectorized_sms = vectorizer.transform([sms])
        prediction = model.predict(vectorized_sms)[0]
        proba = model.predict_proba(vectorized_sms)[0]
        confidence = round(proba[prediction] * 100, 2)

        if prediction == 0:
            st.markdown(f'<div class="result-box not-spam">✅ This message is <strong>Not Spam</strong>.', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box spam">🚫 This message is <strong>Spam</strong>.', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ...existing code...

# Footer
st.markdown(
    """
    <hr style="margin-top:2rem;margin-bottom:0.5rem;">
    <div style="text-align:center; color:#7f8c8d; font-size:0.95rem;">
        Made by Palak Gupta | &copy; 
    </div>
    """,
    unsafe_allow_html=True
)