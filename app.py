import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for pro look
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #FF4B4B;
    font-size: 48px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered news authenticity checker</p>', unsafe_allow_html=True)
st.write("")

# Sample news buttons
st.write("### 🎯 Try a sample:")
col1, col2 = st.columns(2)

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

with col1:
    if st.button("📰 Sample REAL News"):
        st.session_state.sample_text = "The Reserve Bank of India announced a new monetary policy during a press conference held in Mumbai on Friday. The decision was made after careful consideration of current economic conditions."

with col2:
    if st.button("⚠️ Sample FAKE News"):
        st.session_state.sample_text = "BREAKING: Scientists confirm Earth is flat! NASA has been lying for decades! Government officials refuse to comment on this shocking discovery!"

# Input
st.write("### ✍️ Enter news text:")
news_text = st.text_area("Paste news article here:", value=st.session_state.sample_text, height=200, label_visibility="collapsed")

# Analyze button
if st.button("🔍 Analyze News", use_container_width=True, type="primary"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("🤖 AI is analyzing..."):
            vectorized = vectorizer.transform([news_text])
            prediction = model.predict(vectorized)[0]
            decision = model.decision_function(vectorized)[0]
            
            # Confidence
            confidence = min(abs(decision) * 100, 99.9)
        
        st.write("")
        if prediction == 1:
            st.markdown('<div class="result-box" style="background-color: #d4edda; color: #155724;">✅ This news appears to be REAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background-color: #f8d7da; color: #721c24;">❌ This news appears to be FAKE</div>', unsafe_allow_html=True)
        
        st.write("### 📊 AI Confidence:")
        st.progress(confidence / 100)
        st.write(f"**{confidence:.1f}%** confident")

# Footer
st.write("")
st.write("---")
st.caption("Built with ❤️ using Streamlit | ML Model: 99.43% accurate | Sabi's AI Project 🚀")