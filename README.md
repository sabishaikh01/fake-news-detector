# 📰 Fake News Detector using AI & Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

A real-time Fake News Detection web app powered by Machine Learning and Natural Language Processing. Paste any news article or headline and instantly know if it's REAL or FAKE! 🚀

## 🌐 Live Demo
👉 **[Click here to try the app!](https://sabishaikh01-fake-news-detector.streamlit.app)**

## ✨ Features
- 🤖 AI-powered news classification
- ⚡ Real-time predictions
- 📊 Confidence score for each prediction
- 🎨 Clean and modern UI
- 📱 Mobile-friendly design

## 🛠️ Tech Stack
- **Python 3.11**
- **Scikit-learn** (Machine Learning)
- **Streamlit** (Web Framework)
- **Pickle** (Model Serialization)
- **NLP** (Natural Language Processing)

## 📂 Project Structure

fake-news-detector/
├── app.py              # Streamlit web app
├── train_model.py      # Model training script
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## 🚀 How to Run Locally

1. Clone the repository
   git clone https://github.com/sabishaikh01/fake-news-detector.git

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

4. Open browser at http://localhost:8501

## 🧠 How It Works
1. User pastes a news article/headline
2. Text is converted to numerical features using TF-IDF Vectorization
3. A trained Machine Learning Classifier analyzes the text
4. The model predicts whether the news is REAL or FAKE with a confidence score

## 📊 Model Performance
- Accuracy: ~95%
- Algorithm: Passive Aggressive Classifier
- Dataset: Kaggle Fake News Dataset

## 💡 What I Learned
- Text preprocessing and NLP techniques
- TF-IDF vectorization for text data
- Training and evaluating ML classifiers
- Model serialization using Pickle
- Deploying ML models as web apps with Streamlit
- Git and GitHub for version control

## 🔮 Future Improvements
- Add deep learning models (LSTM, BERT)
- Multi-language support
- Source credibility checker
- Browser extension
- API endpoint for integration

## 👨‍💻 Author
**Sabi Shaikh**
- GitHub: @sabishaikh01
- Email: sabishaikhnasa01@gmail.com

## 📄 License
This project is open source and available under the MIT License. ⭐ If you like this project, please give it a star! ⭐
