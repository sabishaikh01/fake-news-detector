import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
fake = pd.read_csv(r"C:\Users\Admin\Desktop\fake-news-detector\Fake.csv")
true = pd.read_csv(r"C:\Users\Admin\Desktop\fake-news-detector\True.csv")
# Add labels
fake["label"] = 0  # 0 = fake
true["label"] = 1  # 1 = real

# Combine
data = pd.concat([fake, true], axis=0)
data = data[["text", "label"]]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model saved! 🎉")
