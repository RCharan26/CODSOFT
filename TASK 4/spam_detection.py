import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("Loading and cleaning dataset..")
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df.iloc[:, :2]
df.columns = ['label', 'message']

df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully.")

sample_sms = [
    "Congratulations! You’ve won a free iPhone. Click the link to claim.",
    "Hi, are we still meeting for lunch tomorrow?"
]
sample_tfidf = vectorizer.transform(sample_sms)
sample_pred = model.predict(sample_tfidf)

print("\nSample Predictions:")
for sms, label in zip(sample_sms, sample_pred):
    print(f"Message: {sms}\nPrediction: {'SPAM' if label == 1 else 'HAM'}\n")
