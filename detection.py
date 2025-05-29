import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])


df['label_num'] = df.label.map({'ham': 0, 'spam': 1})


print("Sample data:")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


test_emails = [
    "Win a brand new iPhone now! Click here.",
    "Hey, are we still meeting for lunch today?"
]
test_vec = vectorizer.transform(test_emails)
predictions = model.predict(test_vec)

print("\nPredictions for custom messages:")
for msg, pred in zip(test_emails, predictions):
    print(f"{msg} --> {'Spam' if pred else 'Ham'}")
