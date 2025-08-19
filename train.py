import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset (update path as needed)
df = pd.read_csv(r"C:\Users\adity\Machine Learning Project\spam.csv", encoding="latin-1")

# Use Kaggle dataset columns: v1 = label, v2 = message
X = df['v2']
y = df['v1'].map({'ham': 0, 'spam': 1})  # convert labels to 0/1

# Vectorizer
tfidf = TfidfVectorizer(max_features=3000)
X_vectorized = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save vectorizer & model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Training complete. Files saved: vectorizer.pkl, model.pkl")
