import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/processed/grocery_categories.csv")

X = df["item_name"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

test_items = ["apple", "milk", "bagel", "olive oil", "coffee"]
predictions = model.predict(test_items)

print("\nSample Predictions:")
for item, category in zip(test_items, predictions):
    print(f"{item} -> {category}")

joblib.dump(model, "models/task1_grocery_classifier.pkl")
print("\nModel saved to models/task1_grocery_classifier.pkl")