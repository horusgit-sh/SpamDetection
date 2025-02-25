from TextPreparation import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]
model_filename = 'spam_detection_model.joblib'

df = textPreparing()
loaded_model = joblib.load(model_filename)

X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"].apply(lambda x: 1 if x == "spam" else 0), test_size=0.2, random_state=42)


pipeline = Pipeline([
    ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])


pipeline.fit(X_train, y_train)
processed_messages = [preprocess_message(msg) for msg in new_messages]


X_new = pipeline.named_steps["vectorizer"].transform(processed_messages)
predictions = loaded_model.predict(new_messages)
prediction_probabilities = pipeline.named_steps["classifier"].predict_proba(X_new)


for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1]  # Вероятность спама
    ham_probability = prediction_probabilities[i][0]  # Вероятность не-спама

    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Ham Probability: {ham_probability:.2f}")
    print("-" * 50)

model_filename = 'spam_detection_model.joblib'

print(f"Model saved to {model_filename}")