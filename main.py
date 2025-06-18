import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from feature_extractor import extract_features

df = pd.read_csv("spambase.data", header=None)


print(df.head())




print("Shape of dataset:", df.shape)


print("Any null values?", df.isnull().values.any())


print(df.describe())



X = df.iloc[:, :-1]


y = df.iloc[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])




model = LogisticRegression(max_iter=3000)


model.fit(X_train, y_train)



y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


new_email = input("Enter your email.")
features = extract_features(new_email)
prediction = model.predict([features])[0]

print("SPAM" if prediction == 1 else "NOT SPAM")