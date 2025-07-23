import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset and inspect columns
file_path = 'C:/Users/skarm/Downloads/spam.csv'
df_raw = pd.read_csv(file_path, encoding='latin-1')
print("🧾 CSV Columns:", df_raw.columns)

# If correct columns are v1 and v2
if 'v1' in df_raw.columns and 'v2' in df_raw.columns:
    data = df_raw[['v1', 'v2']]
    data.columns = ['label', 'text']
else:
    # Try to find first two valid string columns
    data = df_raw.iloc[:, :2]
    data.columns = ['label', 'text']

# Drop rows with missing values
data.dropna(inplace=True)

# Convert labels to numbers
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label_num'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict new email
def predict_email(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example
example_email = "Congratulations! You've won a $1000 gift card. Claim now."
print("\n📧 Prediction for Example Email:", predict_email(example_email))
