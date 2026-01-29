import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk
import os

# Download stopwords
nltk.download('stopwords')


# Step 1: Load Dataset (TSV)

news_dataset = pd.read_csv(
    'data/train.tsv', 
    sep='\t', 
    engine='python', 
    on_bad_lines='skip', 
    encoding='utf-8'
)

# Strip any extra spaces in headers
news_dataset.columns = news_dataset.columns.str.strip()

# Optionally, force column names (if TSV headers are wrong)
expected_cols = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'speaker_job', 'state', 'party', 'barely_true', 'false_count',
    'pants_on_fire', 'half_true', 'mostly_true', 'context'
]

if len(news_dataset.columns) == len(expected_cols):
    news_dataset.columns = expected_cols

# Fill missing values
news_dataset = news_dataset.fillna('')

print("Dataset shape:", news_dataset.shape)
print("Columns:", news_dataset.columns)
print(news_dataset.head())

# Step 2: Map Labels (Optional Binary)

real_labels = ['true', 'mostly-true', 'half-true']
fake_labels = ['false', 'barely-true', 'pants-on-fire']


if 'label' not in news_dataset.columns:
    raise ValueError("No 'label' column found in dataset!")


# Map labels to 0 (Real) / 1 (Fake)
def map_labels(label):
    label = str(label).lower().strip()
    if label in real_labels:
        return 0
    else:
        return 1


news_dataset['label'] = news_dataset['label'].apply(map_labels)
print("Label distribution:\n", news_dataset['label'].value_counts())


# Step 3: Preprocess Text

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)


# Use 'statement' as main text
news_dataset['content'] = news_dataset['statement'].apply(preprocess_text)


# Step 4: Prepare Features & Labels

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)


# Step 5: Train Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Step 6: Save Model & Vectorizer

os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")


# Optional: Check Accuracy


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Test Accuracy:", accuracy_score(Y_test, test_pred))
