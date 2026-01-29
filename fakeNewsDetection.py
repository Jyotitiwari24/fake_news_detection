import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# Printing stopwords in english
print(stopwords.words('english'))

# Loading the Dataset to a Pandas Dataframe
#news_dataset = pd.read_csv('/content/train.csv')
import pandas as pd

# For pandas >= 1.3.0, use 'on_bad_lines' instead
news_dataset = pd.read_csv('/content/train.csv',
                          engine='python',
                          on_bad_lines='skip',  # Replaces error_bad_lines=False
                          encoding='utf-8')

print(f"Successfully loaded dataset with {len(news_dataset)} rows")
print(f"Dataset shape: {news_dataset.shape}")

news_dataset.shape

# printing first five rows of  dataset
news_dataset.head()

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null value with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+ ' ' +news_dataset['title']

print(news_dataset['content'])

# seprating the data & label
X = news_dataset.drop(columns = 'label', axis = 1)
Y = news_dataset['label']

print(X)
print(Y)

print(X.shape)
print(Y.shape)

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ' , content)  # removing all things except alphabtes and words
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in  stemmed_content if  not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# Separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

Y.shape

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

X_train, X_test , Y_train , Y_test =  train_test_split(X,Y, test_size = 0.2 , stratify = Y , random_state = 2)

model = LogisticRegression()

model.fit(X_train , Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)

print('Accuracy Score of the Training Data :' , training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print('Accuracy Score of the Training Data :' , test_data_accuracy)

X_new = X_test[0]

prediction = model.predict(X_new)
print(prediction)

if(prediction[0] == 0):
  print('The News is Real :')
else:
  print('The News Is Fake')

print(Y_test[0])

X_new = X_test[5]

prediction = model.predict(X_new)
print(prediction)

if(prediction[0] == 0):
  print('The News is Real :')
else:
  print('The News Is Fake')

print(Y_test[5])


