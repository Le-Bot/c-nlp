import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

# Read the data set
df = pd.read_csv('data/dataset.csv')

# Retried features and Labels column
x_read = list(df["feature"])
y_read = list(df["labels"])

# Train data except for last three
x_train = x_read[:-3]
y_train = y_read[:-3]

# Test data only last three
x_test = x_read[-3:]
y_test = y_read[-3:]

# Classifier pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier()),
])

# Test to run number of times
test_run = 100
result = []
for i in range(test_run):
    text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    result.append(np.mean(predicted == y_test))

# Average of all the result
print(np.mean(result))