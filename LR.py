# Importing Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Loading the Data
data = pd.read_csv("./cancer_data.csv")
data.head()

# Data Preparation
X = data[data.columns[2:9]]
y = data['diagnosis']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Building a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)

# Model Evaluation
model.score(X_test, y_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, model.predict(X_test)))
print(confusion_matrix(y_train, model.predict(X_train)))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
