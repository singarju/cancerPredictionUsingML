# Cancer Prediction Using Machine Learning

## Overview

This project aims to predict cancerous tumors using machine learning (ML) techniques, particularly **Logistic Regression**, to help in cancer diagnosis and prognosis. It focuses on predicting whether a patient has cancer or not based on diagnostic measurements from a dataset. The project includes data processing, training a machine learning model, and evaluating its performance.

## Project Objectives

- **Diagnostically predict** whether or not a patient has cancer based on certain diagnostic measurements.
- **Study cancer symptoms** using machine learning.
- **Extract important variables** from the dataset for cancer prediction.
- **Visualize the data** to understand the relationships between variables and improve prediction results.

## Problem Statement

Every year, millions of people are diagnosed with cancer, and pathologists' success rates for diagnosis are high (96-98%). However, the accuracy of prognoses (predicting the development of cancer after diagnosis) is only about 60%. This project aims to bridge that gap using machine learning techniques to improve cancer prognosis accuracy.

## Goal

The goal of this project is to develop a **Machine Learning model** that can help medical professionals predict the progression of cancer using logistic regression and various other techniques. This model could assist in better treatment planning and clinical decision-making.

## Approach

### 1. Data Gathering

The dataset used in this project comes from various pathology labs and testing clinics. The data includes diagnostic measurements that are used to train the model.

### 2. Data Preprocessing

Data cleaning is performed to remove null values, redundant data, and misleading data to create a clean dataset for the machine learning model.

### 3. Model Creation

The **Logistic Regression** algorithm is used to build the predictive model. This model predicts the likelihood of a patient having cancer based on the features in the dataset.

### 4. Model Training

The dataset is split into a training set and a testing set, with 75% used for training the model and 25% used for testing its accuracy.

### 5. Model Evaluation

The model’s performance is evaluated using accuracy, confusion matrix, and classification report metrics.

## Methodology

### Libraries Used

- **pandas**: Used for data manipulation and analysis.
- **seaborn**: Used for data visualization.
- **matplotlib**: Used for creating static, animated, and interactive visualizations.
- **numpy**: Used for numerical computations.
- **scikit-learn**: Used for implementing machine learning algorithms, including logistic regression and model evaluation.

### Logistic Regression

Logistic regression is used to model the probability of a certain event occurring, like whether a patient has cancer or not. The model uses a logistic function to map input features to a probability.

## Steps for Model Building

1. **Data Preprocessing**:
   - Handle missing values, remove duplicates, and clean data.
   
2. **Train-Test Split**:
   - Split the data into a training set (75%) and testing set (25%).

3. **Model Training**:
   - Fit the logistic regression model to the training data.

4. **Model Evaluation**:
   - Evaluate the model on the test data using accuracy, confusion matrix, and classification report.

## Future Scope

With advancements in diagnostic tools, such as genomic and proteomic assays, more complex datasets are available for predicting cancer progression. Machine learning can be used for more personalized and predictive medicine, leading to better treatment decisions and improved patient outcomes.

## Limitations

- The model's performance depends on the quality of data.
- Requires large, structured datasets for effective training.
- It is essential to perform offline/batch training for better results.
- The model may not handle all high-level symbolic reasoning or planning.

## Requirements

Before running the project, make sure you have the following libraries installed:

- pandas
- seaborn
- matplotlib
- numpy
- scikit-learn

You can install the necessary libraries using pip:

```bash
pip install pandas seaborn matplotlib numpy scikit-learn
```

## How to Run

1. Clone the repository or download the files.
2. Place the `cancer_data.csv` in the same directory as the Jupyter notebook.
3. Run the Jupyter notebook cell by cell to load the dataset, train the logistic regression model, and evaluate its performance.

## Results

The model will output:
- **Accuracy** of the model on the test set.
- **Confusion Matrix** showing the number of correct and incorrect predictions.
- **Classification Report** with precision, recall, and F1-score for each class.

## Bibliography

- [Kaggle - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Machine Learning and Cancer Prediction](https://towardsdatascience.com/machine-learning-is-the-future-of-cancer-prediction-e4d28e7e6dfa#:~:text=Machine%20Learning%20is%20a%2Fbranch,predicting%20the%20development%20of%20cancer.)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2001037014000464)
