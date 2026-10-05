import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import joblib 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

 

'''
Function: loads the breast cancer dataset, returns it as a pandas DataFrame
'''
def input_data():
    df = load_breast_cancer(as_frame=True).frame

    print(f"Dataset shape: {df.shape}")
    print(f"First 5 entries: {df.head()}")
    return df



'''
Function: preprocesses the data by splitting into training and testing sets, and standardizing the features
'''
def preprocessing(dataframe): 
    # Step 1: Split the data into features and target
    X = dataframe.drop('target', axis=1)
    y = dataframe['target']

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Standardize the features  -> recommended for logistic regression to improve convergence 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test



'''
Function: trains a logistic regression model on the training data and saves the model to a file
'''
def train_model(X_train, y_train):
    # Importing the model definition from sklearn
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)     # produces the logistic regression model for the data 

    joblib.dump(model, 'logistic_regression.pkl')  # Save the trained model to a file

    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")

    return model 



'''
Function: evaluates the trained model on the test data and prints the accuracy
'''
def evaluate_model(X_test, y_test):
    # Load the trained model from the file
    model = joblib.load('logistic_regression.pkl')

    # Evaluate the model on the test set
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}") 



def main(train, evaluate): 
    # Step 1: load the breast cancer data from scikit-learn 
    dataframe = input_data()

    # Step 2: preprocessing splits and standardises the data 
    x_train, x_test, y_train, y_test = preprocessing(dataframe)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Step 3: train the model 
    if train: 
        model = train_model(x_train, y_train)

    # Step 4: evaluate the model -> omit for inference benchmarking (limitation: small dataset size) 
    if evaluate:
        evaluate_model(x_test, y_test)

    return 0

    

if __name__ == "__main__":
    main(train=True, evaluate=True)



