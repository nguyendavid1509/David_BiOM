import numpy as np
import joblib

def test_model(X_test): 
    # Step 1: load the model 
    loaded_model = joblib.load('linear_regression.pkl')

    # Step 2: make predictions
    y_pred = loaded_model.predict(X_test)
    return y_pred

def test_model_single(years_of_experience): 
    # Step 1: load the model 
    loaded_model = joblib.load('linear_regression.pkl')

    # Step 2: make a single prediction
    predicted_salary = loaded_model.predict(np.array([[years_of_experience]]))
    return predicted_salary

