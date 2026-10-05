import joblib
import numpy as np

def inference_single(years_of_experience): 
    # Step 1: load the model 
    loaded_model = joblib.load('linear_regression.pkl')

    # Step 2: make a single prediction
    predicted_salary = loaded_model.predict(np.array([[years_of_experience]]).reshape(-1, 1))
    return predicted_salary

