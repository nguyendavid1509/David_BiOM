import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess_data
from train import train_model
from test import test_model, test_model_single

def main(train=True, test=True): 
    # Step 1: importing the dataset
    dataset = pd.read_csv('salary_data.csv')

    # Step 2: data preprocessing + splitting the dataset
    x_data, y_data, X_train, X_test, y_train, y_test = preprocess_data(dataset)


    if train == True: 
        # Step 3: training and saving the model 
        train_model(X_train, y_train)
        y_pred = test_model(X_test)  # Make predictions on the test set after training

        # Step 5: visualizing the results
        plt.figure(figsize=(10, 6))
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.title('Fitting the model to the data')

        plt.scatter(x_data, y_data, color='red', label='Actual Data')       # plotting the actual data (measurements)
        plt.plot(X_test, y_pred, color='blue', label='Predicted Data')      # plotting the predicted data (model output) 
        plt.legend()
        plt.show()

    if test == True: 
        # Step 4: load the saved model and make predictions  
        y_pred = test_model_single(7)
        print(f"Predicted salary for 7 years of experience: ${y_pred[0]:.2f}")


if __name__ == "__main__":
    main(train=False, test=True)
