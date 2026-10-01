import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess_data
from train import train_model
from test import test_model
from predict import inference_single

def main(train, test, predict): 

    if train == True: 
        # Step 1: importing the dataset
        dataset = pd.read_csv('salary_data.csv')

        # Step 2: data preprocessing + splitting the dataset
        x_data, y_data, X_train, X_test, y_train, y_test = preprocess_data(dataset)

        # Step 3: training and saving the model 
        my_model = train_model(X_train, y_train)


    if test == True:

        if train == False: 
            # Re-importing the dataset and extracting X_test
            dataset = pd.read_csv('salary_data.csv')
            x_data, y_data, X_train, X_test, y_train, y_test = preprocess_data(dataset)

        # Step 4: evaluating the model with the test sub-set
        y_pred = test_model(X_test)  # Make predictions on the test sub-set after training
        
        # Step 5: visualizing the results
        plt.figure(figsize=(10, 6))
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.title('Fitting the model to the data')

        plt.scatter(x_data, y_data, color='red', label='Actual Data')       # plotting the actual data (measurements)
        plt.plot(X_test, y_pred, color='blue', label='Predicted Data')      # plotting the predicted data (model output) 
        plt.legend()
        plt.show()


    if predict == True: 
        # Step 4: load the saved model and make predictions  
        y_pred = inference_single([3, 5, 2, 6, 4])
        print(f"Predicted salary for 7 years of experience: ${y_pred}")
        print(type(y_pred))



if __name__ == "__main__":
    main(train=False, test=False, predict=True)
