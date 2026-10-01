from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train): 
    # Step 1: training the model
    model = LinearRegression()
    model.fit(X_train, y_train)     # produces the linear equation for the data 
    joblib.dump(model, 'linear_regression.pkl')  # Save the trained model to a file
    
    return model 

