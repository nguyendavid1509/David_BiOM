from sklearn.model_selection import train_test_split

def preprocess_data(dataset): 
    # Step 1: defining the features and target variable 
    x_data = dataset.iloc[:, 0].values
    y_data = dataset.iloc[:, 1].values

    # Step 2: splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(x_data.reshape(-1, 1), y_data, test_size=0.2, random_state=0)

    return x_data, y_data, X_train, X_test, y_train, y_test
