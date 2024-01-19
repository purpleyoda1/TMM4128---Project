import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_loader(filepath):
    #Load the data to a table
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(filepath, header=0, names=columns)

    #Split into input and result
    """ X = data.drop("species", axis=1) """
    X = data.drop(["petal_length", "petal_width", "species"], axis=1)
    y = data["species"]

    #Make result numerical data
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le.classes_

def split_data(X, y, test_size=0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test