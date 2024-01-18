from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def create_knn_model(n_neighbors = 3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def make_predictions(model, X_test):
    prediction = model.predict(X_test)
    return prediction

def evalution(y_test, prediction):
    acc = accuracy_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)
    class_report = classification_report(y_test, prediction)

    return acc, conf_matrix, class_report