import utilities
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



X_train, X_test, y_train, y_test = utilities.get_data()

""" scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train) """

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

parameters = {
    'knn__n_neighbors' : [8]
}

grid_search = GridSearchCV(pipeline, parameters, cv= 5, verbose= 1, n_jobs= -1)
grid_search.fit(X_train, y_train)
knn = grid_search.best_estimator_


train_pred = knn.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Train accuracy: {train_acc:.5f}")

test_pred = knn.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Train accuracy: {test_acc:.5f}")
