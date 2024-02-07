import data_loader
import knn_model
import plotting
import numpy as np

def main():

    file_path = 'TMM4128---Project/Datasets/Iris/iris.data'

    X, y_encoded, class_names = data_loader.data_loader(file_path)

    X_train, X_test, y_train, y_test = data_loader.split_data(X, y_encoded)

    neighbors_range = range(1, 11)
    best_acc = 0
    best_n = 0

    #Find best k
    for i in neighbors_range:
        model = knn_model.create_knn_model(i)
        knn_model.train_model(model, X_train, y_train)
        predictions = knn_model.make_predictions(model, X_test)

        acc,_,_ = knn_model.evalution(y_test, predictions)

        if(acc > best_acc):
            best_acc = acc
            best_n = i
    
    model = knn_model.create_knn_model(best_n)
    knn_model.train_model(model, X_train, y_train)

    print(f"Best accuracy was {best_acc} achieved with n={best_n} ")

    plotting.plot_decision_boundary(model, X_test, y_test, 8)


if __name__ == "__main__":
    main()
