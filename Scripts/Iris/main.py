import data_loader
import knn_model

def main():

    file_path = "../Datasets/iris.data"

    X, y_encoded, class_names = data_loader.data_loader(file_path)

    X_train, X_test, y_train, y_test = data_loader.split_data(X, y_encoded)

    neighbors_range = range(1, 21)
    best_acc = 0
    best_n = 0

    for i in neighbors_range:
        model = knn_model.create_knn_model(i)
        knn_model.train_model(model, X_train, y_train)
        predictions = knn_model.make_predictions(model, X_test)

        acc,_,_ = knn_model.evalution(y_test, predictions)

        if(acc > best_acc):
            best_acc = acc
            best_n = i
    
    print(f"Best accuracy was {best_acc} achieved with n={best_n} ")


if __name__ == "__main__":
    main()
