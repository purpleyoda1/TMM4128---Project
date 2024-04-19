import utilities
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import joblib

def univariate_feature_selection(num_features = 57, enable_print= False):
    #Load data
    X_train, X_test, y_train, y_test = utilities.get_data(get_scaled= True)

    #Create the ranker, then transform the data based on the ranking
    best_features = SelectKBest(f_classif, k= num_features)
    best_features.fit(X_train, y_train)

    #Apply feature selection
    selected_features = best_features.get_support()
    X_train_reduced = best_features.transform(X_train)
    X_test_reduced = best_features.transform(X_test)
    X_train = pd.DataFrame(X_train_reduced, index=X_train.index, columns=X_train.columns[selected_features])
    X_test = pd.DataFrame(X_test_reduced, index=X_test.index, columns=X_test.columns[selected_features])

    #IMPORTANT: can use this best_features to alter the data later

    selected_features = best_features.get_support(indices= True)
    feature_scores = best_features.scores_[selected_features]

    if enable_print:
        print(f"Selected features:\n{selected_features}\n")
        print(f"Feauture score:\n{feature_scores}\n")
    
    return selected_features, feature_scores, X_train, X_test



def recursive_feature_elimnination(num_features = 1, enable_print= False, enable_save= False):
    #Load data
    X_train, X_test, y_train, y_test = utilities.get_data(get_scaled= True)

    #Initialize and fit selector
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features_to_select= num_features, step= 1)
    selector = selector.fit(X_train, y_train)

    #Apply selector
    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)
    selected_features = X_train.columns[selector.support_]
    X_train_reduced = pd.DataFrame(X_train_reduced, index=X_train.index, columns=selected_features)
    X_test_reduced = pd.DataFrame(X_test_reduced, index=X_test.index, columns=selected_features)
    
    #Get ranking and sort it
    feature_ranking = pd.Series(selector.ranking_, index= X_train.columns)
    feature_ranking = feature_ranking.sort_values()

    #Save to a file
    if enable_save:
        with open('Results/FeatureSelection/recursive_ranking', 'w') as file:
            for rank, feature in feature_ranking.items():
                file.write(f"{feature}: {rank}\n")

    if enable_print:
        print(f"Feature ranking:\n {feature_ranking}\n")
        print(f"Selected features:\n{selector.support_}\n")
        print(f"X_reduced:\n{X_train_reduced}\n")

    return feature_ranking, X_train_reduced, X_test_reduced



def plot_feature_ranking(method, filename):
    if method == "Univariate":
        selected_features, scores, X_train, X_test = univariate_feature_selection()
    elif method == "Recursive":
        scores, X_train, X_test = recursive_feature_elimnination()

    features = X_train.columns
    indices = np.argsort(scores)

    #plot feature distribution for best and worst feature
    plot_feature_distribution_by_genre(features[indices[0]], method + '_best')
    plot_feature_distribution_by_genre(features[indices[-1]], method + '_worst')


    #plot horizontal bar chart
    plt.figure(figsize=(10,8))
    plt.barh(range(len(indices)), scores[indices], color= 'r', align= 'center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel("Importance score/ranking")
    plt.title(f"{method} feature ranking")
    plt.savefig('Results/FeatureSelection/' + filename)
    plt.show()
    

def plot_feature_distribution_by_genre(feature, filename):
    #Load data
    X_train, _, y_train, _ = utilities.get_data(get_scaled= True)

    #Turn labels into string instead encoded numerical values
    y_train = utilities.decode_labels(y_train)

    #Combine
    data = pd.concat([X_train, y_train], axis= 1)

    #Plot figure showing distribution of a feature
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=feature, hue='label', multiple='stack', bins=30, kde=False)
    plt.title('Distribution of '+feature+' by Genre')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.savefig('Results/FeatureSelection/' + filename)
    plt.show()



if __name__ == '__main__':
    #plot_feature_ranking("Univariate", "univariate_allfeatures")
    #recursive_feature_elimnination(enable_print= True, enable_save= True)
    plot_feature_distribution_by_genre("perceptr_var", 'recursive_best')
    plot_feature_distribution_by_genre("mfcc15_var", 'recursive_worst')

