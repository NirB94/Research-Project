import argparse
import os
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


'''
Running command should be:
python3 RandomForestModel.py
OR
python3 RandomForestModel.py -k k_value

For example:
python3 RandomForestModel.py -k 10
'''


def main(arguments):
    """This function accepts the CMD input and runs the module.
    If no input was inserted - the default value is 5."""
    k = arguments.k if arguments.k is not None else 5
    run_all(k)


def run_all(k):
    """This function runs all the functions in this module.
    The function gets the train and test frequencies table.
    The function then modifies the data. For further documentation please see relevant function.
    The function then trains and plots the results.
    Finally, the function tests and plots the results."""
    directory = os.path.join(os.getcwd(), "Frequencies")
    train_data = pd.read_csv(os.path.join(directory, 'Frequencies_of_All_Train_Samples.csv'), index_col=0)
    test_data = pd.read_csv(os.path.join(directory, 'Frequencies_of_All_Test_Samples.csv'), index_col=0)
    data_modification(train_data, test_data)
    model_training(train_data, k)
    model_testing(test_data, train_data)


def data_modification(train, test):
    """This function is used to unify features and remove samples in order to improve model results.
    The function receives both the train and the test sets.
    The function calls for mutual_manipulation for each data.
    The function then removes 4500 samples from Human Microbiome environment.
    The function calls for find_feats to find features that do not appear in the sets feature-intersection.
    The function removes these feature.
    The function modifies the datasets in-place and therefore has no return value."""
    mutual_manipulation(train)
    mutual_manipulation(test)
    train.drop(sample(train[train['Environments'] == 'Human Microbiome'].index.to_list(), 4500), inplace=True)
    remove_from_test = find_feats(test, train)
    remove_from_train = find_feats(train, test)
    test.drop(remove_from_test, axis=1, inplace=True)
    train.drop(remove_from_train, axis=1, inplace=True)  # Remove columns from train set


def mutual_manipulation(data):
    """This function is used to save code duplications.
    The function receives a dataset.
    The function removes all samples that were not classified into an environment.
    The function also removes the GroundWater environment, as it has a low number of samples for both sets.
    The function fills any NaN value with 0.
    The function modifies the datasets in-place and therefore has no return value."""
    data.dropna(subset=['Environments'], inplace=True)
    data.drop(data[data['Environments'] == 'GroundWater'].index, inplace=True)
    data.fillna(0, inplace=True)


def find_feats(data1, data2):
    """This function is used to find features that are in the symmetric difference of the train ant test sets.
    The function receives two datasets.
    The function finds these said features and appends them to a list.
    The function returns that list."""
    feats_to_remove = []
    for feat in data1.columns[1:]:
        if feat not in data2.columns:
            feats_to_remove.append(feat)
    return feats_to_remove


def model_training(data, k):
    """This is a k-Fold Cross Validation (CV) Training of a Random Forest model on the training set.
    The function receives the training dataset, and k - an int indicating how many folds in the CV. Default is 5.
    While training, the function performs feature selection to improve model performances.
    The function then plots a feature importance chart.
    Finally, the function plots the model performances in forms of PRC and ROC graphs with std.
    The function has no return value."""
    path = os.path.join(os.getcwd(), 'Model_Results')

    # Encoding the 'Environments' column as it's the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Environments'])
    class_names = label_encoder.classes_

    # Dropping non-feature columns
    X = data.drop(['Environments'], axis=1)

    # Initialize Stratified K-Fold
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Variables to store metrics for each fold and class
    tprs_class = {i: [] for i in range(len(class_names))}
    precisions_class = {i: [] for i in range(len(class_names))}
    mean_fpr_class = np.linspace(0, 1, 100)

    # Iterate over each fold
    for train, test in cv.split(X, y):

        # Initialize Random Forest Classifier
        classifier = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

        # Feature Selection using SelectFromModel
        feature_selector = SelectFromModel(classifier)

        # Fit feature selector to the training data and transform datasets
        X_train_selected = feature_selector.fit_transform(X.iloc[train], y[train])
        X_test_selected = feature_selector.transform(X.iloc[test])

        # Train and predict with the selected features
        probas_ = classifier.fit(X_train_selected, y[train]).predict_proba(X_test_selected)

        # Compute ROC curve and ROC area for each class
        for i in range(probas_.shape[1]):
            fpr, tpr, _ = roc_curve(y[test], probas_[:, i], pos_label=i)
            interp_tpr = np.interp(mean_fpr_class, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_class[i].append(interp_tpr)

            precision, recall, _ = precision_recall_curve(y[test], probas_[:, i], pos_label=i)
            interp_precision = np.interp(mean_fpr_class, recall[::-1], precision[::-1])
            precisions_class[i].append(interp_precision)

    feature_importances = classifier.feature_importances_
    feature_names = X.columns
    plot_feature_importances(feature_importances, feature_names, path, top_n=20)

    # Calculate mean and std of AUC for ROC and PRC
    mean_roc_auc = {}
    std_roc_auc = {}
    mean_prc_auc = {}
    std_prc_auc = {}
    for i in range(probas_.shape[1]):
        mean_roc_auc[i] = np.mean([auc(mean_fpr_class, tpr) for tpr in tprs_class[i]])
        std_roc_auc[i] = np.std([auc(mean_fpr_class, tpr) for tpr in tprs_class[i]])
        mean_prc_auc[i] = np.mean([auc(mean_fpr_class, precision) for precision in precisions_class[i]])
        std_prc_auc[i] = np.std([auc(mean_fpr_class, precision) for precision in precisions_class[i]])

    # Plotting ROC Curves for each class with AUC and STD in the legend
    plt.figure(figsize=(8, 8))
    for i in range(probas_.shape[1]):
        mean_tpr = np.mean(tprs_class[i], axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs_class[i], axis=0)
        plt.plot(mean_fpr_class, mean_tpr,
                 label=f'{class_names[i]} (AUC = {mean_roc_auc[i]:.2f} ± {std_roc_auc[i]:.2f})')
        plt.fill_between(mean_fpr_class, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ttl = f'{k}-Fold Cross Validation Training data ROC'
    plt.title(ttl)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')

    # Plotting Precision-Recall Curves for each class with AUC and STD in the legend
    plt.figure(figsize=(8, 8))
    for i in range(probas_.shape[1]):
        mean_precision = np.mean(precisions_class[i], axis=0)
        std_precision = np.std(precisions_class[i], axis=0)
        plt.plot(mean_fpr_class, mean_precision,
                 label=f'{class_names[i]} (AUC = {mean_prc_auc[i]:.2f} ± {std_prc_auc[i]:.2f})')
        plt.fill_between(mean_fpr_class, mean_precision - std_precision, mean_precision + std_precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ttl = f'{k}-Fold Cross Validation Training data PRC'
    plt.title(ttl)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')


def plot_feature_importances(importances, feature_names, path, top_n=10):
    """This function is used to plot the feature importance chart. The function receives the importances,
    the features names, a destination path to save the plot, and a number of features to plot, defaulted to 10.
    The function plots and saves the chart in the destination path."""
    # Sort the feature importances in descending order and select the top n
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]

    # Prepare labels and their corresponding importances
    labels = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_importances, align='center')
    plt.yticks(range(top_n), labels)
    plt.ylabel('Features')
    plt.xlabel('Importance')
    ttl = "Top Feature Importances"
    plt.title(ttl)
    plt.gca().invert_yaxis()  # To display the highest importance on top
    plt.tight_layout()
    try:
        os.mkdir(path)
    except OSError:
        pass
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')


def model_testing(test, train):
    """This function is used to test the Random Forest model.
    The function receives both datasets.
    The function fits the model on the train set.
    The function then predicts probabilities of classifiers.
    Finally, the function evaluates performances, and plots them in the form of PRC and ROC graphs.
    The function has no return value."""
    path = os.path.join(os.getcwd(), 'Model_Results')

    # Encode the target variable for both datasets
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train['Environments'])
    y_test = label_encoder.transform(test['Environments'])

    # Prepare the feature matrices
    X_train = train.drop(['Environments'], axis=1)
    X_test = test.drop(['Environments'], axis=1)

    # Initialize and fit the Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    classifier.fit(X_train, y_train)

    # Predict probabilities on the test set
    probas_ = classifier.predict_proba(X_test)

    # Variables to store metrics for ROC and PRC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    prc_auc = dict()

    # Calculate ROC and PRC for each class
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test, probas_[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(y_test, probas_[:, i], pos_label=i)
        prc_auc[i] = auc(recall[i], precision[i])

    # Plotting ROC Curves for each class
    plt.figure(figsize=(8, 8))
    for i in range(len(label_encoder.classes_)):
        plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ttl = 'Testing data ROC'
    plt.title(ttl)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')

    # Plotting Precision-Recall Curves for each class
    plt.figure(figsize=(8, 8))
    for i in range(len(label_encoder.classes_)):
        plt.plot(recall[i], precision[i], label=f'Class {label_encoder.classes_[i]} (AUC = {prc_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ttl = 'Testing data PRC'
    plt.title(ttl)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', '-k', type=int, help='Insert number of desired folds for the Cross Validation training. '
                                              'Default is 5', required=False)
    args = parser.parse_args()
    main(args)

