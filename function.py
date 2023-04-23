import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where
from imblearn.over_sampling import ADASYN

import warnings
warnings.filterwarnings( action= 'ignore')

sns.set_theme()


# metric 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, recall_score, make_scorer

from imblearn.pipeline import make_pipeline

# tuning parameter
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score


# function to get statistic data as a feature choosing window size for 1 minute with sliding window 30 second1 minute
def stat_feature(df, min = 1):
    """
    Function to calculate Statistic feature windiw size set to 1 minute with sliding window 1 minute
    No overlap
    
    Input: Raw signal Dataframe
    Output: Dataframe for Stat feature for each window
    """
    y = df['label']
    df = df.drop(['label'], axis = 1)
    
    w = round(min * 60 * 32) # because now sampling rate is 32 Hz
    feature = pd.DataFrame()
    for i in range(0, len(df)-w, w): #Slide window equal to windowsize
        # Calculate statistic feature
        mean_f = df[i: i+w].mean()
        std_f = df[i: i+w].std()
        min_f = df[i: i+w].min()
        max_f = df[i: i+w].max()
        skw_f = df[i: i+w].skew(axis = 0)
        kur_f = df[i: i+w].kurtosis()
        med_f = df[i: i+w].median()
        
        # make new df with statistic feature
        data = {'X_mean': [mean_f['X']], 'Y_mean': [mean_f['Y']], 
                'Z_mean': [mean_f['Z']],'EDA_mean': [mean_f['EDA']], 
                'HR_mean': [mean_f['HR']],'ACC_mean': [mean_f['ACC']],
               'TEMP_mean': [mean_f['TEMP']], 'IBI_mean': [mean_f['IBI']],
               'X_std': [std_f['X']], 'Y_std': [std_f['Y']], 
                'Z_std': [std_f['Z']],'EDA_std': [std_f['EDA']], 
                'HR_std': [std_f['HR']],'ACC_std': [std_f['ACC']],
               'TEMP_std': [std_f['TEMP']], 'IBI_std': [std_f['IBI']],
               'X_min': [min_f['X']], 'Y_min': [min_f['Y']], 
                'Z_min': [min_f['Z']],'EDA_min': [min_f['EDA']], 
                'HR_min': [min_f['HR']],'ACC_min': [min_f['ACC']],
               'TEMP_min': [min_f['TEMP']], 'IBI_min': [min_f['IBI']],
               'X_max': [max_f['X']], 'Y_max': [max_f['Y']], 
                'Z_max': [max_f['Z']],'EDA_max': [max_f['EDA']], 
                'HR_max': [max_f['HR']],'ACC_max': [max_f['ACC']],
               'TEMP_max': [max_f['TEMP']], 'IBI_max': [max_f['IBI']],
               'X_min': [min_f['X']], 'Y_min': [min_f['Y']], 
                'Z_min': [min_f['Z']],'EDA_min': [min_f['EDA']], 
                'HR_min': [min_f['HR']],'ACC_min': [min_f['ACC']],
               'TEMP_min': [min_f['TEMP']], 'IBI_min': [min_f['IBI']],
               'X_skew': [skw_f['X']], 'Y_skew': [skw_f['Y']], 
                'Z_skew': [skw_f['Z']],'EDA_skew': [skw_f['EDA']], 
                'HR_skew': [skw_f['HR']],'ACC_skew': [skw_f['ACC']],
               'TEMP_skew': [skw_f['TEMP']], 'IBI_skew': [skw_f['IBI']],
               'X_kur': [kur_f['X']], 'Y_kur': [kur_f['Y']], 
                'Z_kur': [kur_f['Z']],'EDA_kur': [kur_f['EDA']], 
                'HR_kur': [kur_f['HR']],'ACC_kur': [kur_f['ACC']],
               'TEMP_kur': [kur_f['TEMP']], 'IBI_kur': [kur_f['IBI']],
               'X_med': [med_f['X']], 'Y_med': [med_f['Y']], 
                'Z_med': [med_f['Z']],'EDA_med': [med_f['EDA']], 
                'HR_med': [med_f['HR']],'ACC_med': [med_f['ACC']],
               'TEMP_med': [med_f['TEMP']], 'IBI_med': [med_f['IBI']],}
        
        new_df = pd.DataFrame(data)
        
        feature = pd.concat([feature, new_df])
    feature = feature.reset_index(drop=True)
    y = y[:len(feature)].reset_index(drop=True)
    return pd.concat([feature, y], axis = 1)

def feature_extraction(df) :
    """
    This function will extract feature from raw signal into mean, std, median, kurtosis, min, max, and skewness
     for each signal with window size 30 second non-overlaping 
    """
    
    # extract the feature on each group of class
    df0 = stat_feature(df[df['label'] == 0])
    df1 = stat_feature(df[df['label'] == 1])
    df2 = stat_feature(df[df['label'] == 2])

    # concat them all together
    fn_df = pd.concat([df0, df1, df2])
    fn_df = fn_df.reset_index(drop=True)
    return fn_df

# function to evaluate model by crossvalidation 5 Folds
def Model_train(X_train, y_train, isSMOTE, model):
    """
    Train model and Evaluate using 5 Folds crossvalidation

    Input
        X_train: numpy array 
        y_train: numpy array (binary vector to represent each class)
        isSMOTE: If apply SMOTE pick True, if not pick False
        model: Machine learning model

    Output
        clf: Trained model
        scalar: use for normalize input
        df_metric: mean of k-Folds cross validation accuracy, precision, recall, f1-score
    """
    # define index of k-folds
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    # initial list to store score values
    df = []
    accuracy_l = []
    precision_l = []
    recall_l = []
    f1_l = []

    # loop through each fold
    for train_idx, val_idx in sss.split(X_train, y_train):
        # train-validation split
        X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        # apply SMOTE to training set if function request
        if isSMOTE:
            smote = SMOTE(random_state=42)
            X_train_sub, y_train_sub = smote.fit_resample(X_train_sub, y_train_sub)
        # normalize X
        scaler = preprocessing.StandardScaler()
        X_train_sub = scaler.fit_transform(X_train_sub)
        X_val = scaler.transform(X_val)
        # define ML model
        clf = model[1]
        # convert binary vector back if using SVC or XGB model cuz they not request binary form 
        if model[0] == 'SVC' or model[0] == 'XGB':
            y_train_sub = np.argmax(y_train_sub, axis=1)
            y_val = np.argmax(y_val, axis=1)
        # train model
        clf.fit(X_train_sub, y_train_sub)
        # predict
        y_pred = clf.predict(X_val)

        # append score to list
        accuracy_l.append(accuracy_score(y_val, y_pred))
        precision_l.append(precision_score(y_val, y_pred, average='macro'))
        recall_l.append(recall_score(y_val, y_pred, average='macro'))
        f1_l.append(f1_score(y_val, y_pred, average='macro'))
    
    # append to df list
    df.append({
        'model': model[0],
        'accuracy': np.mean(accuracy_l),
        'precision': np.mean(precision_l) ,
        'recall': np.mean(recall_l),
        'f1': np.mean(f1_l),
    })
    # convert to dataframe
    df_metric = pd.DataFrame(df,columns=['model','accuracy','precision','recall','f1'])
    # return trained model, scaler, and metric score 
    return clf, scaler, df_metric

# function to evaluate model with test set
def eval_testset(name, model, scaler, X_test, y_test) -> None:
    """
    Show result in form of classification report and Confusion Matrix

    Input
        name: Name of model
        model: ML model
        scaler: get from model_train function
        X_test: numpy array
        y_test: numpy array (binary vector to represent each class)

    """
    # normalize X 
    X_test = scaler.transform(X_test)
    # predict
    y_pred = model.predict(X_test)
    # convert binary vector back
    try:
        y_pred = y_pred.argmax(axis=1)
    except: pass
    try:
        y_test = np.argmax(y_test, axis=1).reshape((-1, 1))
    except: pass
    
    print(f"================== {name} ==================")
    print(classification_report(y_test, y_pred))
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create heatmap with seaborn
    sns.heatmap(cm, annot=True, cmap="Oranges", fmt = 'g')


    # Set axis labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Show the plot
    plt.show()

# Tuning parameter function
def SearchCV(X_train, y_train, name, mp):
    """
    Funtion to find the best parameter of each model
    scoring = None

    Input:
        X_train: numpy array
        y_train: numpy array (binary vector to represent each class)
        name: name of model
        mp: dictionary of model and its parameters
    Output:
        list of best parameter
    """
    print('Scoring = None')
    best_params = []
    
    # Define the classifier
    clf = mp['model']

    # Define the parameter grid for hyperparameter tuning
    param_grid = mp['params']

    # Define the SMOTE and StandardScaler instances
    smote = SMOTE()
    scaler = preprocessing.StandardScaler()

    if name == 'SVC' or name == 'XGB':
        y_train = np.argmax(y_train, axis=1)
        
    # Apply SMOTE to the training data
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Scale the training and test data
    X_train_scaled = scaler.fit_transform(X_train_smote)

    # Perform random search CV on the training data
    # to tune model parameter base on f1 score
    #scorer = make_scorer(f1_score, average='macro')

    random_search_clf = RandomizedSearchCV(clf, param_grid, cv=5, n_iter = 10, n_jobs=4)
    random_search_clf.fit(X_train_scaled, y_train_smote)

    best_params.append({
        'best_score': random_search_clf.best_score_,
        'best_params': random_search_clf.best_params_,
        'best_estimator':random_search_clf.best_estimator_
    })

    return best_params

def model_performance_bar(df)->None:
    """
    Plot model performance in bar chart
    """
    # Define data
    models = df['model']
    accuracy = df['accuracy']
    precision = df['precision']
    recall = df['recall']
    f1 = df['f1']

    # Set up bar chart
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize = (15,8))
    rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
    rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
    rects4 = ax.bar(x + 1.5*width, f1, width, label='F1')

    # Add labels and title
    ax.set_ylabel('Scores')
    ax.set_title('Model Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Show plot

    plt.show()
