from typing import List
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

#Read the aligned data 
aligned_data = pd.read_csv('cleaned_aligned_data.csv')

most_northely_latitude = 60.9

most_southerly_latitude = 49.9


# creates a new column which contains the region a particular instance belongs to using the latitude,
# given the num regions and the labels for the region
def divide_UK_stations_into_regions(num_regions:int, labels,df:pd.DataFrame):
    diff = (most_northely_latitude - most_southerly_latitude) / num_regions 
    diff_rounded = round(diff,1)
    
    latitude_band = np.arange(most_southerly_latitude, most_northely_latitude + diff_rounded, diff_rounded)
    print(list(latitude_band))
     
    if len(labels) != num_regions:
        raise ValueError(f"Number of labels must match the number of regions ({num_regions})")
    
    df['region'] = pd.cut(df['latitude'], bins=latitude_band, labels=labels)
       
    return df

# divides the data into test set and training set by using data for the lst 5 stations as 
# test set and the rest for training
def divide_station_data_into_two(dataframe:pd.DataFrame):
    
    # Extract the station names from these rows
    last_five_stations = np.unique(dataframe['station']).tolist()[-5:]

    # Select the rows corresponding to these stations
    last_five_data = dataframe[dataframe['station'].isin(last_five_stations)]

    # Create a new dataframe with the last five stations and their data
    last_five_df = pd.DataFrame(last_five_data)

    indices_to_drop = last_five_df.index

    # Drop the rows from the original dataframe
    updated_df = dataframe.drop(indices_to_drop)

    # print(len(np.unique(updated_df['station'])))
    # print(np.unique(updated_df['station']))
    
    # print(np.unique((last_five_df['station'])))
    # last_five_df.to_excel('last_five.xlsx')
    # updated_df.to_excel('updated.xlsx')
    
    return last_five_df, updated_df


def perform_K_fold(num_fold:int,training_df:pd.DataFrame,test_df:pd.DataFrame):

    cols = ['tmax(degC)', 'tmin(degC)', 'af(days)', 'rain(mm)', 'sun(hours)']
    X = training_df[cols].values
    y = training_df['region'].values
    
    X_test = test_df[cols].values
    y_test = test_df['region'].values 

    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_test = label_encoder.transform(y_test)
    

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_tst = scaler.transform(X_test)

    # Defining classifiers
    logr = LogisticRegression(solver='sag', multi_class='multinomial', random_state=0, max_iter=5000)
    svm = SVC(random_state=0)
    rf = RandomForestClassifier(random_state=0)

    # Creating k-fold cross-validation object
    k_fold = KFold(n_splits=num_fold,shuffle=True,random_state=0)

    # lists to store accuracy and f1 scores for training and validation sets for all classifiers
    train_acc, train_f1, val_acc, val_f1 = [], [], [], []

    # Training and evaluating each classifier using k-fold cross-validation
    for train_index, val_index in k_fold.split(X):
        # Splitting data into training and validation sets
        X_train, y_train = X[train_index], y[train_index] 
        X_val, y_val = X[val_index], y[val_index]

        # Training logistic regression classifier
        logr.fit(X_train, y_train)
        y_logr_pred_train = logr.predict(X_train)
        y_logr_pred_val = logr.predict(X_val)

        # Training SVM classifier
        svm.fit(X_train, y_train)
        y_svm_pred_train = svm.predict(X_train)
        y_svm_pred_val = svm.predict(X_val)

        # Training random forest classifier
        rf.fit(X_train, y_train)
        y_rf_pred_train = rf.predict(X_train)
        y_rf_pred_val = rf.predict(X_val)

        # Evaluating classifiers on training set
        train_acc.append([accuracy_score(y_train, y_logr_pred_train), 
                        accuracy_score(y_train, y_svm_pred_train),
                        accuracy_score(y_train, y_rf_pred_train)])
        
        train_f1.append([f1_score(y_train, y_logr_pred_train,average='weighted'), 
                        f1_score(y_train,y_svm_pred_train,average='weighted'),
                        f1_score(y_train, y_rf_pred_train,average='weighted')])

        # Evaluating classifiers on validation set
        val_acc.append([accuracy_score(y_val, y_logr_pred_val), 
                        accuracy_score(y_val, y_svm_pred_val),
                        accuracy_score(y_val,y_rf_pred_val)])
        
        val_f1.append([f1_score(y_val, y_logr_pred_val,average='weighted'), 
                    f1_score(y_val, y_svm_pred_val,average='weighted'),
                    f1_score(y_val, y_rf_pred_val,average='weighted')])

    # Calculating mean accuracy and f1 scores for training and validation sets for each classifier
    train_acc_mean = np.mean(train_acc, axis=0)
    train_f1_mean = np.mean(train_f1, axis=0)
    val_acc_mean = np.mean(val_acc, axis=0)
    val_f1_mean = np.mean(val_f1, axis=0)

    # LR = logistic regression, DT = Decision tree, RF=Random Forest
    print(f"Training mean accuracy: LR:{train_acc_mean[0]:.3f},SVM:{train_acc_mean[1]:.3f}, RF:{train_acc_mean[2]:.3f}")
    print(f"Training mean f1 score: LR:{train_f1_mean[0]:.3f},SVM:{train_f1_mean[1]:.3f}, RF:{train_f1_mean[2]:.3f}")
    print(f"Validation mean accuracy: LR:{val_acc_mean[0]:.3f},SVM:{val_acc_mean[1]:.3f}, RF:{val_acc_mean[2]:.3f}")
    print(f"Validation mean f1 score: LR:{val_f1_mean[0]:.3f},SVM:{val_f1_mean[1]:.3f}, RF:{val_f1_mean[2]:.3f}")
    
    
    print("predicting on test set")
    y_pred_logr = logr.predict(X_tst)
    accuracy_lr = accuracy_score(y_test, y_pred_logr)
    f1_lr= f1_score(y_test,  y_pred_logr, average='weighted')
    print(f"Test accuracy: LR:{accuracy_lr:.3f}")
    print(f"Test f1 score: LR:{f1_lr:.3f}")
    
    y_pred_svm = svm.predict(X_tst)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test,  y_pred_svm, average='weighted')
    print(f"Test accuracy: SVM:{accuracy_svm:.3f}")
    print(f"Test f1 score: SVM:{f1_svm:.3f}")
    
    y_pred_rf = rf.predict(X_tst)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test,  y_pred_rf, average='weighted')
    print(f"Test accuracy: RF:{accuracy_rf:.3f}")
    print(f"Test f1 score: RF:{f1_rf:.3f}")
    

labels = ['Southern','Central','Northern']
num_regions = 3

dataset = divide_UK_stations_into_regions(num_regions,labels,aligned_data)

last_five_stations, all_excluded_five = divide_station_data_into_two(dataset)

perform_K_fold(10, all_excluded_five, last_five_stations)





