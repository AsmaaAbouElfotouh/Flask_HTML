def train_preprocessing_1 (classifier) :
    import pandas as pd 
    import numpy  as np
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from classifier import KNN_classifier ,logistic_regression_classifier, svm_classifier,naive_bayes
    dataset = pd.read_csv ("dataset2_eng.Sayed.csv")

    # feature engineering
    features = ["gender","SeniorCitizen","Dependents",'tenure','Contract','PaymentMethod','MonthlyCharges','TotalCharges','Churn']
    dataset =dataset[features]
  

    # label encoding
    from sklearn.preprocessing import LabelEncoder  
    le =LabelEncoder()
    dataset["Churn"]= le.fit_transform(dataset["Churn"])


     # standarization
    dataset["tenure"] = (dataset["tenure"] - dataset["tenure"].mean()) / dataset["tenure"].std()
    dataset["MonthlyCharges"] = (dataset["MonthlyCharges"] - dataset["MonthlyCharges"].mean()) / dataset["MonthlyCharges"].std()
    dataset["TotalCharges"] = (dataset["TotalCharges"] - dataset["TotalCharges"].mean()) / dataset["TotalCharges"].std()

    dataset = train_preprocessing_2 (dataset)

     #test train split
    X = dataset.iloc[:,:-1 ].values
    y = dataset.iloc[:, -1].values
    #print ( X.shape)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    if classifier == "KNN":
        classifier_result = KNN_classifier(X_train, X_test, y_train, y_test )
    
    elif classifier == "Logistic regression":

        classifier_result = logistic_regression_classifier(X_train, X_test, y_train, y_test )

    elif  classifier == "SVM":
        classifier_result = svm_classifier(X_train, X_test, y_train, y_test )

    elif  classifier == "naive bayes":
        classifier_result = naive_bayes(X_train, X_test, y_train, y_test )

    

    return classifier_result


def train_preprocessing_2 (dataset) :

    from sklearn import preprocessing

    # missing values treatment
    dataset["SeniorCitizen"].fillna(dataset["SeniorCitizen"].mode()[0],inplace = True)
    dataset["tenure"].fillna(dataset["tenure"].mode()[0],inplace = True)

    # label encoding
    from sklearn.preprocessing import LabelEncoder  
    le =LabelEncoder()
    dataset["gender"]= le.fit_transform(dataset["gender"])
    dataset["Dependents"]= le.fit_transform(dataset["Dependents"])
    dataset["Contract"]= le.fit_transform(dataset["Contract"])
    dataset["PaymentMethod"]= le.fit_transform(dataset["PaymentMethod"])
  
    return dataset

