
#X_train, X_test, y_train, y_test = train_preprocessing_2()
def svm_classifier (X_train, X_test, y_train, y_test ):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix , classification_report
    cm= confusion_matrix(y_test, y_pred)
    print (cm)
    print (classification_report(y_test, y_pred))

    return classifier

def KNN_classifier(X_train, X_test, y_train, y_test ):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
    classifier = classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix , classification_report
    cm= confusion_matrix(y_test, y_pred)
    print (cm)
    print (classification_report(y_test, y_pred))

    return classifier


def logistic_regression_classifier (X_train, X_test, y_train, y_test ):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier = classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix , classification_report
    cm= confusion_matrix(y_test, y_pred)
    print (cm)
    print (classification_report(y_test, y_pred))

    return classifier


def naive_bayes (X_train, X_test, y_train, y_test ):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix , classification_report
    cm= confusion_matrix(y_test, y_pred)
    print (cm)
    print (classification_report(y_test, y_pred))

    return classifier


def confusion_mat (classifier,X_test ):

    y_pred = classifier.predict(X_test)
    
    return y_pred

    



