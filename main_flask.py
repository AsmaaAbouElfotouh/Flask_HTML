from __future__ import with_statement
from flask import Flask,render_template, request
from classifier import KNN_classifier ,logistic_regression_classifier ,confusion_mat
from train_preprocessing import train_preprocessing_1 ,train_preprocessing_2
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def input():
    return render_template ("html_flask.html")



@app.route("/processing", methods=['POST', 'GET'])
def processing ():
    

    if request.method == 'POST' :

        gender =request.form ["gender"]
        SeniorCitizen =request.form ["SeniorCitizen"]
        Dependents =request.form ["Dependents"]
        tenure =request.form ["tenure"]
        Contract =request.form ["Contract"]
        PaymentMethod =request.form ["PaymentMethod"]
        MonthlyCharges =request.form ["MonthlyCharges"]
        TotalCharges =request.form ["TotalCharges"]
        
        data = np.array([gender,SeniorCitizen,Dependents,tenure,Contract,PaymentMethod,MonthlyCharges,TotalCharges])
        columns = ["gender","SeniorCitizen","Dependents","tenure","Contract","PaymentMethod","MonthlyCharges","TotalCharges"]
        dataset = pd.DataFrame([data],columns=columns)
        dataset = train_preprocessing_2(dataset)
        classifier = request.form ["classifier"]
        print (classifier)
        if classifier == "KNN" :
            classifier_result =train_preprocessing_1 (classifier)
            y_pred =  confusion_mat (classifier_result,dataset)
            

        elif  classifier == "Logistic regression" :
            classifier_result =train_preprocessing_1 (classifier)
            y_pred =  confusion_mat (classifier_result,dataset)
        
        elif  classifier == "SVM" :
            classifier_result =train_preprocessing_1 (classifier)
            y_pred =  confusion_mat (classifier_result,dataset)

        elif  classifier == "naive bayes" :
            classifier_result =train_preprocessing_1 (classifier)
            y_pred =  confusion_mat (classifier_result,dataset)

    

    prediction =y_pred
    return render_template ("results.html",prediction=y_pred)
    
  

if __name__ == "__main__":
  app.run(debug=True)