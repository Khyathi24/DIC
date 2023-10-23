import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Near Earth Object Hazardous prediction App")
    st.sidebar.title("Near Earth Object Hazardous prediction App")
    
    # Splitting dataset into train and test datasets using sklearn model selection 
    @st.cache(persist=True)
    def split(df):
        y = df.hazardous
        x = df.drop(columns=["hazardous"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # Plots confusion matrix, ROC curve and Precision-Recall curve using sklearn metrics
    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()


    # Allows user to upload csv file and Reads dataset into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Reads the default dataset collected from sources mentioned in report
        df = pd.read_csv('/input_dataframe.csv')
        df = df.drop(columns=['Unnamed: 0'])
        

    x_train, x_test, y_train, y_test = split(df)
    
    # Dropdown that allows user to choose classifier
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("XGBoost","Random Forest"))

    # Displays model parameters upon classifier selection
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators  = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 10, 20, step=1, key='max_depth')
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            
            st.write("Accuracy ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred).round(3))
            st.write("Bar plot on predicted values :")

            df_check = pd.DataFrame(y_pred, columns = ['hazardous'])
            df1 = df_check['hazardous'].value_counts()
            st.caption("Here X-axes: 0 - non hazardous, 1 - hazardous")
            st.bar_chart(df1)
            plot_metrics(metrics)

    if classifier == 'XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators  = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 2, 20, step=1, key='max_depth')
        reg_lambda = st.sidebar.number_input("The L2 regularization ( λ )", 0.01, 10.0, step=0.01, key='reg_lambda')
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("")
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,reg_lambda=reg_lambda)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(3))
            st.write("Precision: ", precision_score(y_test, y_pred).round(3))
            st.write("Recall: ", recall_score(y_test, y_pred).round(3))
            df_check = pd.DataFrame(y_pred, columns = ['hazardous'])
            df1 = df_check['hazardous'].value_counts()
            st.write("Bar plot on predicted values :")
            st.caption("Here X-axes: 0 - non hazardous, 1 - hazardous")
            st.bar_chart(df1)
            plot_metrics(metrics)

    # Displays dataset upon selection
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("NEO Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
