# Complete corrected Streamlit ML app (shortened for chat file)
import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import *

st.set_page_config(page_title="ML App")

st.title("Machine Learning Application")

src=st.sidebar.selectbox("Data Source",["Example","Upload"])
data=None
if src=="Example":
    name=st.sidebar.selectbox("Dataset",["iris","tips","titanic"])
    data=sns.load_dataset(name)
else:
    f=st.sidebar.file_uploader("Upload",type=["csv","xlsx","tsv"])
    if f:
        if f.name.endswith(".csv"):
            data=pd.read_csv(f)
        elif f.name.endswith(".xlsx"):
            data=pd.read_excel(f)
        else:
            data=pd.read_csv(f,sep="\t")

if data is not None:
    st.write(data.head())
    features=st.multiselect("Features",list(data.columns))
    target=st.selectbox("Target",list(data.columns))
    ptype=st.selectbox("Problem",["Classification","Regression"])
    if features and target:
        X=data[features]
        y=data[target]
        num=X.select_dtypes(include=["number"]).columns
        cat=X.select_dtypes(exclude=["number"]).columns
        pre=ColumnTransformer([
            ("num",Pipeline([("imp",SimpleImputer(strategy="mean")),("sc",StandardScaler())]),num),
            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
        ])
        X=pre.fit_transform(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        if ptype=="Regression":
            mname=st.sidebar.selectbox("Model",["Linear Regression","Decision Tree","Random Forest","SVM"])
            model={"Linear Regression":LinearRegression(),
                   "Decision Tree":DecisionTreeRegressor(),
                   "Random Forest":RandomForestRegressor(),
                   "SVM":SVR()}[mname]
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            st.write("MAE",mean_absolute_error(y_test,pred))
            st.write("MSE",mean_squared_error(y_test,pred))
            st.write("R2",r2_score(y_test,pred))
        else:
            from sklearn.preprocessing import LabelEncoder
            if y.dtype=="object":
                y=LabelEncoder().fit_transform(y)
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            mname=st.sidebar.selectbox("Model",["Decision Tree","Random Forest","SVM"])
            model={"Decision Tree":DecisionTreeClassifier(),
                   "Random Forest":RandomForestClassifier(),
                   "SVM":SVC()}[mname]
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            st.write("Accuracy",accuracy_score(y_test,pred))
            st.write("Precision",precision_score(y_test,pred,average="weighted",zero_division=0))
            st.write("Recall",recall_score(y_test,pred,average="weighted",zero_division=0))
            st.write("F1",f1_score(y_test,pred,average="weighted",zero_division=0))
        if st.button("Save Model"):
            with open("model.pkl","wb") as f:
                pickle.dump(model,f)
            st.success("Model saved.")
