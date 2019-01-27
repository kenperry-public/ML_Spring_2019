# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Common imports
import os

# Sklearn imports

TITANIC_PATH = os.path.join("./external/jack-dies", "data")

train_data = pd.read_csv( os.path.join(TITANIC_PATH, "train.csv") )
test_data  = pd.read_csv( os.path.join(TITANIC_PATH, "test.csv")  )

# Numeric pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

num_features = ["Age", "SibSp", "Parch", "Fare"]
num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(num_features)),
        ("imputer", SimpleImputer(strategy="median")),
    ])

# Categorical pipeline
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

class SexToInt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        sex = X["Sex"]
        X["Sex"] = 0
        X[ sex == "female" ] = 1
        
        return(X)

# NOTE:
# The cat pipeline is run AFTER the numeric pipeline
# - so it needs to select not only the categorical features, but the numeric ones as well so that they will be included ??
# DOESN'T SEEM RIGHT -- may have duplicated the columns

cat_features =  ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch" ]

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(cat_features)),
        ("imputer", MostFrequentImputer()),
        ("sex_encoder", SexToInt() ),
    ])

# Union the numberic and categorical pipelines
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# Run the pipelinem return an ndarray
X_train = preprocess_pipeline.fit_transform(train_data)


# Extract the training target
target = "Survived"
y_train = train_data[target]

# Create models
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

from sklearn import linear_model, preprocessing, model_selection 

logistic_clf = linear_model.LogisticRegression()

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()


num_folds = 10

# Train models
for name, clf in { "Logistic": logistic_clf,
                   "SVM": svm_clf,
                   "Decision Tree": tree_clf,
                   "Random Forest": forest_clf
                 }.items():
    
    print("Model: ", name)
    clf.fit(X_train, y_train)
    
    X_test = preprocess_pipeline.transform(test_data)
    y_pred = clf.predict(X_test)

    scores = cross_val_score(clf, X_train, y_train, cv=num_folds)
    print("\tMean score ({nf}-fold validation: {s:0.3f}".format(
            nf=num_folds,
            s=scores.mean()
            )
          )
