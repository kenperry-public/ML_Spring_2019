import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pdb

import os
import subprocess

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

# Tools
from sklearn import preprocessing, model_selection 
from sklearn.tree import export_graphviz

# Models
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier


# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

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
        """
        I am really cheating here ! Am ignoring all columns except for "Sex"
        """
        
        # To see that I am cheating, look at the number of columns of X !
        print("SexToInt:transform: Cheating alert!, X has {c} columns.".format(c=X.shape[-1]) )
        
        sex = X["Sex"]
        X["Sex"] = 0
        X[ sex == "female" ] = 1
        
        return(X)

class TitanicHelper():
    def __init__(self, **params):
        return

    def make_numeric_pipeline(self, features):
        num_pipeline = Pipeline([
                ("select_numeric", DataFrameSelector( features )),
                ("imputer", SimpleImputer(strategy="median")),
                ])
        
        return num_pipeline

    def make_cat_pipeline(self, features):
        cat_pipeline = Pipeline([
                ("select_cat", DataFrameSelector( features )),
                ("imputer", MostFrequentImputer()),
                ("sex_encoder", SexToInt() ),
                ])

        return cat_pipeline

    def make_pipeline(self, num_features=["Age", "SibSp", "Parch", "Fare"], cat_features=["Sex", "Pclass"] ):
        num_pipeline = self.make_numeric_pipeline(num_features)
        cat_pipeline = self.make_cat_pipeline(cat_features)

        preprocess_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline),
                ]
                                           )
        feature_names = num_features.copy()
        feature_names.extend(cat_features)

        return preprocess_pipeline, feature_names

    def run_pipeline(self, pipeline, data):
        # Run the pipelinem return an ndarray
        data_trans = pipeline.fit_transform(data)

        return data_trans


    def make_logit_clf(self):
        # New version of sklearn will give a warning if you don't specify a solver (b/c the default solver -- liblinear -- will be replaced in future)
        logistic_clf = linear_model.LogisticRegression(solver='liblinear')

        return logistic_clf

    def make_tree_clf(self):
        tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)

        return tree_clf

    def fit(self, clf, train_data, target_name):
        pipeline, feature_names = self.make_pipeline()
        self.feature_names = feature_names

        train_transf = self.run_pipeline(pipeline, train_data)

        train_transf_df = pd.DataFrame(train_transf, columns=feature_names)

        y_train = train_data[target_name]
        X_train = train_transf_df

        self.X_train = X_train
        self.y_train = y_train

        clf.fit(X_train, y_train)
    
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        self.scores = scores

        return clf

    def export_tree(self, tree_clf, out_file, feature_names, target_classes, to_png=True):
        dot_file = out_file + ".dot"

        ret = { "dot_file": dot_file }

        export_graphviz(
            tree_clf,
            out_file=dot_file,
            feature_names=feature_names,
            class_names=target_classes,
            rounded=True,
            filled=True
            )

        if to_png:
            png_file = out_file + ".png"
            cmd = "dot -Tpng {dotf} -o {pngf}".format(dotf=dot_file, pngf=png_file)
            ret["png_file"] = png_file

            retval = subprocess.call(cmd, shell=True)
            ret["dot cmd rc"] = retval

        return ret

    def partition(self, X, y, conds=[]):
        mask = pd.Series(data= [ True ] * X.shape[0], index=X.index )
        X_filt = X.copy()

        for cond in conds:
            (col, thresh) = cond
            print("Filtering column {c} on {t}".format(c=col, t=thresh) )
            cmp = X[ col ] <= thresh
            mask = mask & cmp

        return (X[mask], y[mask], X[~mask], y[~mask])
            

