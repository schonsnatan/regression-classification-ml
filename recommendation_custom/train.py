import pandas as pd

from sklearn import naive_bayes
from sklearn.base import BaseEstimator, ClassifierMixin

class TeoMeWhatPrior(BaseEstimator, ClassifierMixin):


    def __init__(self, product_names: list):
        self.product_names = product_names
        self.product_names.sort()
        self.models = {i: None for i in self.product_names}


    def fit_one(self, X: pd.DataFrame, product_name: list):
        features = list(set(X.columns.tolist()) - set([product_name]))
        self.models[product_name] = naive_bayes.BernoulliNB()
        self.models[product_name].fit(X[features], X[product_name])


    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        self.classes_ = self.product_names
        for p in self.product_names:
            self.fit_one(X, p)
        return self


    def predict_proba_one(self, X: pd.DataFrame, product_name:str):
        features = list(set(X.columns.tolist()) - set([product_name]))
        probas = self.models[product_name].predict_proba(X[features])[:,1]
        return probas


    def predict(self, X: pd.DataFrame):
        probas = self.predict_proba(X)
        return (probas.apply(lambda x: x == probas.max(axis=1))
                      .apply(lambda x: x[x == True].index.tolist()[0], axis=1))


    def predict_proba(self, X: pd.DataFrame):
        df = pd.DataFrame()
        for p in self.product_names:
            df[p] = self.predict_proba_one(X, p)
        return df