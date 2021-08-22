import os
import random
import numpy as np
import pandas as pd
from sklearn import datasets
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

def random_seed(seed=42):
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def run_fold(fold, df):
    df1 = df.copy()
    
    df_train = df1.loc[df['kfold']!=0].reset_index(drop=True)
    df_valid = df1.loc[df['kfold']==0].reset_index(drop=True)
    
    model = LogisticRegression(n_iters=1000, lr=0.0001)
    model.fit(df_train.iloc[:,:-2].values, df_train['target'].values)
    predictions = model.predict(df_valid.iloc[:,:-2].values)
    
    print("LR classification accuracy:", accuracy(df_valid['target'].values, predictions))
    

class LogisticRegression:
    def __init__(self, n_iters = 100, lr = 0.001):
        self.n_iters = n_iters
        self.lr = lr
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0 
        
        for _ in range(self.n_iters):
            predicted = self._sigmoid(np.dot(X, self.weight) + self.bias)
            
            dw = 1/n_samples * np.dot(X.T, (predicted - y))
            db = 1/n_samples * np.sum(predicted - y)
            
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
            
    def _sigmoid(self, X):
        sig = 1/(1+np.exp(-X))
        return sig
    
    
if __name__ == "__main__":
    random_seed(seed = 42)

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    df = pd.DataFrame(X)
    df['target'] = y

    df = df.sample(frac=1).reset_index(drop=True)

    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
    for f, (t_, v_) in enumerate(kfold.split(df, df['target'])):
        df.loc[v_, 'kfold'] = f
        
    delayed_func = [delayed(run_fold)(fold, df) for fold in range(3)]
    Parallel(n_jobs = 3, prefer = 'processes')(delayed_func)

