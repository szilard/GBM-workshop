
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import model_selection 
from sklearn import metrics
import lightgbm as lgb
import multiprocessing


d = pd.read_csv("../data/airline100K.csv")

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d[col] = preprocessing.LabelEncoder().fit_transform(d[col])
  
X_cat = preprocessing.OneHotEncoder().fit_transform(d[vars_cat])     # sparse mx   (less RAM, but also XGB runs 30x faster)
X = sparse.hstack((X_cat, d[vars_num]))                              # sparse mx
      
y = np.where(d["dep_delayed_15min"]=="Y",1,0)                        # numpy array


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=123)



## Method 1 - sklearn API

## TRAIN
md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=100)
%time md.fit(X_train, y_train)


## SCORE
y_pred = md.predict_proba(X_test)[:,1]

metrics.confusion_matrix(y_test, y_pred>0.5)
metrics.roc_auc_score(y_test, y_pred)



## Method 2 - orig lightgbm API

dlgb_train = lgb.Dataset(X_train, label = y_train)
dlgb_test = lgb.Dataset(X_test)


## TRAIN
param = {'num_leaves':512, 'learning_rate':0.1, 'verbose':0}
%time md = lgb.train(param, dlgb_train, num_boost_round = 100)


## SCORE
y_pred = md.predict(X_test)   

metrics.confusion_matrix(y_test, y_pred>0.5)
metrics.roc_auc_score(y_test, y_pred)


## try playing with the hyperparams e.g. num_leaves = 100,300,1000,3000; learning_rate=0.01,0.03,0.1;
## num_boost_round = 100,300,1000; check out further params in the docs
## (re-run from "TRAIN" part above)


