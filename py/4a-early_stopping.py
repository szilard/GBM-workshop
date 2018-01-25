

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
  
X_cat = preprocessing.OneHotEncoder().fit_transform(d[vars_cat])     
X = sparse.hstack((X_cat, d[vars_num]))                             
      
y = np.where(d["dep_delayed_15min"]=="Y",1,0)                    


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=123)
X_subtrain, X_earlystop, y_subtrain, y_earlystop = model_selection.train_test_split(X_train, y_train, test_size=0.2)




## TRAIN
md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=10000)
%time md.fit(X_subtrain, y_subtrain, early_stopping_rounds=10,  eval_set=[(X_earlystop, y_earlystop)], verbose = False)

md.best_iteration_


## SCORE
y_pred = md.predict_proba(X_test, num_iteration=md.best_iteration_)[:,1]

metrics.confusion_matrix(y_test, y_pred>0.5)
metrics.roc_auc_score(y_test, y_pred)



## overfitting:

md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=1000)
%time md.fit(X_subtrain, y_subtrain, eval_set=[(X_earlystop, y_earlystop)], eval_metric = "auc", verbose = True)

## TODO: get scoring history

y_pred = md.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, y_pred)

