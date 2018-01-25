
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import model_selection 
from sklearn import metrics
from sklearn import ensemble


d = pd.read_csv("../data/airline100K.csv")

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d[col] = preprocessing.LabelEncoder().fit_transform(d[col])
  
X_cat = preprocessing.OneHotEncoder().fit_transform(d[vars_cat])     # sparse mx  (less RAM, but also training runs 30x faster)
X = sparse.hstack((X_cat, d[vars_num]))                              # sparse mx
      
y = np.where(d["dep_delayed_15min"]=="Y",1,0)                        # numpy array


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=123)



## TRAIN
md = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 10)

%time md.fit(X_train, y_train)

## %time md.fit(X_train.toarray(), y_train)   # slow if not sparse!


## SCORE
y_pred = md.predict_proba(X_test)[:,1]

metrics.confusion_matrix(y_test, y_pred>0.5)
metrics.roc_auc_score(y_test, y_pred)


