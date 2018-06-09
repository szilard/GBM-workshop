
library(xgboost)
library(ROCR)


d <- read.csv("https://raw.githubusercontent.com/szilard/GBM-workshop/master/data/airline100K.csv") 
head(d)


set.seed(123)
N <- nrow(d)
idx <- sample(1:N, 0.9*N)  
d_train <- d[idx,]
d_test <- d[-idx,]

X <- Matrix::sparse.model.matrix(dep_delayed_15min ~ . - 1, data = d)   
## 1-hot encoding + sparse 
## needs to be done *together* (train+test) for alignment (otherwise error for new cats at scoring)
## still problem in live scoring scenarios
X[1:10,1:10]
X_train <- X[idx,]
X_test <- X[-idx,]

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
## special optimized data structure



## TRAIN CPU
system.time({
  md <- xgb.train(data = dxgb_train, objective = "binary:logistic", 
           nround = 1000, max_depth = 10, eta = 0.1)
})

## SCORE / AUC
phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]



## TRAIN GPU
system.time({
  md <- xgb.train(data = dxgb_train, objective = "binary:logistic", 
           nround = 1000, max_depth = 10, eta = 0.1, 
           tree_method = "gpu_hist")
})

## SCORE / AUC
phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]


