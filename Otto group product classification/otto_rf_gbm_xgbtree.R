library(caret)
library(doParallel)

# setwd("D:/Data Science/Algorithmica/Otto")
setwd("D:/Data Science/Competitions/Otto group product classification")

otto_train = read.csv("train.csv", header=TRUE, sep=",")
# convert target from "Class_n" to "n"
result = sapply(as.character(otto_train$target), FUN=function(x) {(strsplit(x, split='[_]')[[1]][2])})
otto_train$target = as.numeric(result)
otto_train$target = as.factor(otto_train$target)

otto_test = read.csv("test.csv", header=TRUE, sep=",")
# adding 9 more columns (95 to 103); initializing them to 0
new_cols = paste("Class", 1:9, sep="_")
otto_test[, new_cols] = 0  # equivalent to: otto_test[, c(95:103)] = 0

head(otto_train)
head(otto_test)

resampling_strategy = trainControl(method="none")

c1 = makeCluster(detectCores())
registerDoParallel(c1)


rf_grid = expand.grid(.mtry=18)
set.seed(100)
rf_model = train(otto_train[,c(2:94)], otto_train$target, method="rf",
                 trControl=resampling_strategy, tuneGrid=rf_grid, ntree=2000)
rf_result = predict(rf_model, otto_test[,2:94], type="prob") # score - 0.52
otto_test[, new_cols] = rf_result
write.csv(otto_test[,c(1,95:103)], "otto_submission_rf.csv", row.names = F)


gbm_grid = expand.grid(interaction.depth = 6,
                       n.trees = c(1000),
                       shrinkage = c(0.1),
                       n.minobsinnode = c(10))

set.seed(100)
gbm_model = train(otto_train[,2:94], otto_train$target, method="gbm", trControl=resampling_strategy,
                  tuneGrid=gbm_grid)
gbm_result = predict(gbm_model, otto_test[,2:94], type="prob") # score - 0.49309
otto_test[, new_cols] = gbm_result
write.csv(otto_test[,c(1,95:103)], "otto_submission_gbm.csv", row.names = F)

xgbtree_grid = expand.grid(nrounds = 70,
                            max_depth = 8,
                            eta = 0.5,
                            gamma = 1, 
                            colsample_bytree = 1.0,
                            min_child_weight = 1.0)

set.seed(100)
# works only with formula interfae
xgbtree_model = train(target ~ ., otto_train[,2:95], method="xgbTree", trControl=resampling_strategy,
                      tuneGrid=xgbtree_grid)
xgbtree_result = predict(xgbtree_model, otto_test[,2:94], type="prob") # score - 0.48505
otto_test[, new_cols] = xgbtree_result
write.csv(otto_test[,c(1,95:103)], "otto_submission_xgbtree.csv", row.names = F)