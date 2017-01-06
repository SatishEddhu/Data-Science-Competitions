library(caret)
# install.packages("doParallel")
library(doParallel)

setwd("D:/Data Science/Algorithmica/Restaurant/") # set the working directory

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = TRUE)
restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = TRUE)

xtabs(~Type, restaurant_train) # missing MB type
summary(restaurant_train$revenue)

max_id = which(restaurant_train$revenue == max(restaurant_train$revenue))
restaurant_train[max_id,] # outlier revenue

# Smoothen the outlier revenue - change max. revenue to average revenue
restaurant_train$revenue[max_id] = mean(restaurant_train$revenue)
# Change outlier's year from 2000 to 2007 (average year)
restaurant_train$Open.Date = as.character(restaurant_train$Open.Date)
restaurant_train$Open.Date[max_id] = "1/7/2007"
# Change outlier's type to the missing value of "MB"
levels(restaurant_train$Type) = c(levels(restaurant_train$Type), "MB")
restaurant_train$Type[max_id] = "MB"

# transformed outlier
restaurant_train[max_id,]

# Compute num_days
restaurant_train$num_days = as.Date(c("31-12-2014"), format="%d-%m-%Y") - as.Date(restaurant_train$Open.Date, format="%m/%d/%Y")
restaurant_train$num_days = as.numeric(restaurant_train$num_days)

# Repeat the same feature engineering on test data
restaurant_test$Open.Date = as.character(restaurant_test$Open.Date)
restaurant_test$num_days = as.Date(c("31-12-2014"), format="%d-%m-%Y") - as.Date(restaurant_test$Open.Date, format="%m/%d/%Y")
restaurant_test$num_days = as.numeric(restaurant_test$num_days)

restaurant_train1 = restaurant_train[,-c(1,2,3)] # filter unwanted features
restaurant_test1 = restaurant_test[,-c(1,2,3)]
dim(restaurant_test1)
str(restaurant_train1)


# Using cforest
library(party)

c1 = makeCluster(detectCores())
registerDoParallel(c1)


resampling_strategy = trainControl(method="cv", number=50)
set.seed(100)
cforest_grid = expand.grid(mtry=c(8))
cforest_model = train(restaurant_train1[,-c(40)], restaurant_train1$revenue, method="cforest", trControl=resampling_strategy, controls=cforest_unbiased(ntree=2000), tuneGrid = cforest_grid)
cforest_model
# Private leaderboard RMSE - 1793662, rank #121
cforest_prediction = predict(cforest_model, restaurant_test1)
result = data.frame("Id"=restaurant_test$Id, "Prediction"=cforest_prediction)
write.csv(result, "submission_cforest_demo.csv", row.names = F)



# Gradient boosting on un-regularized trees

resampling_strategy = trainControl(method="cv", number=50)
gbm_grid = expand.grid(interaction.depth = 10,
                       n.trees = c(550),
                       shrinkage = c(0.003),
                       n.minobsinnode = c(5))
set.seed(100)

gbm_model = train(restaurant_train1[,-c(40)], restaurant_train1$revenue, method="gbm", trControl=resampling_strategy, tuneGrid = gbm_grid)
gbm_model
# Private leaderboard RMSE - 1777435, rank #44
gbm_prediction = predict(gbm_model, restaurant_test1)
result = data.frame("Id"=restaurant_test$Id, "Prediction"=gbm_prediction)
write.csv(result, "submission_gbm_demo.csv", row.names = F)



# Gradient boosting on regularized trees (extreme gradient boosted trees)

resampling_strategy = trainControl(method="cv", number=50)

xgb_tree_grid = expand.grid(nrounds = c(550), 
                            max_depth = 2, 
                            eta = c(0.005), 
                            gamma = c(1), 
                            colsample_bytree = c(0.3), 
                            min_child_weight = c(1.5) )

set.seed(100)
# Works only with formula interface; error with dataframe interface -> Error in train.default(restaurant_train[, -c(1, 2, 3, 43)], restaurant_train$revenue, : Stopping
xgbTree_model = train(revenue ~ ., restaurant_train1, method="xgbTree", trControl=resampling_strategy, tuneGrid = xgb_tree_grid)
xgbTree_model
# Private leaderboard RMSE - 1746262, rank #6
xgbTree_prediction = predict(xgbTree_model, restaurant_test1)
result = data.frame("Id"=restaurant_test$Id, "Prediction"=xgbTree_prediction)
write.csv(result, "submission_xgbTree_demo.csv", row.names = F)

combined_prediction = 0.75*xgbTree_prediction + 0.25*gbm_prediction
result = data.frame("Id"=restaurant_test$Id, "Prediction"=combined_prediction)
write.csv(result, "submission_combined_demo.csv", row.names = F)

stopCluster(c1)