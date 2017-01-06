library(rpart)
library(caret)
library(e1071)

setwd("D:/Data Science/Algorithmica/Titanic/") # set the working directory
titanic_train = read.csv("train.csv", na.strings = c("NA",""))
titanic_test = read.csv("test.csv")
str(titanic_train)

titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_test$Survived = NA
titanic_test$Pclass = as.factor(titanic_test$Pclass)
str(titanic_test)

# combine both the train and test data to apply 'feature engineering' on both
combi = rbind(titanic_train, titanic_test)
dim(combi)
str(combi)

combi$Name = as.character(combi$Name)

strsplit(combi$Name[1], split='[,.]')[[1]][2]

# extract the title like 'Mr', "Miss', 'Sir', etc.
combi$Title = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title = sub(' ', '', combi$Title)

# examine the titles
table(combi$Title)

# combine some titles into one
combi$Title[combi$Title %in% c('Mlle', 'Mme')] = 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
combi$Title[combi$Title %in% c('the Countess', 'Dona', 'Jonkheer', 'Lady')] = 'Lady'

# Make 'Title' as factor
combi$Title = factor(combi$Title)

# Family size is parents + children + siblings + spouse + oneself
combi$FamilySize = combi$Parch + combi$SibSp + 1

# Extract the surname
combi$Surname = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
table(combi$Surname)

# Family ID = 'Family size' + 'Surname'
combi$FamilyID = paste(as.character(combi$FamilySize), combi$Surname, sep="")
# This logic is not fool-proof: Husband+wife - 1 ticket. Wife's two brothers in 2 separate tickets.
# Surnames 'Backstrom' & 'Gustafsson'. Family sizes are, wife=4, husband=2, both brothers = 3.
# So, husband and wife will get different family IDs, both brothers will get same family ID, but
# different from their sister's (the wife)
# Example 2: Mother and her two children, along with the mother's parents/siblings. Only the two
# children get the same family ID; mother gets another ID. Surname 'Richards'
# Example 3: Two different families of same size and with same surname get the same familyID.
# Surname 'Davies' - two families with 3Davies; see both test and train data. Five people get
# this family ID

table(combi$FamilyID)
# Three single 'Johnson's of different families would have the same family ID. So, knock them off
# and call them a small family
combi$FamilyID[combi$FamilySize <= 2] = 'Small'

xtabs(~FamilyID + FamilySize, combi)

# A few seem to have slipped through the cracks here.Shows we still have plenty of families 
# with only 1 or 2 members. Perhaps some families had different last names. Let's clean them up.
table(combi$FamilyID)

famIDs = data.frame(table(combi$FamilyID)) # Map of familyID & count of people with that familyID
famIDs = famIDs[famIDs$Freq <= 2, ] # Get familyIDs to be cleared up

# Cleaning up size < 3 families
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] = 'Small'

# Random forest also cannot handle categorical variables with more than 53 levels.
# But FamilyID has 61 levels. Let us tone that down.
combi$FamilyID2 = combi$FamilyID
combi$FamilyID2[combi$FamilySize <= 3] = "Small"
# FamilyID2 has only 22 levels now

# Mark FamilyID as a 'factor' attribute
combi$FamilyID2 = as.factor(combi$FamilyID2)
combi$FamilyID = factor(combi$FamilyID)

# filling missing fares
which(is.na(combi$Fare)) # 1044
# manually setting the only missing 'Fare' by observing similar entries (viz. PClass,Embarked,SibSp,Parch)
combi$Fare[1044] = 7.8875

# Extracting per person fare ( = fare/[number of people on that ticket])
combi$Ticket = as.character(combi$Ticket)
tickets = data.frame(table(combi$Ticket))
tickets$Var1 = as.character(tickets$Var1)
str(tickets)
head(tickets)

# mapping 'ticket number' to its 'frequency of occurence'/count
ht = list()
ht[tickets$Var1] = tickets$Freq
head(ht)
head(tickets$Var1)

combi$ticketCount = sapply(combi$Ticket, FUN = function(x){ht[[x]]})
combi$FarePP = combi$Fare / combi$ticketCount


# Filling the missing data

# filling in the 177 missing 'Age' values using ML approach
summary(combi$Age)
age_model = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                  combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] = predict(age_model, combi[is.na(combi$Age),])

# filling in the 2 missing 'Embarked' values with the majority value
summary(combi$Embarked)
combi$Embarked[which(is.na(combi$Embarked))] = "S"

# Divide the data (including FamilyID2 and other corrected attributes) into train and test
titanic_train = combi[1:891,]
titanic_test = combi[892:1309,]
summary(titanic_train)


# model building. Produced a better prediction accuracy of 0.80861 on Kaggle!
resampling_strategy = trainControl(method="cv", number=10)
tree_grid = expand.grid(.cp=seq(0.014,0.015,0.0001))
# kaggle accuracy of 0.80861
set.seed(100)
tree_model = train(titanic_train[,c("Sex","Pclass","Embarked","FarePP","ticketCount","Age","SibSp","Parch","FamilySize","FamilyID","Title")], titanic_train$Survived, method="rpart", trControl = resampling_strategy, tuneGrid = tree_grid)
titanic_test$Survived = predict(tree_model, titanic_test, type="raw")
titanic_test$treeVote = titanic_test$Survived
write.csv(titanic_test[,c("PassengerId","Survived")], "submission_tree.csv", row.names = F)





rf_resampling_strategy = trainControl(method="boot", number=25)
rf_grid = expand.grid(.mtry=c(3))
set.seed(100)
# kaggle accuracy of 0.75598
rf_model = train(titanic_train[,c("Sex","Pclass","Embarked","FarePP","Age","SibSp","Parch","FamilySize","FamilyID2","Title")], titanic_train$Survived, method="rf", trControl=rf_resampling_strategy, tuneGrid=rf_grid, ntree=2000)
rf_model
titanic_test$Survived = predict(rf_model, titanic_test[,c("Sex","Pclass","Embarked","FarePP","Age","SibSp","Parch","FamilySize","FamilyID2","Title")], type="raw")
titanic_test$rfVote = titanic_test$Survived
write.csv(titanic_test[,c("PassengerId","Survived")], "submission_rf.csv", row.names = F)


#kNN approach
dummy_obj = dummyVars(~Sex +  Pclass + Embarked + FarePP + Age + SibSp + Parch + FamilySize + FamilyID + Title, titanic_train)
titanic_train1 = as.data.frame(predict(dummy_obj, titanic_train))
titanic_train1$Survived = titanic_train$Survived
titanic_test1 = as.data.frame(predict(dummy_obj, titanic_test))

knn_resample_strategy = trainControl(method="cv", number=10)
set.seed(100)
# kaggle accuracy of 0.72727
knn_model = train(Survived ~ ., titanic_train1, method="knn", trControl=knn_resample_strategy)
knn_model
titanic_test1$Survived = predict(knn_model, titanic_test1)
titanic_test1$PassengerId = titanic_test$PassengerId
write.csv(titanic_test1[,c("PassengerId","Survived")], "submission_knn.csv", row.names = F)
titanic_test$knnVote = titanic_test1$Survived

# Trying with another ensemble model, a forest of conditional inference trees
#install.packages("party")
library(party)
# Kaggle accuracy of 0.81818
ci_model = cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + FarePP + 
                     Embarked + Title + FamilySize + FamilyID, data = titanic_train,
                   controls=cforest_unbiased(ntree=2000, mtry=3))

titanic_test$Survived = predict(ci_model, titanic_test, OOB = TRUE, type = "response")
titanic_test$ciVote = titanic_test$Survived
write.csv(titanic_test[,c("PassengerId","Survived")], "submission_ci.csv", row.names = F)



# convert factors to numbers for applying majority votes & weights
titanic_test$treeVote = as.numeric(titanic_test$treeVote) - 1
titanic_test$ciVote = as.numeric(titanic_test$ciVote) - 1
titanic_test$rfVote = as.numeric(titanic_test$rfVote) - 1
titanic_test$knnVote = as.numeric(titanic_test$knnVote) - 1

titanic_test$weightedVote = (0.4*titanic_test$ciVote) + (0.3*titanic_test$treeVote) + (0.2*titanic_test$rfVote) + (0.1*titanic_test$knnVote)
# Kaggle accuracy of 0.82775 (an improvement over conditional inference trees!)
# We had 5 entries where ciVote = 0, and all the other three votes = 1. They made the difference :)
titanic_test$Survived = ifelse(titanic_test$weightedVote > 0.5, 1, 0)
write.csv(titanic_test[,c("PassengerId","Survived")], "submission_ensemble_weightedVote.csv", row.names = F)
