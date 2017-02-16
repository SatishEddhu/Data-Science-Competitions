This is the popular 'Titanic' problem on Kaggle hosted at https://www.kaggle.com/c/titanic
My submission scored an accuracy of 0.82775 on the public leaderboard with a corresponding
rank #122 as of February 2017. Please screen the attached screenshot.

Used an ensemble of the following 4 models; used a weighted voting strategy to compute final vote
- forest of conditional trees (cforest)
- random forest (rf)
- CART (rpart)
- KNN

Did feature engineering to create the following new features:
- Title
- Fare per person in a ticket
- Family size
- Family ID