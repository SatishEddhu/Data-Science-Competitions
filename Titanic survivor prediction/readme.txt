Scored prediction accuracy of 0.82775 on titanic survivors public leaderboard.

Used an ensemble of the following 4 models; used a weighted voting strategy to compute final vote
- forest of conditional trees (cforest)
- random forest (rf)
- CART (rpart)
- KNN

Did feature engineering to create many new features
- Title
- Fare per person in a ticket
- Family size
- Family ID