This Kaggle competition problem is hosted at https://www.kaggle.com/c/otto-group-product-classification-challenge.
It is a 'Multiclass classification' problem.

I used an ensemble of the following models:
- deep neural network (in Python using Tensorflow)
- xgbTree (in R)
- gbm (in R)
- random forest (in R)

The input files train.csv and test.csv were downloaded from Kaggle.
train2.csv is a slight modification of train.csv - in last column, converted 'Class_x' to 'x'

How to run the code to generate my submission:
1) Run otto-deepnn.py using Tensorflow to generate the output file
   - otto_submission_nn.csv
2) Run R program otto_rf_gbm_xgbtree.R to generate the following 3 output files:
   - otto_submission_rf.csv
   - otto_submission_gbm.csv
   - otto_submission_xgbtree.csv
3) Run R program otto_ensembler.R to combine the above 4 outputs to generate the final output:
   - otto_submission_final.csv

This submission got a 'Multiclass Loss' public score of 0.45173 and private score of 0.45025 corresponding to the 
private leaderboard rank of #700

See the attached screenshot showing the public score.