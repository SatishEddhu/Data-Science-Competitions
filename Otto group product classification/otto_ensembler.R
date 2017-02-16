setwd("D:/Data Science/Competitions/Otto group product classification")

# Read the results of base models
nn_result = read.csv("otto_submission_nn.csv", header=TRUE, sep=",") #0.52
head(nn_result)
rf_result = read.csv("otto_submission_rf.csv", header=TRUE, sep=",") #0.53191
head(rf_result)
gbm_result = read.csv("otto_submission_gbm.csv", header=TRUE, sep=",") #0.49309
head(gbm_result)
xgbtree_result = read.csv("otto_submission_xgbtree.csv", header=TRUE, sep=",") #0.48505
head(xgbtree_result)

# Combine the results to generate the final result
combi_result = 0.4*xgbtree_result + 0.3*nn_result + 0.2*gbm_result + 0.1*rf_result
combi_result$id = xgbtree_result$id
# Score: 0.45025; rank #700
write.csv(combi_result, "otto_submission_final.csv", row.names = F)
