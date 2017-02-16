from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
from sklearn import metrics
import numpy as np
import pandas as pd
import os

tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/Otto")

otto_train = pd.read_csv("train2.csv") # in last column, converted 'Class_x' to 'x'
otto_train = otto_train.as_matrix()
otto_train.shape
x_train = otto_train[:,1:94]
x_train = x_train.astype(np.float32) # this must be float32 for classifier to work
y_train = otto_train[:,[94]] # this must NOT be float32 so that it can be one-hot encoded

otto_test = pd.read_csv("test.csv")
otto_test = otto_test.as_matrix()
otto_test.shape
x_test = otto_test[:,1:94]
x_test = x_test.astype(np.float32) # this must be float32 for classifier to work


# Hidden layers generally use sigmoid perceptrons
# Output layer uses softmax for overall interpretability of all the 9 outputs
def model_function(features, targets, mode):      
    targets = tf.one_hot(targets,9,1,0)
    
    #hlayers = layers.stack(features, layers.fully_connected, [100,50], activation_fn=tf.sigmoid)
    hlayers = layers.stack(features, layers.fully_connected, [200], activation_fn=tf.nn.relu,
                           weights_initializer=layers.xavier_initializer(uniform=True,seed=100),
                           weights_regularizer=layers.l1_l2_regularizer(0.5,0.5))
    
    outputs = layers.fully_connected(inputs=hlayers,     
                                     num_outputs=9,
                                     activation_fn=None) # Use "None" as activation function specified in "softmax_cross_entropy" loss
    
    
    # Calculate loss using cross-entropy error; also use the 'softmax' activation function
    loss = losses.softmax_cross_entropy (outputs, targets)
    
    optimizer = layers.optimize_loss(
                  loss=loss,                  
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.8,
                  optimizer="SGD")

    # Class of output (i.e., predicted number) corresponds to the perceptron returning the highest fractional value
    # Returning both fractional values and corresponding labels    
    probs = tf.nn.softmax(outputs)
    # label should be 'argmax()+1' to transform 0-8 range to 1-9 range
    return {'probs':probs, 'labels':(tf.argmax(probs, 1)+1)}, loss, optimizer 
    # Applying softmax on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')

# See the issue with one-hot-encoding for labels from 1 to 9
#a = [0,1,2,3,4,5,6,7,8,9]
#b = tf.one_hot(a,9,1,0)
#tf.Session().run(b)
#c = [[1,2,4,31],[21,66,5,7]]
#d=tf.argmax(c,1)+1
#tf.Session().run(d)

# y_train-1 is used so that 1:9 gets mapped to 0:8 which then gets one-hot-encoded to the desired form
# Otherwise, in the one-hot-encoded form, the 1st bit is never set to 1
classifier.fit(x=x_train, y=y_train-1, steps=10000, batch_size=1000)
for var in classifier.get_variable_names()    :
    print var, ": ", classifier.get_variable_value(var).shape, " - ", classifier.get_variable_value(var)

# Predict the outcome of test data using model
predictions = classifier.predict(x_test, as_iterable=True)
for i, p in enumerate(predictions):
   print("Prediction %s: %s, probs = %s" % (i+1, p["labels"], p["probs"]))

# Compute the accuracy metrics
# call with as_iterable=False to get all predictions together
predictions = classifier.predict(x_test)
#metrics.accuracy_score(np.argmax(y_test, 1), predictions['labels'])

# See how it fares against the 'train' data itself
predictions = classifier.predict(x_train)
metrics.accuracy_score(y_train, predictions['labels'])


#col_names = ['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
col_names = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
df = pd.DataFrame(data=predictions['probs'],index=otto_test[:,0],columns=col_names)
df.to_csv("otto_submission_nn.csv")
# modify the header of csv file manually to add 'id' as the first column