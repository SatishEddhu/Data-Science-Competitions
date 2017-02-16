from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import metrics
import numpy as np
import os
import pandas as pd


tf.logging.set_verbosity(tf.logging.INFO)

# read digit images of 28 x 28 = 784 pixels size
# target is image value in [0,9] range; one-hot encoded to 10 columns
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_validation = mnist.validation.images
y_validation = mnist.validation.labels

x_test = mnist.test.images
y_test = mnist.test.labels

type(x_train)
type(y_train)
x_train.shape
x_validation.shape
x_test.shape
y_train.shape
y_validation.shape
y_test.shape


# For maximum learning, combining all the data we have
a = np.concatenate((x_train,x_validation,x_test),axis=0)
a.shape
b = np.concatenate((y_train,y_validation,y_test))
b.shape


# Hidden layers generally use sigmoid perceptrons
# Output layer uses softmax for overall interpretability of all the 10 outputs
def model_function(features, targets, mode):      
    
    hlayers = layers.stack(features, layers.fully_connected, [1000,100,50,20],
                           activation_fn=tf.nn.relu, weights_regularizer=layers.l1_l2_regularizer(1.0,2.0),
                           weights_initializer=layers.xavier_initializer(uniform=True,seed=100))
    
    # hidden layers have to be fully connected for best performance. So, no option in tensorflow for
    # non-fully connected layers; need to write custom code to do that
    
    outputs = layers.fully_connected(inputs=hlayers, 
                                     num_outputs=10, # 10 perceptrons in output layer for 10 numbers (0 to 9)
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
    return {'probs':probs, 'labels':tf.argmax(probs, 1)}, loss, optimizer 
    # Applying softmax on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')
# train on entire MNIST data
classifier.fit(x=a, y=b, steps=60000, batch_size=100)

# checking how much of overfitting/learning has happened
predictions = classifier.predict(a)
metrics.accuracy_score(np.argmax(b, 1), predictions['labels'])

# Read Kaggle's test data and predict its digits using the above model
os.chdir("/home/algo/Algorithmica/DigitRecognizer")
digit_test = pd.read_csv("test.csv")
digit_test = digit_test.as_matrix()
digit_test = digit_test.astype(np.float32) # this must be float32 for classifier to work

# Predicting on Kaggle's test data
predictions = classifier.predict(digit_test)

length = digit_test.shape[0]
digit_id = range(1,length+1,1)
dict = {'ImageId':digit_id, 'Label': predictions['labels']}

# Kaggle score of up to 0.99600 (sometimes 0.99300 due to randomization of batch splits)
out_df = pd.DataFrame.from_dict(dict)
out_df.set_index('ImageId', inplace=True)
out_df.to_csv("submit_dnn.csv")

