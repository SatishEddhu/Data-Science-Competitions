from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
from sklearn import metrics
import numpy as np
import os
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)
os.chdir("/home/algo/Algorithmica/DigitRecognizer")

digit_train = pd.read_csv("train.csv")
digit_train = digit_train.as_matrix()
digit_train.shape
x_train = digit_train[:,1:785]
x_train = x_train.astype(np.float32) # this must be float32 for classifier to work
y_train = digit_train[:,[0]] # this must NOT be float32 so that it can be one-hot encoded

digit_test = pd.read_csv("test.csv")
digit_test = digit_test.as_matrix()
digit_test.shape
x_test = digit_test
x_test = x_test.astype(np.float32) # this must be float32 for classifier to work


# Hidden layers generally use sigmoid perceptrons
# Output layer uses softmax for overall interpretability of all the 10 outputs
def model_function(features, targets, mode):
    targets = tf.one_hot(targets,10,1,0)
    
    # input layer
    # Reshape features to 4-D tensor (55000x28x28x1)
    # MNIST images are 28x28 pixels
    # batch size corresponds to number of images: -1 represents ' compute the # images automatically (55000)'
    # +1 represents the # channels. Here #channels =1 since grey image. For color image, #channels=3
    input_layer = tf.reshape(features, [-1,28,28,1])
    
    
    # Computes 32 features using a 5x5 filter
    # Padding is added to preserve width
    # Input Tensor Shape: [batch_size,28,28,1]
    # Output Tensor Shape: [batch_size,28,28,32]
    conv1 = layers.conv2d(
                inputs=input_layer,
                num_outputs=32,
                kernel_size=[5,5],
                stride=1,
                padding="SAME", # do so much padding such that the feature map is same size as input
                activation_fn=tf.nn.relu)
    
    # Pooling layer 1
    # Pooling layer ith a 2x2 filter and stride 2
    # Input shape: [batch_size,28,28,32]
    # Output shape: [batch_size,14,14,32]
    pool1 = layers.max_pool2d(inputs=conv1,kernel_size=[2,2], stride=2)
    
    # Convolution layer 2
    # Input: 14 x 14 x 32 (32 channels here)
    # Output: 14 x 14 x 64  (32 features/patches fed to each perceptron; discovering 64 features)
    conv2 = layers.conv2d(
                inputs=pool1,
                num_outputs=64,
                kernel_size=[5,5],
                stride=1,
                padding="SAME", # do so much padding such that the feature map is same size as input
                activation_fn=tf.nn.relu)
    
    # Pooling layer 2
    # Input: 14 x14 x 64
    # Output: 7 x 7 x 64
    pool2 = layers.max_pool2d(inputs=conv2,kernel_size=[2,2], stride=2)
    
     
    # Flatten the pool2 to feed to the 1st layer of fully connected layers
    # Input size: [batch_size,7,7,64]
    # Output size: [batch_size, 7x7x64]
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
         
     
    # Connected layers with 100, 20 neurons
    # Input shape: [batch_size, 7x7x64]
    # Output shape: [batch_size, 10]
    fclayers = layers.stack(pool2_flat, layers.fully_connected, [100,50], 
                             activation_fn=tf.nn.relu, weights_regularizer=layers.l1_l2_regularizer(1.0,2.0),
                             weights_initializer=layers.xavier_initializer(uniform=True,seed=100))
    
    
    outputs = layers.fully_connected(inputs=fclayers, 
                                     num_outputs=10, # 10 perceptrons in output layer for 10 numbers (0 to 9)
                                     activation_fn=None) # Use "None" as activation function specified in "softmax_cross_entropy" loss
    
    
    # Calculate loss using cross-entropy error; also use the 'softmax' activation function
    loss = losses.softmax_cross_entropy (outputs, targets)
    
    optimizer = layers.optimize_loss(
                  loss=loss,                  
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.001,
                  optimizer="Adam")

    # Class of output (i.e., predicted number) corresponds to the perceptron returning the highest fractional value
    # Returning both fractional values and corresponding labels    
    probs = tf.nn.softmax(outputs)
    return {'probs':probs, 'labels':tf.argmax(probs, 1)}, loss, optimizer 
    # Applying softmax on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')
classifier.fit(x=x_train, y=y_train, steps=100000, batch_size=100)

# Compute the accuracy metrics
predictions = classifier.predict(x_test)

length = x_test.shape[0]
digit_id = range(1,length+1,1)
dict = {'ImageId':digit_id, 'Label': predictions['labels']}

# Kaggle score of 0.992
out_df = pd.DataFrame.from_dict(dict)
out_df.set_index('ImageId', inplace=True)
out_df.to_csv("submit_cnn.csv")