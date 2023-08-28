# Karavatt, Nikhil Das
# 1002_085_391
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf

#function to choose which activation to be called per layer
def activation_function(activations):
  if activations.lower() == "sigmoid":#.lower() will help to convert the text into lowercase and then check the condition
    return sigmoid #calling sigmoid for activation
  elif activations.lower() == "linear":#.lower() will help to convert the text into lowercase and then check the condition
    return linear #calling linear for activation
  elif activations.lower() == "relu":#.lower() will help to convert the text into lowercase and then check the condition
    return relu #calling linear for activation

#function to split the data
def split_data(X_train, Y_train, split_range=[0.2, 0.7]): #helper code given by TA Jason to split the given data into X_train, Y_train, X_val and Y_val 
  start = int(split_range[0] * X_train.shape[0]) # defining the start for splitting
  end = int(split_range[1] * X_train.shape[0]) # defining the stop for splitting
  return np.concatenate((X_train[:start], X_train[end:])), np.concatenate((Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end] # returning the spilt

#Sigmoid Function
def sigmoid(x):
  return tf.nn.sigmoid(x) #returning sigmoid activation on input x

#Linear Function
def linear(x):
  return x #returning same value as its linear 

#Relu Function:
def relu(x):
  return tf.nn.relu(x) #returning relu activation on input x

#Function for Adding Bias layer to different input matrices
def add_bias_layer(X):
  z=tf.ones([X.shape[0],1],dtype=tf.float32) #making matrix of 1 column and rows is number of samples using tensorflow
  return tf.concat([z,X],axis=1) # returning concatinated value of matrix made above and input X

#initializing weights
def initial_weights(X_train,layers,seed):
  weights=[] #creating an empty list for weights to be appended first
  for i in range (len(layers)): # starting the for loop to go across the length of the layers
    np.random.seed(seed) #using seed to get exact random values for weights
    if i == 0: #when i is 0 the row of weight is the number of features from X input plus a bias while column is number of nodes in that layer
      w= tf.Variable(np.random.randn(X_train.shape[1]+1,layers[i]),dtype=tf.float32,trainable=True) 
      #using tf.varaiable to make sure that the whole weight is taken into considertion while performing gradient using gradient tape
      weights.append(w) #appending weight to the weights list
    else: #for weights after first layer the input will be output of previous layer and a bias is added in the row while column is number of nodes
      w= tf.Variable(np.random.randn(layers[i-1]+1,layers[i]),dtype=tf.float32,trainable=True) 
      weights.append(w) #appending weight to the weights list 
  return weights #returning updated weights

#Function to get activated values of each code
def Activation(weights, X_batch, layers,activations):
  activated_value = [tf.matmul(tf.transpose(weights[0]), tf.transpose(X_batch))] #performing WX for the first layer
  Activated_nodes = [activation_function(activations[0])(activated_value[0])] #using the value from the previous line getting the activated function value for that layer
  for i in range(1,len(layers)): #for remaining layers we will have to add bias to the input of those layers
    X_batch = tf.transpose(add_bias_layer(tf.transpose(Activated_nodes[-1]))) #adding bias
    activated_value.append(tf.matmul(tf.transpose(weights[i]), X_batch)) #performing WX for the ith layer
    Activated_nodes.append(activation_function(activations[i])(activated_value[i]))#using the value from the previous line getting the activated function value for that layer
  return Activated_nodes #returning activated values 

#Function to get output
def output(weights, X_batch, layers,activations):
  y_pred = tf.transpose(Activation(weights, X_batch, layers,activations)[-1]) #transposing and taking the activated values in the last layer
  return y_pred #returning the predicted value

#function to perform loss
def cost(loss,Y_pred,Y_batch):
  if loss.lower() == 'mse': #.lower() will help to convert the text into lowercase and then check the condition
    return tf.reduce_mean(tf.square(Y_pred-Y_batch)) #returning mean square error between Y_pred and Y_batch
  elif loss.lower() == 'svm': #.lower() will help to convert the text into lowercase and then check the condition
    return tf.reduce_mean(tf.maximum(0,1-Y_batch*Y_pred)) #returning hinge loss between Y_pred and Y_batch
  elif loss.lower() == 'cross_entropy': #.lower() will help to convert the text into lowercase and then check the condition
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred,labels=Y_batch)) #returning cross entropy between Y_pred and Y_batch

#function to train the neural network
def training(weights,X_batch,Y_batch,layers,activations,alpha,loss):
  with tf.GradientTape() as tape: #using gradient tape to update weights
    tape.watch(weights) # this makes sure that weight is a part of computation so derivative of weight with respect to loss is found for all the  weights 
    Y_pred = output(weights, X_batch, layers,activations) #function to get the output
    error = cost(loss,Y_pred,Y_batch) #function to get the respective error 
  gradients = tape.gradient(error,weights) #this helps to get the gradient which was discussed a few lines above
  for i in range(len(weights)): # we need to update the weight for each layer as per its derivatives.
    weights[i].assign_sub(alpha*gradients[i]) #using learning rate to get the weights updated accordingly
  return weights #returning the updated weights

# The Multi Layer Neural Network using Tensor Flow function
def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],weights=None,seed=2):
  X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train, split_range=validation_split) #spilt function as discussed before to split the data
  if weights is None: #if weights are given none then we need to call the weights initization function
    weights = initial_weights(X_train,layers,seed)
  else: #if weights is given then we need to make sure that we make it as tensor flow variable to make sure that 
    for i in range(len(layers)): # these weights get counted for finding the gradients while using gradient tape
      weights[i] = tf.Variable(weights[i],dtype=tf.float32,trainable=True) #this will help weights to be a tensor variable
  X_val = tf.convert_to_tensor(X_val,dtype=tf.float32) #converting  X_val data to tensor
  X_train = tf.convert_to_tensor(X_train,dtype=tf.float32) #converting  X_train data to tensor
  Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float32) #converting  Y_train data to tensor
  Y_val = tf.convert_to_tensor(Y_val,dtype=tf.float32) #converting  Y_val data to tensor
  X_train_b = add_bias_layer(X_train) #adding bias to the input values for training
  X_val_b = add_bias_layer(X_val) #adding bias to the input values for testing
  MSE_Per_Epoch = [] #MSE list to append MSE per epoch
  for _ in range(epochs): #loop for epoch
    for j in range(0,X_train_b.shape[0],batch_size): #loop for batch
      X_batch = X_train_b[j:j+batch_size] #taking one batch at a time to train the model
      Y_batch = Y_train[j:j+batch_size] #taking one batch at a time to train the model
      weights = training(weights,X_batch,Y_batch,layers,activations,alpha,loss) #updating the weight matrices with the new weights
    if X_train_b.shape[0] % batch_size != 0: #handling samples which is remaining after the series of whole batch is over
      X_batch = X_train_b[-(X_train_b.shape[0] % batch_size):] #taking remaining samples to train the model
      Y_batch = Y_train[-(X_train_b.shape[0] % batch_size):] #taking remaining samples to train the model
      weights = training(weights,X_batch,Y_batch,layers,activations,alpha,loss) #updating the weight matrices with the new weights
    Y_Pred = output(weights, X_val_b, layers,activations) # finding output after each epoch using validation sample
    MSE_Per_Epoch.append(cost(loss,Y_Pred,Y_val)) #list of average MSE for each epoch using validation samples
  out = output(weights, X_val_b, layers,activations) #output of the multilayer neural network using tensor flow
  return [weights, MSE_Per_Epoch, out] #returning final weights, MSE per Epoch, and final Output

                                                  #############
                                                  ###  Note ###
                                                  #############
"""
In the above code the remainder samples when number of samples are not perfectly divisible by batch size the rremaining samples are taken twice
This is because of the bug in the batch code shared by TA Jason. As discusssed with him, I am keeping the same bugged batch code as then 
only the whole test cases will pass or else some test cases will fail as the test cases are made as per the batch code shared by TA
if we dont use the second condition (if X_train_b.shape[0] % batch_size != 0) in batch code it will rectify the code as then the 
remanining samples wont be repeated twice
"""