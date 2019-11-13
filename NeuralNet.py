#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:54:22 2019

@author: kavin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.impute import SimpleImputer



class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        train_dataset, test_dataset = model_selection.train_test_split(raw_input, test_size=0.2)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        cols = len(train_dataset.columns)
        rows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(cols -1)].values.reshape(rows, cols-1)
        self.y = train_dataset.iloc[:, (cols-1)].values.reshape(rows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])
       
        
        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.test_dataset = test_dataset
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        
        if activation == "sigmoid":
            #self.__sigmoid(self, x)
            return self.__sigmoid(x)
        elif activation == "tanh":
            #self.__tanh(self, x)
            return self.__tanh(x)
        elif activation == "ReLu":
            #self.__ReLu(self, x)
            return self.__ReLu(x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            #self.__sigmoid_derivative(self, x)
            return self.__sigmoid_derivative(x)
        elif activation == "tanh":
            #self.__tanh_derivative(self, x)
            return self.__tanh_derivative(x)
        elif activation == "ReLu":
            #self.__ReLu_derivative(self, x)
            return self.__ReLu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self,x):
        return np.tanh(x);
    
    def __ReLu(self,x):
        return np.maximum(0,x)
        


    
    
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self,x):
        return (1 - np.tanh(x)**2)
    
    def __ReLu_derivative(self,x):
        return 1.0*(x>0)


    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):

        encode_X = LabelEncoder()
        X=X.apply(encode_X.fit_transform)
        Impute = SimpleImputer(missing_values=np.nan, strategy='mean')
        Impute.fit(X)
        X = X.values
        #Normalizing the X value to the range 0,1 with MinMax normalizer
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        df_preprocessed = min_max_scaler.fit_transform(X)
        #Converting the values to a DataFrame for further processing
        df_preprocessed = pd.DataFrame(df_preprocessed)
        df_preprocessed= df_preprocessed.drop_duplicates()

        return df_preprocessed

    # Below is the training function

    def train(self, activation, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        self.X12 = self.__activation(in1, activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, activation)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))
            
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))
            
        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))

            self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self,activation="sigmoid", header = True):
        res_value = 0
        #raw_test_input = self.test_dataset
        test_data = self.test_dataset
        #print("Raw Test Input: " + str(test_data))
        test_dataset = self.preprocess(test_data)
        self.X = test_dataset.iloc[:, 0:(len(test_dataset.columns) -1)].values.reshape(len(test_dataset.index), len(test_dataset.columns)-1)
        self.y = test_dataset.iloc[:, (len(test_dataset.columns)-1)].values.reshape(len(test_dataset.index), 1)
        res_value = self.forward_pass(activation)
        test_error = 0.5 * np.power((res_value - self.y), 2)
        print("Test Output error: "+ str(np.sum(test_error)))
        
        return test_error

    

if __name__ == "__main__":
    

    dataset="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    #dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    #dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
    print("Sigmoid Activation Function\n")
    neural_network_sigmoid = NeuralNet(dataset)
    neural_network_sigmoid.train(activation="sigmoid")
    testError = neural_network_sigmoid.predict(activation="sigmoid")

    print("\n\nTanh Activation Function\n")
    neural_network_tanh = NeuralNet(dataset)
    neural_network_tanh.train(  activation="tanh")
    testError = neural_network_tanh.predict(activation="tanh")
    
    print("\n\nReLu Activation Function\n")
    neural_network_ReLu = NeuralNet(dataset)
    neural_network_ReLu.train(activation="ReLu")
    testError = neural_network_ReLu.predict(activation="ReLu")  
    
    
    
    
    
    

    