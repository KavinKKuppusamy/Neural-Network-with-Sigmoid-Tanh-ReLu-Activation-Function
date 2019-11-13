# Neural-Network-with-Sigmoid-Tanh-ReLu-Activation-Function
Implementation of Neural Network with Back-propagation algorithm with Sigmoid,TanH,ReLu activation functions.

Title: 
Implementation of Neural Network.
Assumptions Considered:
User will supply the accurate dataset paths for Training, Validation and Testing process.
Accomplishments:
•	Given the dataset, the pre-processing of dataset was performed with help of sckitlearn packages. Some of the pre-processing steps are
    Converting Non-numerical values to numerical values
    If there is a missing value, replace it with the mean of the column.
    Feature scaling the data using MinMax normalization method to scale the data values in range (0,1).
•	Along with the Input layer and output layer, created the Neural network with 2 hidden layers
•	Further with the Sigmoid activation, we implement the other two activation function namely ReLu and Tanh.
•	After trained the network with the training set, we calculate the accuracy with the test error for the test dataset.
•	Corrected the weights while propagating through the layers.

Datasets Used:
 Iris Dataset
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Hepatitis Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data
