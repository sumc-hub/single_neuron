# Python program to implement a single neuron neural network

# Import all necessary libraries
import numpy as np
from numpy import exp, array, random, dot

# Class to create a neural network with single neuron
class NeuralNetwork():
	
    def __init__(self):
        
        random.seed(1)
        # Creating 3x1 weight matrix
        self.weight_matrix = 2 * random.random((3, 1)) - 1
        
    def sigmoid(self, x):
        # Applying sigmoid as activation function
        return 1 / (1+np.exp(-x))
    
    def sigmoid_derv(self, x):
        # Derivative of sigmoid function
        return x * (1-x)
	
    def forward(self, inputs):
        # Forward propagation
        return self.sigmoid(dot(inputs, self.weight_matrix))

    def train(self, training_inputs, training_outputs, epochs):
        """
        Train the neural network using backpropagation.

        Parameters:
        - training_inputs (array): Input data for training.
        - training_outputs (array): Expected output data for training.
        - epochs (int): Number of training iterations.
	    """
         
        for epoch in range(epochs):
            # Forward pass with training inputs
            output = self.forward(training_inputs)

            # Calculate the error
            error = training_outputs - output

            # Backpropagation
            adjustment = dot(training_inputs.T, error * self.sigmoid_derv(output))

            # Update weights
            self.weight_matrix += adjustment

if __name__ == "__main__":
    # Example usage
    neural_network = NeuralNetwork()

    # Example training data 
    training_inputs = array([[1, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 0]])

    # Example training outputs
    training_outputs = array([[1, 0, 0, 1]]).T

    # Train the neural network for 10000 epochs
    neural_network.train(training_inputs, training_outputs, epochs=10000)

    # Test the trained neural network with new input
    new_input = array([0, 1, 0])
    prediction = neural_network.forward(new_input)
    print(f"Prediction for {new_input}:", prediction)

