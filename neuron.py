from random import random
from math_utils import sigmoid, sigmoid_derivative, relu 

class Neuron:
    def __init__(self, nb_inputs, activation_function=sigmoid, derivative_activation_function=None):
        self.weights = [random() for _ in range(nb_inputs)]
        self.bias = random()
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function or sigmoid_derivative
        
    def activate(self, inputs):
        self.inputs = inputs # Store inputs for backpropagation
        self.z = sum(w * x for w,x in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_function(self.z)
        return self.output
    
    def compute_gradient(self, error):
        # Local error
        self.delta = error * self.derivative_activation_function(self.z)
        # Weight gradients
        self.weight_gradients = [self.delta * x for x in self.inputs]
        self.bias_gradient = self.delta
    
    def update_weights(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.weight_gradients)]
        self.bias -= learning_rate * self.bias_gradient
    
    def __str__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"