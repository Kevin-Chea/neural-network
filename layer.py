from neuron import Neuron
from math_utils import sigmoid

class Layer:
    def __init__(self, nb_neurons, nb_inputs, activation_function=sigmoid):
        self.neurons = [Neuron(nb_inputs=nb_inputs, activation_function=activation_function) for _ in range(nb_neurons)]
        
    def forward(self, inputs):
        self.outputs = [neuron.activate(inputs) for neuron in self.neurons]
        return self.outputs
    
    def backward(self, output_errors):
        for i, neuron in enumerate(self.neurons):
            neuron.compute_gradient(output_errors[i])
        input_errors = [
            sum(neuron.delta * neuron.weights[j] for neuron in self.neurons)
            for j in range(len(self.neurons[0].weights))
        ]
        return input_errors
    
    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
            
    
    def __str__(self):
        return f"Layer: {[str(neuron) for neuron in self.neurons]}"