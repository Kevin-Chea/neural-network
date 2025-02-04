from neuron import Neuron
from math_utils import sigmoid, sigmoid_derivative, softmax

class Layer:
    def __init__(self, nb_neurons, nb_inputs, activation_function=sigmoid, derivative_function=sigmoid_derivative):
        self.neurons = [Neuron(nb_inputs=nb_inputs, activation_function=activation_function, derivative_activation_function=derivative_function) for _ in range(nb_neurons)]
        
    def forward(self, inputs):
        if self.neurons[0].activation_function == softmax:
            all_z = [sum(w * x for w,x in zip(neuron.weights, inputs)) + neuron.bias for neuron in self.neurons]
            self.outputs = softmax(all_z)
            for (i, neuron) in enumerate(self.neurons):
                neuron.inputs = inputs
                neuron.z = all_z[i]
                neuron.output = self.outputs[i]
        else:
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