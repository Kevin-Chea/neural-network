from math_utils import sigmoid
from layer import Layer

class Neural_Network():
    def __init__(self, layers=[]):
        self.layers = layers
        
    def add_layer(self, nb_neurons, nb_inputs, activation_function=sigmoid):
        self.layers.append(Layer(nb_neurons, nb_inputs, activation_function))
    
    def forward(self, inputs):
        if not self.layers:
            raise ValueError("The neural network has no layers.")
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, target_outputs):
        output_errors = [output-target for output, target in zip(self.layers[-1].outputs, target_outputs)]
        for layer in reversed(self.layers):
            output_errors = layer.backward(output_errors)
    
    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def train(self, inputs_train, expected_outputs_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, expected_outputs in zip(inputs_train, expected_outputs_train):
                # Forward
                outputs = self.forward(inputs)
                # Mean squarred error
                loss = sum((output - expected) ** 2 for expected, output in zip(expected_outputs, outputs))
                total_loss += loss
                # Backward
                self.backward(expected_outputs)
                self.update_weights(learning_rate)
            if epoch % 100 == 0:
                avg_loss = total_loss / len(inputs_train)
                print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
    
    def __str__(self):
        return "\n".join([f"Layer {i+1}: {layer}" for i, layer in enumerate(self.layers)])