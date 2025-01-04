from math_utils import sigmoid
from layer import Layer
from neuron import Neuron
import pickle

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
    
    def train_with_batch(self, load_balanced_batch, batch_per_epoch, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            total_samples = 0
            for _ in range(batch_per_epoch):
                inputs_train, expected_outputs_not_formatted = load_balanced_batch()
                # One-hot labels
                expected_outputs_train = [[1 if i == y else 0 for i in range(10)] for y in expected_outputs_not_formatted]
                
                batch_loss = 0
                for inputs, expected_outputs in zip(inputs_train, expected_outputs_train):
                    # Forward
                    outputs = self.forward(inputs)
                    # Mean squarred error
                    loss = sum((output - expected) ** 2 for expected, output in zip(expected_outputs, outputs))
                    batch_loss += loss
                    # Backward
                    self.backward(expected_outputs)
                    self.update_weights(learning_rate)

                # Update total loss and samples
                total_loss += batch_loss
                total_samples += len(inputs_train)
                
            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    def save_model(self, filepath):
        """
        Save weights and bias in the specified file
        
        Params:
        - filepath (str): path of the file which will contain data of the neural network
        """
        model_data = {
            #"architecture": [len(layer.neuron) for layer in self.layers],
            "layers": [
                {
                    "weights": [neuron.weights for neuron in layer.neurons],
                    "biases": [neuron.bias for neuron in layer.neurons],
                    "activations": [neuron.activation_function for neuron in layer.neurons],
                    "derivatives": [neuron.derivative_activation_function for neuron in layer.neurons]
                }
                for layer in self.layers
            ]
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved in {filepath}")
        
    def load_model(filepath):
        """
        Load a complete model from a file
        
        Params:
        - filepath (str): path of the file which contains data of the neural network
        
        Returns:
        - Neural_Network: loaded model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        layers_data = model_data["layers"]
        
        layers = []
        for layer_data in layers_data:
            layer = Layer(0, 0)
            neurons = []
            for weights, bias, activation_function, derivative_activation_function in zip(
              layer_data["weights"]  ,
              layer_data["biases"],
              layer_data["activations"],
              layer_data["derivatives"]
            ):
                neuron = Neuron(0)
                neuron.weights = weights
                neuron.bias = bias
                neuron.activation_function = activation_function
                neuron.derivative_activation_function = derivative_activation_function
                neurons.append(neuron)
            layer.neurons = neurons
            layers.append(layer)
        
        nn = Neural_Network(layers)
        print(f"Loading model of neural network from {filepath}:")
        print(nn)
        return nn
        
    def __str__(self):
        return "\n".join([f"Layer {i+1}: {layer}" for i, layer in enumerate(self.layers)])