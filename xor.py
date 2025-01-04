from neural_network import Neural_Network

# Create neural network
nn = Neural_Network()
# A 3 neurons layer
nn.add_layer(3, 2)
# 1 output
nn.add_layer(1, 3)

inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

outputs = [
    [0],
    [1],
    [1],
    [0],
]

nn.train(inputs, outputs, 50000, 0.1)
for input in inputs:
    print(nn.forward(input))
print(nn)
# If you want to save the model, uncomment the following line and adapt the filepath if needed
# nn.save_model("xor.pkl")