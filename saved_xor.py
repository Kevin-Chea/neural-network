from neural_network import Neural_Network

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

nn = Neural_Network.load_model("xor.pkl")
for input in inputs:
    print(f"For input {input}, the output is : {nn.forward(input)}")