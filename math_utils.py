import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

def relu(x):
    return max(0, x)
