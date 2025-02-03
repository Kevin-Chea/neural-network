import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0, x)

def softmax(inputs):
    """
    Compute softmax of a vector inputs.
    """
    exp_values = [math.exp(x) for x in inputs]
    total = sum(exp_values)
    return [exp_value / total for exp_value in exp_values]

def categorical_loss_entropy(expected_one_hot, pred_one_hot):
    epsilon = math.exp(-12) # Avoid log(0) issues
    pred_one_hot = [max(min(p, 1 - epsilon), epsilon) for p in pred_one_hot] # Clip values between epsilon and 1 - epsilon
    return -sum(e * math.log(p) for e,p in zip(expected_one_hot, pred_one_hot))