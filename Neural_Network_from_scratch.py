import numpy as np

#Activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def Relu(x):
    if x >= 0:
        return x
    else:
        return 0
    
def Relu_deriv(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2  # FIX: correct tanh derivative (was 1/cos(x)**2 which is tan derivative)

activation_functions = {
    sigmoid : sigmoid_derivative,
    Relu: Relu_deriv,
    tanh: tanh_deriv
}
   
class Neuron:
    def __init__(self, n_inputs, activation):
        self.activation = activation
        self.activation_derivative = activation_functions[activation]
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        self.output = 0
        self.delta = 0

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.output = np.dot(self.weights, self.inputs) + self.bias
        self.output = self.activation(self.output)
        return self.output
    
    def compute_delta(self, target=None, next_weights=None, next_deltas=None):
        if target is not None:
            # Output layer
            self.delta = (self.output - target) * self.activation_derivative(self.output)
        else:
            # when we are in the hidden layer we get the deltas from the nth layer
            # and multiply it by the matrix with weights - so to see the contribution
            # of each neuron to the loss
            self.delta = self.activation_derivative(self.output)*np.dot(next_weights, next_deltas)
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.delta * self.inputs
        self.bias -= learning_rate * self.delta


class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
    
    def forward(self, inputs):
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        self.outputs = outputs
        return outputs
    
    def compute_deltas(self, targets=None, next_layer=None):
        if targets is not None:
            for i, neuron in enumerate(self.neurons):
                neuron.compute_delta(target=targets[i])
        else:
            for i, neuron in enumerate(self.neurons):
                next_weights = np.array([n.weights[i] for n in next_layer.neurons])
                next_deltas = np.array([n.delta for n in next_layer.neurons])
                neuron.compute_delta(next_weights=next_weights, next_deltas=next_deltas)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

class NeuralNetwork:
    def __init__(self, layers_info, n_input):
        self.layers = []
        input_size = n_input
        for n_neurons, activation in layers_info:
            layer = DenseLayer(n_neurons, input_size, activation)
            self.layers.append(layer)
            input_size = n_neurons

    def forwardNN(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.forward(result)
        return result
    
    def backpropagation(self, targets, learning_rate):
        self.layers[-1].compute_deltas(targets=targets)
        for i in (range(len(self.layers) - 2,-1, -1)):
            self.layers[i].compute_deltas(next_layer=self.layers[i+1])
        for layer in self.layers:
            layer.update_weights(learning_rate)

samples_inputs = [[0,0,1], [1,1,1]]
samples_outputs = [[0], [1]]
NN = NeuralNetwork([
    (2, Relu),
    (1, sigmoid)
], n_input=3)

learning_rate = 0.5
epochs = 200
        
for epoch in range(epochs):
    for x, y in zip(samples_inputs, samples_outputs):
        output = NN.forwardNN(x)
        loss = 0.5 * sum((np.array(output) - np.array(y))**2)
        NN.backpropagation(y, learning_rate)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


#advanced implementation
class Neuron2:
    def __init__(self,n_inputs,bias=0.1):
        self.weights = np.random.randn(len(n_inputs))
        self.bias = bias
        self.output = sum([n_inputs[i]*self.weights[i] for i in range(len(n_inputs))]) + bias
    def update_neuron(self, n_inputs, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = sum([n_inputs[i]*self.weights[i] for i in range(len(n_inputs))]) + bias

    def forward_neuron(self,activ_func):
        self.output = activ_func(self.output)
    
    def backpropagation(self, derivs, learning_rate):
        self.delta = 0
        for i in range(len(derivs)):
            self.delta += derivs[i]*self.weights[i]
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*self.delta
        self.bias -= learning_rate*self.bias

class DenseLayer2:
    def __init__(self, number_neurons, activation_function, n_inputs):
        self.neurons = []
        for i in range(number_neurons):
            neuron=Neuron2(n_inputs)
            self.neurons.append(neuron)
        self.activation_function = activation_function
    
    def forward(self):
        for n in self.neurons:
            n.forward_neuron(self.activation_function)
    def getOutput(self):
        result = []
        for item in self.neurons:
            result.append(item.output)
        return result
    
    def backProp(self,derivs,learning_rate):
        for i in range(len(self.neurons)):
            self.neurons[i].backpropagation(derivs,learning_rate)

class NeuralNetwork2:
    def __init__(self, layers_info, n_input, n_output):
        first_layer = DenseLayer2(layers_info[0][0], layers_info[0][1], n_input)
        self.layers = []
        self.layers.append(first_layer)
        self.target = n_output
        for i in range(len(layers_info)):
            if i == 0:
                continue
            layer = DenseLayer2(layers_info[i][0], layers_info[i][1], self.layers[-1].getOutput())  # FIX: use getOutput() instead of .output
            self.layers.append(layer)

    def add_layer(self, one_layer_info):
        layer=DenseLayer2(self.layers[-1].getOutput(), one_layer_info[0], one_layer_info[1])
        self.layers.append(layer)
    
    def forwardNN(self, input):
        for i in range(len(self.layers)):
            j = 0
            for item in self.layers[i].neurons:
                if j == 0:
                    result = input
                else:
                    result = self.layers[i-1].getOutput()
                item.update_neuron(result, item.weights, item.bias)
                j += 1
        return self.getResult()
    
    def getResult(self):
        print(self.layers[-1].getOutput())
    
    def ADAM_optimization(self, learning_rate, target):
        # FIX: closed parenthesis correctly; np.log receives neuron output value
        self.loss = -sum([target[i]*np.log(self.layers[-1].neurons[i].output) for i in range(len(target))])
        delta_one = [target[i]*(1-self.layers[-1].neurons[i].output) for i in range(len(target))]
        for i in range(len(self.layers[-1].neurons)):
            self.layers[-1].neurons[i].bias -= learning_rate*self.layers[-1].neurons[i].bias
            self.layers[-1].neurons[i].backpropagation(delta_one,learning_rate)

        for i in range(len(self.layers) - 2,0,-1):
            derivs = [item.delta for item in self.layers[i+1].neurons]
            self.layers[i].backProp(derivs,learning_rate)

def Relu2(x):
    if x >= 0:
        return x
    else:
        return 0
    
def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

def f(x):
    return 1/x

def softmax_func(output):
    suma = sum([np.exp(item) for item in output])
    return [np.exp(item)/suma for item in output]

NN2 = NeuralNetwork2([[2, sigmoid_func]], [0, .7, 0.3], [0, 0, 1])
NN2.getResult()
print(87777)
samples_inputs2 = [[-9,6,6], [3,5,8], [0,0,1]]
samples_outputs2 = [[0,1,0], [0,1,0], [0,0,1]]

def optimizer(NN, epochs, samples_inputs, sample_outputs):
    for i in range(epochs):
        for j in range(len(samples_inputs)):
            NN.forwardNN(samples_inputs[j])
            NN.ADAM_optimization(0.1, sample_outputs[j])
        print(f"Epoch: {i}, Loss: {NN.loss}")  # FIX: f-string instead of string concatenation
