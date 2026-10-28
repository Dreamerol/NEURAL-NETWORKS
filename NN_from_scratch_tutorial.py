import tensorflow as tf

#Neural networks calculation the weights of the inputs for each layer,coded in the most simpler way
inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2
#the most simple way to compute the data to the first layer
simple_output = sum([inputs[i]*weights[i] for i in range(len(inputs))]) + bias
#print(simple_output)

input = [1,2,3,2.5]
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]
weights = [weights1,weights2,weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5

biases = [bias1,bias2,bias3]

def calculate_neuron(input, idx, weights, biases):
    suma = 0
    for i in range(len(input)):
        suma += input[i]*weights[idx][i]
    suma += biases[idx]
    return suma


output = [calculate_neuron(input, i, weights, biases) for i in range(len(biases))]
#print(output)


#better way
import numpy as np
input = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2

output = np.dot(input, weights) + bias
#print(output)
inputs = [1,2,3,2.5]
sample_inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]

weights_first_layer = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases_first_layer = [2,3,0.5]


weights_second_layer = [[0.4,0.33,-0.5],
           [11.2,3.5,6],
           [-0.6,0.5,0.66]]
biases_second_layer = [-1,2,-0.5]

output = [np.dot(weights_first_layer[i], inputs) + biases_first_layer[i] for i in range(len(biases_first_layer))]  # FIX: biases_first_layer instead of biases

output_first_layer = np.dot(sample_inputs, np.array(weights_first_layer).T) + biases_first_layer
output_second_layer = np.dot(output_first_layer, np.array(weights_second_layer).T) + biases_second_layer
#print(output_second_layer)

#we will try to code it in more reusable and flexible way
X = [
    [1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8]
    ]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1,n_neurons))
    def forward(self, input):
        self.output = np.dot(input,self.weights) + self.biases
        #we have transposed the weights matrix in the beginning when initializing it

layer1 = LayerDense(4,5)
layer2 = LayerDense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)

#ReLu function
input = [5,5,-5,56,-66]
output = []

for item in input:
    if item < 0:
        output.append(0)
    else:
        output.append(item)
    #output.append(max(0,item))
print(output)

class Activation_Relu:
    def forward(self,input):
        self.output = np.maximum(0, input)

activationFunc = Activation_Relu()
activationFunc.forward(layer1.output)
print(activationFunc.output)

#for multiclassification applied on the last layer we can use the softmax function
#it gives us the probability distribution and because the fact that we can have
#negative values on the output we will use the exp func
import math
output_layer = [3.4,4,5.6]
output_layers = [[3.4,4,5.6],
                 [6,6.6,3.2],
                 [3,4,5]
]
E = math.e
exp_values = np.exp(output_layer)
sumo = sum([E**item for item in output_layer])
norm_vector = [item/sumo for item in exp_values]
#we can divide np arrays [..]/lambda->[..]->it divides the first el from the output list is the first el from the input divided..
norm_values = exp_values / np.sum(exp_values)
print(norm_values)

#normalizing a matrix
def convert_row(row):
    suma = sum([np.exp(item) for item in row])
    return [float(np.exp(item)/suma) for item in row]

norm_values_matrix = [convert_row(output_layers[i]) for i in range(len(output_layers))]
print(norm_values_matrix)

class Activation_Softmax_function:
    def forward(self,input):
        exp_values = np.exp(input - np.max(input, axis=1,keepdims=True))
        #axis one means get the max element from the current row and keepdims we want the result to be in the same dimension
        self.probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        #we try to sum each column

activation2 = Activation_Softmax_function()
activation2.forward(layer2.output)
print(activation2.probabilities)

#Cross-Entropy
softmax_output = [0.7,0.2,0.1]
expected_output = [1,0,0]

loss = sum([-math.log(softmax_output[i])*expected_output[i] for i in range(len(softmax_output))])
print(loss)

#class targets and softmax outputs - we will compare the class target with the outputs on the given idx/where is the class target
softmax_outputs = np.array([[0.7,0.2,0.1],
                   [0.5,0.4,0.1],
                   [0.08,0.9,0.02]])
class_targets = [0,1,1]
#so we will want to get from the first row the obj on the idx
#then on the second row we will get the first idx..

print(softmax_outputs[[0,1,2], class_targets])
print([-np.log(softmax_outputs[i][class_targets[i]]) for i in range(len(class_targets))])

#print(-np.log(softmax_outputs[[0,1,2], [class_targets]]))
#this is like a map function on the 0th row we get the class target on 0 idx
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_true * y_pred_clipped, axis=1)  # FIX: y_true * y_pred_clipped (not y_pred * y_pred_clipped)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

y = np.array([[0,1], [1,0], [0,1]])
loss_function = Loss_CategoricalCrossEntropy()
print(layer2.output)
loss = loss_function.calculate(layer2.output, y)

print(loss)

#reducing loss with randomness
X = np.array([[1,2]])
y = np.array([[0,0,1]])
dense1 = LayerDense(2,3)
activation1 = Activation_Relu()
dense2 = LayerDense(3,3)
activation2 = Activation_Softmax_function()

loss_function = Loss_CategoricalCrossEntropy()
lowest_loss = 999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iter in range(100000):
    dense1.weights += 0.05*np.random.randn(2,3)
    dense1.biases += 0.05*np.random.randn(1,3)
    dense2.weights += 0.05*np.random.randn(3,3)
    dense2.biases += 0.05*np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.probabilities, y)
    predictions = np.argmax(activation2.probabilities, axis=1)
    accuracy = np.mean(predictions==y)
    #it compares each element to the els in the other list and based on the matching pairs
    #we calculate the accuracy -> [0 0 1] and [0 1 1] ->0=0,1 not=0... accuracy 66 percents
    
    if loss < lowest_loss:
        print("New value, iteration: ", iter, 'loss: ', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        #so we found a better loss -> we decreased the loss
        #so we get the new valued/modified layers
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
