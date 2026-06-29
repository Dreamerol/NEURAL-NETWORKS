import numpy as np
# FIX: removed bogus 'from fontTools.varLib.avar.plan import WEIGHTS' — wrong library, unrelated to this task


def sigmoid(x):
  return 1/(1+np.exp(-x))

def tanh(x):
  return np.tanh(x)
n = 2
#number of neurons in hidden layer
m = 1
# vector dimension [x1,.;......xm]

s = 2 # number of vectors


def NN(X, W1, W2):
  hidden_layer_output = tanh(np.dot(W1, X))
  print(hidden_layer_output)
  output = tanh(np.dot(W2, hidden_layer_output))
  return output

#W1 - [matrix w11,w12...]]nxm.[x1,x2,...xm]mx1 -> W2[w11,122....]nx1.[yh1,yh2,...yhn]1xn -> [output]1x1
weights_one = np.random.randn(n, m)
weights_two = np.random.randn(1, n)
x_vector = np.random.randn(m, 1)

size_weights_one = m * n
size_weights_two = 1 * n

Weights = np.random.rand(size_weights_one + size_weights_two)
print(NN(x_vector, weights_one, weights_two))


def Loss(W, X_input, Y_output):  # FIX: swapped argument order to match args= call below
  sum = 0
  W1 = W[:size_weights_one].reshape(n, m)
  W2 = W[size_weights_one:].reshape(1, n)
  for i in range(s):
    sum += (Y_output[i] - NN(X_input[i].T, W1, W2)) ** 2
  return sum/(m*n)

def f(x):
  return np.sin(x)

#!!!!!!!!!!!!!!!!!always TRANPOSE THE MATRIX!!!!!!!!! ;(
X_input = np.array([[2], [4]])
y_output = np.array([f(X_input[i]) for i in range(s)])
from scipy.optimize import minimize
print(minimize(Loss, Weights, args=(X_input, y_output), method='L-BFGS-B'))  # FIX: args order matches Loss(W, X_input, Y_output)


# print(X_input)
# for i in range(s-1):
# print(X_input[i])

#FINAL WEIGHTS
result = minimize(Loss, Weights, args=(X_input, y_output), method='BFGS').x  # FIX: args order
W1_final = result[:size_weights_one].reshape(n, m)
W2_final = result[size_weights_one:].reshape(1, n)

print(W1_final, W2_final)

x_example = [0.5]
print(NN(x_example, W1_final, W2_final))
