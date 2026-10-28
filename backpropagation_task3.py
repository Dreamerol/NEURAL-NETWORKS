
#task 3
import numpy as np

#so now we have two inputs ->[x1, x2] -> we pass them to two neuron hidden layer so we need to multiply by the matrix of the weights
#then sigmoid so [x1, x2] -> [weights matrix]->[w11.x1 + w12.x2, w21.x1 + w22.x2] -> delta([hidden_neurons_values])
def sigmoid(x):
  return 1/(1+np.exp(-x))

x31 = np.linspace(-50, 50, 101)
x32 = 20*np.random.rand(101)
x3 = np.array([x31, x32]).T
y3 = np.zeros((101,2))


def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

for i in range(101):
  y3[i] = [sigmoid(1/10*x3[i, 0]-1/5*x3[i, 1]-1), sigmoid(1/3*x3[i, 0]+1/4*x3[i, 1]+3)]

w11 = 1
w12 = 2
w21 = 2
w22 = 3

# print(y3[0][0])

#IMPORTANT FORMULA -> when we have two outputs values we yi1 and yi2 the formula is:
#(1/2N)*[sum(yi1 - yi1_precited)^2 +sum(yi2-yi2_predicted)^2
def MSE_LOSS(x_values, y_values):
  suma = 0
  N = len(x_values)
  for i in range(N):
    suma += (y_values[i][0]-sigmoid(w11*x_values[i][0] + w12*x_values[i][1]))**2 + (y_values[i][1] - sigmoid(w21 * x_values[i][0] + w22 * x_values[i][1])) ** 2
  return 1/(2*N)* suma

#Now is time to calculate the derivatives - the cool things is that they are symmetric
#so we have (1/2N)*2(yi-delta(w11.x1+w12.x2)).delta'(w11.x1 + w12.x2).x1 cause the derivative of the f(x1, x2) by w11 is x1
#the second quadratic sum has derivative of zero -> cause has no variables w11 so they are like constant for w11 so c' = 0
def derivative_w11(x_values, y_values):
  suma = 0
  N = len(y_values)
  for i in range(N):
    suma += (y_values[i][0] - sigmoid(w11*x_values[i][0] + w12*x_values[i][1]))*sigmoid_derivative(w11*x_values[i][0] + w12*x_values[i][1])*x_values[i][0]

  return -(1/N) * suma

#so we have (1/2N)*2(yi-delta(w11.x1+w12.x2)).delta'(w11.x1 + w12.x2).x2 cause the derivative of the f(x1, x2) by w12 is x2
def derivative_w12(x_values, y_values):
  suma = 0
  N = len(y_values)
  for i in range(N):
    suma += (y_values[i][0] - sigmoid(w11 * x_values[i][0] + w12 * x_values[i][1])) * sigmoid_derivative(w11 * x_values[i][0] + w12 * x_values[i][1]) * x_values[i][1]

  return -(1 / N) * suma

#analogically (1/2N)*2(yi-delta(w21.x1+w22.x2)).delta'(w21.x1 + w22.x2).x1 cause the derivative of the f(x1, x2) by w21 is x1
def derivative_w21(x_values, y_values):
  suma = 0
  N = len(y_values)
  for i in range(N):
    suma += (y_values[i][0] - sigmoid(w21 * x_values[i][0] + w22 * x_values[i][1])) * sigmoid_derivative(w21 * x_values[i][0] + w22 * x_values[i][1]) * x_values[i][0]  # FIX: w21 instead of literal 21

  return -(1 / N) * suma

#analogically (1/2N)*2(yi-delta(w21.x1+w22.x2)).delta'(w21.x1 + w22.x2).x2 cause the derivative of the f(x1, x2) by w22 is x2
def derivative_w22(x_values, y_values):
  suma = 0
  N = len(y_values)
  for i in range(N):
    suma += (y_values[i][0] - sigmoid(w21 * x_values[i][0] + w22 * x_values[i][1])) * sigmoid_derivative(w21 * x_values[i][0] + w22 * x_values[i][1]) * x_values[i][1]

  return -(1 / N) * suma

for i in range(1000):
  print(MSE_LOSS(x3,y3))
  learning_rate = 0.01
  w11 -= learning_rate*derivative_w11(x3,y3)
  w12 -= learning_rate * derivative_w12(x3, y3)
  w21 -= learning_rate * derivative_w21(x3, y3)
  w22 -= learning_rate *derivative_w22(x3, y3)
#finding the weights/parameters with the least loss error
print(w11,w12,w21,w22)
