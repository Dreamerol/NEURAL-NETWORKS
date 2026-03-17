#task 1
#finding weights using backpropagation and gradient descent
#How our neural network works -> first we get [x] -> f -> ax+b -> sigmoid -> delta(f(x))
#so first we get the output of the hidden neuron and then we call sigmoid func
#[] -> LT/linear transformation/ -> [] -> id/identitet/ -> output

#our training data
x1 = np.linspace(-25, 25, 101)
y1 = 1/10*sigmoid(x1)-100

#first we set random values to our weights
a= 3
b= 4
learning_rate = 0.01

def f(x):
  return a*x + b

#The formula Min Squared Error -> (1/(n+1))*sum(yi - delta(f(xi)))^2
def MSE(y_values, x_values):
  n = len(y_values)
  return (1/(n+1))*sum([(y_values[i] - sigmoid(f(x_values[i])))**2 for i in range(len(y_values))])

#Backpropagation rule -> first we calculate all the derivatives by each weight - and then we substract them with 
#a learning rate from the actual weights values -> that is called Gradient Descent
#so we calculate first the derivative by a

#(1/(n+1))*2*sum(yi-delta(f(xi)))*(-delta'(f(xi)))*f'(xi) by a
#(axi+b)' by a is xi

def derivative_MSE_a(x_values, y_values):
  n = len(y_values)
  suma = 0
  for i in range(n):
    suma += (y_values[i]-sigmoid(f(x_values[i])))* sigmoid_derivative(f(x_values[i]))*x_values[i]
    #sigmoid(f(x_values[i]))*(1-sigmoid(f(x_values[i]))) - this is derivative of sigmoid function
  return -2/(n+1)*suma
# return (2/(n+1)) * sum([y_values[i] - f(x_values[i])for i in range(n)])*(-x_values[i])

#Analogically we calculate the derivative by b
#where i computes between 0 and n
#(1/(n+1))*2*sum[(yi-delta(f(xi)))*(-delta'(f(xi)))*f'(xi)] by b so (-2/(n+1))*sum[(yi-delta(f(xi)))*delta'(f(xi))*1
def derivative_MSE_b(x_values, y_values):
  n = len(y_values)
  suma= 0
  for i in range(n):
    suma += (y_values[i] - f(x_values[i])) * sigmoid(f(x_values[i])) * (1 - sigmoid(f(x_values[i])))
  return -2/(n+1)*suma
#return (2/(n+1)) * sum([y_values[i] - f(x_values[i])for i in range(n)]) *(-1)

#finally we print the loss, so to see the how it reduces due to Backpropagation and Gradient Descent
for i in range(100000):
  print(MSE(y1, x1))
  a -= learning_rate*derivative_MSE_a(x1, y1)
  b -= learning_rate*derivative_MSE_b(x1, y1)
print(a, b)
