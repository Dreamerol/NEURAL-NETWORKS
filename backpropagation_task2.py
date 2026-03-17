
#zad 2
#Here the struct is a bit different 
#[x] -> f -> ax + b -> sigmoid -> g -> c*delta(ax+b) + d -> sigmoid -> finally we get delta(c*delta(ax+b) + d)
#we have four parameters


x2 = np.linspace(-25, 25, 101)
y2 = 1/10*sigmoid(3*x2-2)-100

a = 5
b = 7
c = 9
d = 4
def f(x):
  return a*x + b

def g(x):
  return c*x + d

def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

#so analogically we calculate the derivatives by all parameters


#(1/(n+1))*2(yi-delta(c*delta(f(xi)) + d))*(-delta'(g(delta(f(xi)))))*g' by d g' by a will give us *delta(f(xi))*f' by a which is c*delta'(f(xi))*f'(xi) and f' by a is xi
def derivative_MSE_a(x_values, y_values):
  n = len(y_values)
  suma = 0
  for i in range(n):
    suma += ((y_values[i] - sigmoid(c*sigmoid(a*x_values[i] + b) + d))*sigmoid_derivative(c*sigmoid(a*x_values[i] + b) + d)*c*sigmoid_derivative(a*x_values[i] + b)*x_values[i])
  return -2/(n+1) * suma

#(1/(n+1))*2(yi-delta(c*delta(f(xi)) + d))*(-delta'(g(delta(f(xi)))))*g' by d g' by b will give us *delta(f(xi))*f' by b which is c*delta'(f(xi))*f'(xi) and f' by b is one
def derivative_MSE_b(x_values, y_values):
  n = len(y_values)
  suma = 0
  for i in range(n):
    suma += ((y_values[i] - sigmoid(c*sigmoid(a*x_values[i] + b) + d))*sigmoid_derivative(c*sigmoid(a*x_values[i] + b) + d)*c*sigmoid_derivative(a*x_values[i] + b))
  return -2/(n+1) * suma

#(1/(n+1))*2(yi-delta(c*delta(f(xi)) + d))*(-delta'(g(delta(f(xi)))))*g' by d g' by c will give us *delta(f(xi))
def derivative_MSE_c(x_values, y_values):
  n = len(y_values)
  suma = 0
  for i in range(n):
    suma += ((y_values[i] - sigmoid(c*sigmoid(a*x_values[i] + b) + d))*sigmoid_derivative(c*sigmoid(a*x_values[i] + b) + d)*sigmoid(a*x_values[i] + b))

  return -2/(n+1) * suma 

#(1/(n+1))*2(yi-delta(c*delta(f(xi)) + d))*(-delta'(g(delta(f(xi)))))*g' by d g' by d is one
def derivative_MSE_d(x_values, y_values):
  n = len(y_values)
  suma = 0
  for i in range(n):
    suma += ((y_values[i] - sigmoid(c * sigmoid(a * x_values[i] + b) + d))* sigmoid_derivative(c * sigmoid(a * x_values[i] + b) + d))

  return -2 / (n + 1) * suma
def MSE(x_values, y_values):
  n = len(y_values)
  return (1/(n+1))*sum([(y_values[i] - sigmoid(c*sigmoid(a*x_values[i] + b) + d))**2 for i in range(len(y_values))])

learning_rate = 0.01
for i in range(100):
  print(MSE(x2, y2))
  a -= learning_rate*derivative_MSE_a(x2, y2)
  b -= learning_rate*derivative_MSE_b(x2, y2)
  c -= learning_rate*derivative_MSE_c(x2, y2)
  d -= learning_rate*derivative_MSE_d(x2, y2)

print(a,b,c,d)
