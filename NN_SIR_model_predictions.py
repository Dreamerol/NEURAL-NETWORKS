#using Neural networks for mathematical modelling for predicting and plotting the behaviour of SIR model
#how the variables affect each other during time
import numpy as np
import scipy.integrate as integrate
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

dt = 0.01
T = 10
t = np.arange(0, T+dt, dt)

beta = 0.07 #infectious rate
gama = 0.12 #recovery rate

nn_input = np.zeros((100*(len(t)-1), 3))
nn_output = np.zeros_like(nn_input)

def SIRderiv(s_i_r, to, beta=beta, gama=gama):
  ds = -beta*s_i_r[0] * s_i_r[1] #dS/dt = -b.SI
  di = beta*s_i_r[0] * s_i_r[1] - gama*s_i_r[1] #dI/dt = bSI - yI 
  dr = gama*s_i_r[1] #dR/dt = yI
  return [ds, di, dr]
  
#dirichlet distribution - separates the values so the sum is one
#alpha shows the partitions

alpha = [4,2,1]
x0 = np.random.dirichlet(alpha, size=100)
xt = np.asarray([integrate.odeint(SIRderiv, x0j, t) for x0j in x0])
#odeint - to integrate the SIR derivatives during each time period in dt in 2dt...

for i in range(100):
  for h in range(len(t)-1):
    row = i * (len(t)-1) + h
    nn_input[row] = xt[i][h]
    nn_output[row] = xt[i][h+1]

#so to compare current state with the previous one we map [S0,I0,R0] -> [S1,I1,R1]
#that's why nninput begins from h and the output from h+1 - so we calculated the next derivatives and compare them with the previous ones

print(nn_input)
print(nn_output)  
print(xt)


net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=3, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')
])

net.compile(
    loss='mse',
    optimizer='adam'
)
history = net.fit(nn_input, nn_output, epochs=100)
step = x0[0].reshape(1,3)
trajectories = []

#calculating the predictions
for i in range(400):
  step = net.predict(step)
  trajectories.append(step[0])

trajectories = np.array(trajectories)

#visualizing the trajectories

fig = plt.figure()
ax = fig.sub_plot(111,projection='3d') #111 - means 1 row, 1 col, 1 output
ax.plot(
    trajectories[:,0], #take all rows /:/ where column 0
    trajectories[:,1],
    trajectories[:,2]
)
ax.set_xlabel('S')
ax.set_ylabel('I')
ax.set_zlabel('R')
