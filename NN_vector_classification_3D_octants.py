import numpy as np
import tensorflow as tf
#in which octant is located the vector
X = np.array([[3,5,2], [-2,4,6], [-3,-1,5], [4,-3,1],
     [2,1,-5], [-4,3,2],[-1,-2,-3], [3,-4,-6]])
test_data = np.array([[4,2,1], [-5,3,2], [-1,-4,7], [2,-3,9],
     [1,5,-2], [-3,1,-4],[-2,-3,-5], [5,-6,-3]])

X_better = np.random.uniform(-10, 10, (200, 3))
test_data_better =  np.random.uniform(-10, 10, (200, 3))

def giveOctant(point):
  x = point[0]
  y = point[1]
  z = point[2]

  if x > 0 and y > 0 and z > 0:
    return [1,0,0,0,0,0,0,0]
  elif x < 0 and y > 0 and z > 0:
    return [0,1,0,0,0,0,0,0] 
  elif x < 0 and y < 0 and z > 0:
    return [0,0,1,0,0,0,0,0] 
  elif x > 0 and y < 0 and z > 0:
    return [0,0,0,1,0,0,0,0] 
  elif x > 0 and y > 0 and z < 0:
    return [0,0,0,0,1,0,0,0] 
  elif x < 0 and y > 0 and z < 0:
    return [0,0,0,0,0,1,0,0] 
  elif x < 0 and y < 0 and z < 0:
    return [0,0,0,0,0,0,1,0] 
  else:
    return [0,0,0,0,0,0,0,1]
    
output_training = np.array([giveOctant(item) for item in X])
output_testing = np.array([giveOctant(item) for item in test_data])

output_training_better = np.array([giveOctant(item) for item in X_better])
output_testing_better = np.array([giveOctant(item) for item in test_data_better])

model = tf.keras.Sequential([
      tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
      tf.keras.layers.Dense(8, activation='softmax')
  ])

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )
model.fit(X_better,output_training_better,epochs=100)
predictions = model.predict(test_data_better)

for pred, real in zip(predictions, output_testing_better):
  idx = np.argmax(pred)
  if real[idx] == 1:
    print('accurate\n')
