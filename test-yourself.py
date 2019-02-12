from keras.models import Sequential
from keras.layers import Flatten, Dense
from collections import deque
from keras.models import model_from_json
import os
import random
import numpy as np
import gym



all_envs = gym.envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
a=0
'''
for i in env_ids:
    print(str(a)+" - "+str(i))
    a+=1
'''
selected=494
env = gym.make(env_ids[selected])
print(" Training on : "+str(env_ids[selected]))

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
if not (os.path.exists('./model_weights.h5')):
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
else:
    print("Loading model")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_weights.h5")
    print("Loaded model from disk")
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Parameters
                              # Register where the actions will be stored

observetime = 50                        # Number of timesteps we will be acting on the game and observing results
epsilon = 0.9                            # Probability of doing a random move
gamma = 0.9                              # Discounted future reward. How much we care about steps further in time
mb_size = 50                              # Learning minibatch size
steps=1000
for prog in range(0,steps):
  print("Step ("+str(prog)+"/"+str(steps)+")")
  # Beginning of the game
  observation = env.reset()
  D = deque()  
  obs = np.expand_dims(observation, axis=0)  
  state = np.stack((obs, obs), axis=1)
  done = False
  for i in range(observetime):
      if (np.random.rand()<= epsilon): # Random factor
          action = np.random.randint(0, env.action_space.n, size=1)[0] # Random move
      else:
          Q = model.predict(state) # Q-based move, which is predicted by the modal
          action = np.argmax(Q) 
      observation_new, reward, done, info = env.step(action) # Do the action and get the reward, state..ect.
      obs_new = np.expand_dims(observation_new, axis=0)
      state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1) 
      D.append((state, action, reward, state_new, done)) 
      state = state_new 
      if done:
          env.reset()           # Restart game at the end
          obs = np.expand_dims(observation, axis=0)     
          state = np.stack((obs, obs), axis=1)




  minibatch = random.sample(D, mb_size)                              
  inputs_shape = (mb_size,) + state.shape[1:]
  inputs = np.zeros(inputs_shape)
  targets = np.zeros((mb_size, env.action_space.n))



  for i in range (0, mb_size):
      state = minibatch[i][0]
      action = minibatch[i][1]
      reward = minibatch[i][2]
      state_new = minibatch[i][3]
      done = minibatch[i][4]

      inputs[i:i+1] = np.expand_dims(state, axis=0)
      targets[i] = model.predict(state)
      Q_sa = model.predict(state_new)

      if done:
          targets[i, action] = reward
      else:
          targets[i, action] = reward + gamma * np.max(Q_sa)
      model.train_on_batch(inputs, targets)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")
print("End of the game")
