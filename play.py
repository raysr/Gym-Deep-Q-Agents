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
selected=494
env = gym.make(env_ids[selected])

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.

model = Sequential()
model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
'''
    print("Loading model")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
'''
model.load_weights("model_weights.h5")
print("Loaded model from disk")
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Beginning of the game
print("Play time")
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()                    
    Q = model.predict(state)        
    action = np.argmax(Q)         
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    tot_reward += reward
