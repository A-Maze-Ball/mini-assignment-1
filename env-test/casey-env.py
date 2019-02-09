import gym
import gym_abdul
import numpy as np

env = gym.make('abdul-v0')
state = env.reset()
best_weight = 0
best_reward = 0

for i in range(50):
    done = False
    weight = np.random.uniform(-10.0,10.0,1)
    reward = 0
    while not done:
        env.render()
        action = 1 if np.dot(weight,state) else 0
        state,reward, done, debug = env.step(action)

    if reward > best_reward
