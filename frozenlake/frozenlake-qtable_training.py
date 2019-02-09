import gym
import numpy as np
import random
import time
import os
import pickle
from IPython.display import clear_output

env = gym.make('FrozenLake-v0')


episode_rewards = []

state_size = env.observation_space.n
action_size = env.action_space.n
qtable = np.zeros((state_size,action_size),dtype=float)

num_episodes = 10000
max_steps = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

def saveData():
    f = open('frozenlake-data.pckl','wb')
    pickle.dump(exploration_rate,f)
    np.save('frozenlake-data.npy',qtable)
    f.close()

def loadData():
    global qtable, exploration_rate
    f = open('frozenlake-data.pckl','rb')
    exploration_rate = pickle.load(f)
    f.close()
    qtable = np.load('frozenlake-data.npy') if os.path.isfile('frozenlake-data.npy') else qtable

if os.path.isfile('frozenlake-data.pckl'): loadData()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    current_episode_reward = 0
    for step in range(max_steps):
        threshold = random.uniform(0,1)
        if threshold > exploration_rate:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        new_state, reward, done, debug = env.step(action)
        qtable[state, action] = qtable[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]))
        state = new_state
        current_episode_reward += reward
        if done == True:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    episode_rewards.append(current_episode_reward)

saveData()

# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(episode_rewards),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
env.close()