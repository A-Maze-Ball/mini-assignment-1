import gym
import numpy as np
import math
from gym import wrappers

env = gym.make('BipedalWalker-v2')

best_weight = np.zeros(24)
best_reward = -100

def saveVideoOfBest(environment,weights):
    finished = False
    total_reward = 0
    environment = wrappers.Monitor(environment,'BipedalWalker-Best',force = True)
    state = environment.reset()
    while not finished:
        action = np.matmul(new_weight,state)
        state, reward, finished, debug = environment.step(action)
        total_reward += reward
        print('Reward:',total_reward,'Action:',action)
        if finished:
            break

for episode in range (100):
        state = env.reset()
        done = False
        new_weight = np.random.uniform(-6,6,size=(4,24))
        total_reward = 0
        for i in range (50):
                #env.render()
                action = np.matmul(new_weight,state)
                state, reward, done, debug = env.step(action)
                total_reward += reward
                if done:
                    break
        if total_reward > best_reward:
            best_reward = total_reward
            best_weight = new_weight
        print('Episode:',episode+1,'Reward:',total_reward,"Best Reward:",best_reward)
        if best_reward >= 300:
                print('Game = BEAT, Episodes Taken:',episode+1)
                break

saveVideoOfBest(env,env.action_space.sample())

env.close()