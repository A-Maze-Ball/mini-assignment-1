import gym
import numpy as np
import math
from gym import wrappers

env = gym.make('MountainCar-v0')

best_weight = np.zeros(2)
best_position = -1.2

def sigmoid(x):
        return (1 / (1 + math.exp(-x)))

def choose_action(a,b):
        sig = sigmoid(np.dot(a,b))
        if sig > .666:
                return 0
        elif sig > .333:
                return 1
        else:
                return 2

def saveVideoOfBest(environment,weights):
    finished = False
    position=-1.2
    environment = wrappers.Monitor(environment,'MountainCar-Best',force = True)
    state = environment.reset()
    while not finished:
        action = choose_action(state, weights)
        print('Action:',action,'State:',state,'Weight:',best_weight)
        state, reward, finished, debug = environment.step(action)
        
        if state[0] > position:
                position = state[0]
        if finished:
            print('Farthest position',position)
            break

for episode in range (1000):
        state = env.reset()
        done = False
        new_weight = np.random.uniform(-100,100,2)
        position = state[0]
        for i in range (env._max_episode_steps):
                #env.render()
                action = choose_action(state,new_weight)
                state, reward, done, debug = env.step(action)
                if state[0] > position:
                        position=state[0]
                if done:
                        if position > best_position:
                                best_position = position
                                best_weight = new_weight
                        break
        print('Episode:',episode+1,'Farthest Position:',best_position,"Best Weight:",best_weight)
        if position >= 0.5:
                print('Game = BEAT, Episodes Taken:',episode+1)
                break

#print('Farthest Position:',best_position,'Best Weight:',best_weight)

saveVideoOfBest(env,best_weight)

env.close()
