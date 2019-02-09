import gym
import numpy as np
import time
import os
env = gym.make('FrozenLake-v0')
qtable = np.load('frozenlake-data.npy')
max_steps_per_episode = 100
for episode in range(5):
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):        
        os.system('clear')
        env.render()
        time.sleep(0.3)
        action = np.argmax(qtable[state,:])        
        new_state, reward, done, info = env.step(action)
        if done:
            os.system('clear')
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                os.system('clear')
            break
        state = new_state

env.close()