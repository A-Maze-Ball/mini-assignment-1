import gym
import numpy as np
from gym import wrappers

environment = gym.make('CartPole-v0')
environment._max_episode_steps = 10000

best_steps = 0
episode_steps = []

best_weights = np.zeros(4)

def chooseAction(s,w):
    return 1 if np.dot(s, w) > 0 else 0

def saveVideoOfBest(environment):
    finished = False
    count = 0
    environment = wrappers.Monitor(environment,'CartPole-Best',force = True)
    state = environment.reset()
    while not finished:
        count += 1
        action = chooseAction(state, best_weights)
        print('Action:',action,'State:',state,'Weight:',best_weights)
        state, reward, finished, debug = environment.step(action)

        if finished:
            break

for i in range(1000):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    steps = []
    for j in range(1000):
        state = environment.reset()
        done = False
        step_counter = 0
    while not done:
        # environment.render()
        step_counter += 1
        action = chooseAction(state, new_weights)
        state, reward, done, _ = environment.step(action)

        if done:
            break

    steps.append(step_counter)
    average_steps = float(sum(steps) / len(steps))
    if average_steps > best_steps:
        best_steps = average_steps
        best_weights = new_weights
        if best_steps == 10000:
            episode_steps.append(average_steps)
            print('Episode',i+1,'Average # of Steps:',average_steps,'\t\tBest Weight:',best_weights)
            break
    episode_steps.append(average_steps)
    print('Episode',i+1,'Average # of Steps:',average_steps,'\t\tBest Weight:',best_weights)

saveVideoOfBest(environment)
environment.close()