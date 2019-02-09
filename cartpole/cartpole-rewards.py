import gym
import numpy as np
from gym import wrappers

environment = gym.make('CartPole-v0')
environment._max_episode_steps = 500

best_reward = 0
episode_rewards = []

best_weight = np.zeros(4)


def chooseAction(s,w):
    return 1 if np.dot(s, w) > 0 else 0

def saveVideoOfBest(environment):
    finished = False
    total_reward = 0
    environment = wrappers.Monitor(environment,'BestOne',force = True)
    state = environment.reset()
    while not finished:
        action = chooseAction(state, best_weight)
        print('Action:',action,'State:',state,'Weight:',best_weight)
        state, reward, finished, debug = environment.step(action)
        total_reward += reward
        if finished:
            print('Reward',total_reward)
            break

for i in range(100):
    new_weight = np.random.uniform(-1.0, 1.0, 4)

    rewards = []
    state = environment.reset()
    done = False
    total_reward = 0
    while not done:
        # environment.render()
        action = chooseAction(state, new_weight)
        state, reward, done, _ = environment.step(action)
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    average_reward = float(sum(rewards) / len(rewards))
    if average_reward > best_reward:
        best_reward = average_reward
        best_weight = new_weight

    episode_rewards.append(average_reward)
    print('Episode',i+1,'Average Reward:',average_reward,'\t\tBest Weight:',best_weight)

print('Best Reward:', best_reward)
saveVideoOfBest(environment)
environment.close()