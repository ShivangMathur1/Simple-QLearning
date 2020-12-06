import numpy as np
import gym
import random

# Make Environment and Q-Table
env = gym.make("FrozenLake-v0")

actionSize = env.action_space.n
stateSize = env.observation_space.n

qTable = np.zeros((stateSize, actionSize))

# Hyper Parameters
totalEpisodes = 2000
learningRate = 0.70
maxSteps = 99
gamma = 0.90

# Exploration Parameters
epsilon = 1
maxEpsilon = 1
minEpsilon = 0.01
decayRate = 0.01

# Learning
rewards = []

for episode in range(totalEpisodes):
    state = env.reset()
    done = False
    totalRewards = 0

    for step in range(maxSteps):
        # Exploitation Vs Exploration
        exp = random.uniform(0, 1)
        if exp > epsilon:
            action = np.argmax(qTable[state][:])
        else:
            action = env.action_space.sample()

        # Take Action
        newState, reward, done, info = env.step(action)
        if reward > 0:
            reward += 19
        else:
            reward -= 1
            if done:
                reward -= 9
        print(reward)
        qTable[state][action] = qTable[state][action] + learningRate * (reward + gamma * np.max(qTable[newState][:]) - qTable[state][action])

        totalRewards += reward
        state = newState

        if done:
            break

    epsilon = minEpsilon + (maxEpsilon - minEpsilon) * np.exp(-decayRate * episode)
    rewards.append(totalRewards)

print("Final Score : ", str(sum(rewards)/totalEpisodes))

f = open("qtable.txt", "w")

for i in range(stateSize):
    for j in range(actionSize):
        f.write(str(qTable[i][j]))
        f.write("\n")
f.close()
print(qTable)
