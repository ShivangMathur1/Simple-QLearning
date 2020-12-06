import numpy as np
import gym
import random

# Generate Environment
env = gym.make("Taxi-v3")

# Define Q-Table with initial values as zeroes
actionSize = env.action_space.n
stateSize = env.observation_space.n
qTable = np.zeros((stateSize, actionSize))
print(qTable)

# Hyper Parameters
totalEpisodes = 50000
maxSteps = 99

learningRate = 0.7
gamma = 0.618

# Exploration parameters
epsilon = 1
minEpsilon = 0.01
maxEpsilon = 1
decayRate = 0.01

# Q-Learning Algorithm
for episode in range(totalEpisodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(maxSteps):
        exp = random.uniform(0, 1)

        # Exploitation vs Exploration
        if exp > epsilon:
            action = np.argmax(qTable[state][:])
        else:
            action = env.action_space.sample()

        # Take Action
        newState, reward, done, info = env.step(action)

        # Update Q-Table
        qTable[state][action] = qTable[state][action] + learningRate * (
                    reward + gamma * np.max(qTable[newState][:] - qTable[state][action]))

        state = newState
        if done:
            break

    # Update epsilon
    epsilon = minEpsilon + (maxEpsilon - minEpsilon) * np.exp(-decayRate * episode)

f = open("qtable.txt", "w")

for i in range(stateSize):
    for j in range(actionSize):
        f.write(str(qTable[i][j]))
        f.write("\n")
print(qTable)
f.close()
