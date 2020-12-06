import numpy as np
import gym
import time

# Generate Environment
env = gym.make("Taxi-v3")

rewards = []

# Load Q-Table
actionSize = env.action_space.n
stateSize = env.observation_space.n
qTable = np.zeros((stateSize, actionSize))

f = open("qtable.txt", "r")
for i in range(stateSize):
    for j in range(actionSize):
        qTable[i][j] = float(f.readline())

f.close()

for episode in range(1):
    state = env.reset()
    step = 0
    done = False

    totalRewards = 0
    print("******************************\nEpisode: ", episode)

    for step in range(99):
        env.render()
        action = np.argmax(qTable[state][:])
        newState, reward, done, info = env.step(action)
        totalRewards += reward

        if done:
            rewards.append(totalRewards)
            print("Score: ", totalRewards)
            break
        state = newState
        time.sleep(0.3)
    time.sleep(1)
env.close()
print("Score over time: ", str(sum(rewards)/1))
print(stateSize)