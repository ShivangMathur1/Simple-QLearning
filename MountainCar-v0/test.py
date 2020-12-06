import gym
import numpy as np
from numpy.core.fromnumeric import argmax

# Make the environment
env = gym.make("MountainCar-v0")

discreteSize = [20] * len(env.observation_space.high)
discreteStep = (env.observation_space.high - env.observation_space.low) / discreteSize

# Load Table
qTable = np.load('qTable.npy', allow_pickle=True)

# Funtion to convert continuous states given by the environment to discrete ones
def getDiscreteState(state):
    discreteState = (state - env.observation_space.low) / discreteStep
    return tuple(discreteState.astype(np.int))

# Testing
state = getDiscreteState(env.reset())
done = False
netReward = 0
while not done:
    # Take an action, get rewards + states, and render           
    action = argmax(qTable[state])
    newState, reward, done, _ = env.step(action)
    newDiscreteState = getDiscreteState(newState)
    env.render()
    netReward += reward
    state = newDiscreteState
print(f"Reward {netReward}")
env.close()