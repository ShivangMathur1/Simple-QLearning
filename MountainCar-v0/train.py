import gym
import numpy as np
from numpy.core.fromnumeric import argmax

# Make the environment
env = gym.make("MountainCar-v0")

# Size for our observations = [40, 40], since we divide the two ranges between highest
# and lowest observations values into discrete buckets of 20
discreteSize = [20] * len(env.observation_space.high)
discreteStep = (env.observation_space.high - env.observation_space.low) / discreteSize

# qTable of shape (20, 20, 3)
qTable = np.random.uniform(low=-2, high=0, size=(discreteSize + [env.action_space.n]))


# Hyperparameters
lr = 0.1
gamma = 0.95
episodes = 25000
displayFrequency = 2000
epsilon = 0.5
startDecay = 1
endDecay = episodes // 2
decayValue = epsilon / (endDecay - startDecay)

# Funtion to convert continuous states given by the environment to discrete ones
def getDiscreteState(state):
    discreteState = (state - env.observation_space.low) / discreteStep
    return tuple(discreteState.astype(np.int))

# Training starts
for episode in range(episodes):
    state = getDiscreteState(env.reset())
    # Whether or not to render the episode
    if episode % displayFrequency == 0:
        print(episode)
        render = True
    else:
        render = False
    done = False
    while not done:
        # Take an action(Exploration vs Exploitation), get rewards + states, and render
        if np.random.random() > epsilon:            
            action = argmax(qTable[state])
        else:
            action = np.random.randint(0, env.action_space.n)
        newState, reward, done, _ = env.step(action)
        newDiscreteState = getDiscreteState(newState)
        if render:
            env.render()

        # Update Q-values in the Q-table
        if not done:
            nextQmax = np.max(qTable[newDiscreteState])
            currentQmax = qTable[state + (action, )]
            newQ = (1 - lr) * currentQmax + lr * (reward + gamma * nextQmax)        # Bellman equation
            qTable[state + (action, )] = newQ
        # If goal is reached, assign best Q-value: 0 to the state
        elif newState[0] >= env.goal_position:
            print("Episode of victory: ", episode)
            qTable[state + (action, )] = 0
        
        state = newDiscreteState
    if startDecay <= episode <= endDecay:
        epsilon -= decayValue

env.close()
np.save("qTable.npy", qTable, allow_pickle=True)