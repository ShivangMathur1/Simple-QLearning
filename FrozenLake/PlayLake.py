import numpy as np
import gym
import time

# Make Environment and Q-Table
env = gym.make("FrozenLake-v0")

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
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(99):
        env.render()
        time.sleep(0.3)
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qTable[state][:])
        print(state, action)

        new_state, reward, done, info = env.step(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            time.sleep(0.3)
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()