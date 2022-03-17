import numpy as np
import gym
import matplotlib.pyplot as plt


# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000)
