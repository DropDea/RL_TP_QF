#Import Package
import gym

#Import fonction 
from QLearningFunction.py import QLearning
from PlotFunction.py import Plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000)

#Run plot rewards 
Plt(rewards)


