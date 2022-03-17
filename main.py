#Import Package
import gym

#Import fonction 
from qLearningFunction.py import qLearning
from plotFunction.py import affichage

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


# Run Q-learning algorithm
rewards = qLearning(env, 0.2, 0.9, 0.8, 0, 5000)

#Run plot rewards 
affichae(rewards)


