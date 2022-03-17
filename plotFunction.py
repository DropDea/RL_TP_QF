import matplotlib.pyplot as plt

def affichage(rewards):
  
  plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
  plt.xlabel('Episodes')
  plt.ylabel('Average Reward')
  plt.title('Average Reward vs Episodes')
  plt.savefig('rewards.jpg')     
  plt.close() 
