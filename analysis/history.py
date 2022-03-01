import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join("..","EM_shower_simulator","model_results","history.npy")
hist = np.load(path, allow_pickle=True)
hist = hist.item() # necessary because numpy create a structured dictionary

fig = plt.figure("Evolution of losses per epochs")
ax = plt.subplot(111)
for key in hist:
    plt.plot(hist[key], label=key)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yscale('log')
plt.show()
fig.savefig("losses_evolution_70-140.png")
