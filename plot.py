import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./log.csv')

data['plot_x'] = data['epoch'].astype(str) + '_' + data['iteration'].astype(str)

plt.figure(figsize=(12,6))
plt.plot(data['plot_x'], data['discriminator_loss'], label='discriminator_loss')
plt.plot(data['plot_x'], data['generator_loss'], label='generator_loss')
plt.xticks(np.arange(0, len(data)+1, 10), rotation=-90)
plt.legend()
plt.savefig('plot.png')