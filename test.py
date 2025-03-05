import numpy as np
import matplotlib.pyplot as plt
from robot_math import *

dist = gauss2D_from_polar()

sigma_1, sigma_2 = dist.S[0,0], dist.S[1,1]

x = np.linspace(-3*sigma_1, 3*sigma_1, num=100)
y = np.linspace(-3*sigma_2, 3*sigma_2, num=100)
X, Y = np.meshgrid(x,y)
        
# Generating the density function
# for each point in the meshgrid
pdf = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf[i,j] = dist.probability([X[i,j], Y[i,j]])
        
        # Plotting the density function values
        ax = self.fig.add_subplot(0, projection = '3d')
        ax.plot_surface(X, Y, pdf, cmap = 'viridis')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f'Distribution')
        ax.axes.zaxis.set_ticks([])
        plt.show()