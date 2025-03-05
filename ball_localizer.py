#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from robot_math import *

class BallLocalizer :
    def __init__(self) :
        rospy.init_node("ball_localizer", anonymous=True)
        rospy.Subscriber("/ball_data", Float32MultiArray, self.callback)
        self.fig = plt.figure()
        rospy.spin()

    def variance_from_distance(self, distance) :
        return np.matrix([[0, 0,], [0, 0]])
        
    def callback(self, data) :
        dist = gauss2D_from_polar(data[0], data[1], self.variance_from_distance(data[0]))

        x = np.linspace(-500, 500)
        y = np.linspace(0, 1000)
        X, Y = np.meshgrid(x,y)
        
        # Generating the density function
        # for each point in the meshgrid
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = dist.probability(np.matrix([[X[i,j]], [Y[i,j]]]))
        
        # Plotting the density function values
        ax = self.fig.add_subplot(131, projection = '3d')
        ax.plot_surface(X, Y, pdf, cmap = 'viridis')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f'Distribution')
        ax.axes.zaxis.set_ticks([])
        plt.show()

if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")