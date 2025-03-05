#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2 as cv

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import open3d as o3d



class Gauss2D:

    def __init__(self, u: np.matrix, S: np.matrix) :
        self.u = u
        self.S = S
        
        # add epsilon so no det=0 error
        epsilon = 1e-6
        self.S += np.eye(self.S.shape[0]) * epsilon
        
        self.invS = np.linalg.inv(self.S)
        self.detS = np.linalg.det(self.S)

    def probability(self, x: np.matrix) -> float :
        ud = x - self.u
        
        exponent = -0.5 * (ud.T @ self.invS @ ud).item()
        
        exponent = np.clip(exponent, -700, 700)
        
        return math.exp(exponent) / math.sqrt( (2*math.pi)**2 * self.detS)
        
        # return math.exp(-ud.T * self.invS * ud / 2) / math.sqrt(math.pow(2*math.pi, 2) * self.detS)

def rotational_matrix(theta: float) -> np.matrix :
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], dtype=np.float64)

def gauss2D_from_polar(u_d: float, u_theta: float, S: np.matrix) -> Gauss2D :
    R = rotational_matrix(u_theta)
    S_v = R @ S @ R.T
    return Gauss2D(np.array([[u_d * math.cos(u_theta)], [u_d * math.sin(u_theta)]], dtype=np.float64), S_v)

def local_target_pose_to_global(target_pose: np.ndarray, robot_pose: np.ndarray, robot_theta: float) -> np.ndarray :
    #global_target_pose = target_pose + robot_pose
    #R = rotational_matrix(robot_theta)
    #return R * global_target_pose
    
    # Must rotate before adding
    R = rotational_matrix(robot_theta)
    return R @ target_pose + robot_pose
    
    
    




class BallLocalizer :
    def __init__(self) :
        rospy.init_node("ball_localizer", anonymous=True)
        rospy.Subscriber("/ball_data", Float32MultiArray, self.callback)
        #self.fig, self.ax = plt.subplots(subplot_kw={"projection":"3d"})
        
      
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        
        self.cloud_pub = rospy.Publisher('/ball_cloud', PointCloud2, queue_size=10)
        
        rospy.spin()

    def variance_from_distance(self, distance) :
        # Scale with distance example?
        sigma_d = 0.01 * distance
        sigma_theta = 0.001
        return np.matrix([[sigma_d**2, 0], [0, sigma_theta**2]])
        
        # or...
        # return np.matrix([[1, 0,], [0, 1]])
        
    def callback(self, data) :
        """Data 0 is distance, Data 1 is theta"""
        
        distance = data.data[0]
        theta = data.data[1]
        
        
        dist = gauss2D_from_polar(distance, theta, self.variance_from_distance(distance))

        x = np.linspace(-500, 500)
        y = np.linspace(0, 1000)
        X, Y = np.meshgrid(x,y)
        
        # Generating the density function
        # for each point in the meshgrid
        
        points = []
        pdf = np.zeros(X.shape)
        
        for i in range(X.shape[0]):
        	for j in range(X.shape[1]):
        		prob = dist.probability(np.array([[X[i, j]], [Y[i, j]]], dtype=np.float64))
        		p = [ X[i, j], Y[i, j], prob ]
        		points.append(p)
        
       	header = rospy.Header()
       	header.stamp = rospy.Time.now()
       	header.frame_id = "map"
       	
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        cloud_msg = pc2.create_cloud(header, fields, points)
        cloud_msg.width = len(points)
        cloud_msg.height = 1
        cloud_msg.is_dense = False
        self.cloud_pub.publish(cloud_msg)
        rospy.loginfo("Published ball cloud")
        
        
        # Plotting the density function values
        #ax = self.fig.add_subplot(111, projection = '3d')
        #ax.plot_surface(X, Y, pdf, cmap = 'viridis')
        #plt.xlabel("x1")
        #plt.ylabel("x2")
        #plt.title(f'Distribution')
        #ax.axes.zaxis.set_ticks([])
        #plt.show()

        


if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
