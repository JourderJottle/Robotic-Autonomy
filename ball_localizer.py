#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

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
        
        return math.exp(-ud.T @ self.invS @ ud / 2) / math.sqrt(math.pow(2*math.pi, 2) * self.detS)

def rotational_matrix(theta: float) -> np.matrix :
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], dtype=np.float64)

def gauss2D_from_polar(u_d: float, u_theta: float, S: np.matrix) -> Gauss2D :
    R = rotational_matrix(u_theta)
    S_v = R @ S @ R.T
    return Gauss2D(np.array([[u_d * math.cos(u_theta)], [u_d * math.sin(u_theta)]], dtype=np.float64), S_v)

def local_target_pose_to_global(target_pose: np.ndarray, sensor_translation: np.ndarray, sensor_theta: float, robot_pose: np.ndarray, robot_theta: float) -> np.ndarray :

    R_R = rotational_matrix(sensor_theta)
    R_G = rotational_matrix(robot_theta)
    return R_G @ (R_R @ target_pose + sensor_translation) + robot_pose
    
    
    




class BallLocalizer :
    def __init__(self) :
        rospy.init_node("ball_localizer", anonymous=True)
        rospy.Subscriber("/ball_data", Float32MultiArray, self.callback)
        
        # checked via tape measurer
        self.observable_distance = 3048
        # checked via moving ball towards camera to find minimum computed distance
        self.minimum_observable_distance = 200
        # checked via moving ball to edge of camera FOV to check angle
        self.observable_angle = 0.63
        self.scale = 1 / 5
        self.frame_height = int(self.observable_distance * self.scale)
        self.frame_width = int(self.observable_distance * math.sin(self.observable_angle) * 2 * self.scale)
        
        rospy.spin()

    def variance_from_distance(self, distance) :
        # Scale with distance example?
        #sigma_d = 0.01 * distance
        #sigma_theta = 0.001
        #return np.matrix([[sigma_d**2, 0], [0, sigma_theta**2]])
        
        # or...
        return np.matrix([[50, 0,], [0, 1]])
        
    def callback(self, data) :
        """Data 0 is distance, Data 1 is theta"""
        
        distance = data.data[0]
        theta = data.data[1]
        rospy.loginfo(f'{distance} {theta}')
        
        dist = gauss2D_from_polar(distance, theta, self.variance_from_distance(distance))
        
        u = (int(dist.u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(dist.u[0][0] * self.scale))
        a = dist.S[0, 0]
        b = dist.S[0, 1]
        c = dist.S[1, 1]
        l1 = (a + c) / 2 + math.sqrt(((a - c) / 2)**2 + b**2)
        l2 = (a + c) / 2 - math.sqrt(((a - c) / 2)**2 + b**2)
        angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

        display_frame = np.zeros(shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.observable_distance * self.scale), int(self.observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (150, 0, 30), -1)
        cv.ellipse(display_frame, u, (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (255, 255, 255), -1)
        
        cv.imshow('Space', display_frame)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")
        


if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
