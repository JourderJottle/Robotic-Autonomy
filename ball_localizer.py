#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
from collections import deque

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

def motion_model(u) :
    return u

def sensor_model(x) :
    return x

def derive_gradient(func, location, dl) :
    dimensions = len(location)
    j1 = []
    j2 = []
    for i in range(dimensions) :
        dx = np.zeros(dimensions)
        dx[i] = dl / 2
        x1 = location - dx
        x2 = location + dx
        j1.append(func(x1))
        j2.append(func(x2))
    return (np.array(j2) - np.array(j1)) / dl

def ekf_predict(previous_state, previous_covariance, input, motion_model, motion_noise, dt) :
    A = derive_gradient(motion_model, input, 0.1)
    predicted_mean = previous_state + dt * motion_model(input)
    a1 = A @ previous_covariance
    a2 = a1 @ A.T
    predicted_covariance = a2 + motion_noise
    # Returns (mean, covariance)
    return (predicted_mean, predicted_covariance)


def ekf_correct(predicted_state, predicted_covariance, observation, sensor_model, sensor_noise) :
    C = derive_gradient(sensor_model, predicted_state, 0.1)
    K = predicted_covariance @ C.T @ np.linalg.pinv(C @ predicted_covariance @ C.T + sensor_noise)
    
    # Returns (mean, covariance)
    return (predicted_state + K @ (observation - sensor_model(predicted_state)), (np.eye(len(predicted_state)) - K @ C) @ predicted_covariance)

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
        self.distance_noise = 0.05
        self.angle_noise = 0.01

        self.angle_total = 0
        self.distance_total = 0
        self.observation_queue = deque()
        self.queue_size = 10

        self.motion_control = [1, 0]
        self.motion_noise = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        self.last_dist = Gauss2D(np.array([0.0, 0.0]), np.matrix([[0.0, 0.0], [0.0, 0.0]]))
        self.last_time = rospy.get_rostime().secs

        rospy.loginfo("Starting ball localizer...")

        rospy.spin()

    def variance(self) :
        u_d = self.distance_total / self.queue_size
        u_theta = self.angle_total / self.queue_size
        v_d = 0
        v_theta = 0
        for o in self.observation_queue :
            v_d += (o[0] - u_d)**2
            v_theta += (o[1] - u_theta)**2
        S_d = math.sqrt(v_d / self.queue_size)
        S_theta = math.sqrt(v_theta / self.queue_size)
        return np.matrix([[S_d + u_d * self.distance_noise, 0], [0, S_theta + u_d * self.angle_noise]])
        
    def callback(self, data) :
        """Data 0 is distance, Data 1 is theta"""
        display_frame = np.zeros(shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.observable_distance * self.scale), int(self.observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (150, 0, 30), -1)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.minimum_observable_distance * self.scale), int(self.minimum_observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (0, 0, 0), -1)

        distance = 0
        theta = 0

        time = rospy.get_rostime().secs
        dt = time - self.last_time
        self.last_time = time

        if data.data != None and len(data.data) > 0 :
            init_distance = data.data[0]
            init_theta = data.data[1]
            distance = init_distance
            theta = init_theta
            self.distance_total += init_distance
            self.angle_total += init_theta
            self.observation_queue.append((init_distance, init_theta))
            if len(self.observation_queue) >= self.queue_size :
                self.distance_total -= self.observation_queue[0][0]
                self.angle_total -= self.observation_queue[0][1]
                self.observation_queue.popleft()
                distance = self.distance_total / self.queue_size
                theta = self.angle_total / self.queue_size

        (predicted_mean, predicted_covariance) = ekf_predict(self.last_dist.u, self.last_dist.S, self.motion_control, motion_model, self.motion_noise, dt)

        if distance > self.minimum_observable_distance and distance < self.observable_distance and abs(theta) < self.observable_angle :

            dist = gauss2D_from_polar(distance, theta, self.variance())
            (corrected_mean, corrected_covariance) = ekf_correct(predicted_mean, predicted_covariance, dist.u, sensor_model, dist.S)
            self.last_dist = Gauss2D(corrected_mean, corrected_covariance)

        else :

            self.last_dist = Gauss2D(predicted_mean, predicted_covariance)
               
        u = (int(self.last_dist.u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(self.last_dist.u[0][0] * self.scale))
        a = self.last_dist.S[0, 0]
        b = self.last_dist.S[0, 1]
        c = self.last_dist.S[1, 1]
                
        l1 = (a + c) / 2 + math.sqrt(((a - c) / 2)**2 + b**2)
        l2 = (a + c) / 2 - math.sqrt(((a - c) / 2)**2 + b**2)

        angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

        cv.ellipse(display_frame, u, (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (255, 255, 255), -1)

        cv.imshow('Space', display_frame)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")

if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
