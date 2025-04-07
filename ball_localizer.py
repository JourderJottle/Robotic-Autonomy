#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
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
    """
    rospy.loginfo(f'options of input {input[0]} and {input[1]}')
    rospy.loginfo(f"Input: {motion_model(input)}")
    rospy.loginfo(f"A: {A}")
    rospy.loginfo(f'dt is {dt}')
    """
    mult = dt * np.matrix(motion_model(input)).T
    # rospy.loginfo(f"Mult: {mult}")    
    predicted_mean = previous_state + mult
    a1 = A @ previous_covariance
    a2 = a1 @ A.T
    predicted_covariance = a2 + motion_noise
    # Returns (mean, covariance)
    return (predicted_mean, predicted_covariance)


def ekf_correct(predicted_state, predicted_covariance, observation, sensor_model, sensor_noise) :
    C = derive_gradient(sensor_model, np.array(predicted_state.T)[0], 0.1)
    # rospy.loginfo(f"C: {C}")
    # rospy.loginfo(f"Pred State T: {np.array(predicted_state.T)[0]}")
    K = predicted_covariance @ C.T @ np.linalg.pinv(C @ predicted_covariance @ C.T + sensor_noise)
    
    # Returns (mean, covariance)
    return (predicted_state + K @ (observation - sensor_model(predicted_state)), (np.eye(len(predicted_state)) - K @ C) @ predicted_covariance)

class BallLocalizer :
    def __init__(self) :
        rospy.init_node("ball_localizer", anonymous=True)
        rospy.Subscriber("/ball_data", Float32MultiArray, self.callback)
        self.publisher = rospy.Publisher("/ball_variance_ellipse", Marker, queue_size=10)
        
        # checked via tape measurer
        self.observable_distance = 2400 # extended via smaller radius minimum
        # it to that results in OOB still being in the blue region, idk
        
        # checked via moving ball towards camera to find minimum computed distance
        self.minimum_observable_distance = 155
        # checked via moving ball to edge of camera FOV to check angle
        self.observable_angle = 0.48 # OLD: 0.63
        self.scale = 1 / 5
        self.frame_height = int(self.observable_distance * self.scale)
        self.frame_width = int(self.observable_distance * math.sin(self.observable_angle) * 2 * self.scale)
        self.distance_noise = 0.05
        self.angle_noise = 0.01

        self.angle_total = 0
        self.distance_total = 0
        self.observation_queue = deque()
        self.queue_size = 10
        self.queue_size_ex0 = 0

        # Index 0: (+) is away from camera, (-) is towards camera
        # Index 1: (+) is to the right, (-) is to the left
        # y=200 seems optimal for midterm presentation
        self.motion_control = np.array([0, 200])
        
        # noise in x ... noise in y
        self.motion_noise = np.matrix([[10, 0.0], [0.0, 10]])
        
        self.last_dist = Gauss2D(np.array([[0.0], [0.0]]), np.matrix([[0.0, 0.0], [0.0, 0.0]]))
        self.last_time = rospy.get_rostime().secs

        self.draw_observation = False
        self.draw_estimation = True

        rospy.loginfo("Starting ball localizer...")

        rospy.spin()

    def variance(self) :
        if self.queue_size_ex0 > 0 :
            u_d = self.distance_total / self.queue_size_ex0
            u_theta = self.angle_total / self.queue_size_ex0
            v_d = 0
            v_theta = 0
            for o in self.observation_queue :
                v_d += (o[0] - u_d)**2
                v_theta += (o[1] - u_theta)**2
            S_d = math.sqrt(v_d / self.queue_size_ex0)
            S_theta = math.sqrt(v_theta / self.queue_size_ex0)
            return np.matrix([[S_d + u_d * self.distance_noise, 0], [0, S_theta + u_d * self.angle_noise]])
        else :
            return np.matrix([[0, 0], [0, 0]])
        
    def callback(self, data) :
        """Data 0 is distance, Data 1 is theta"""
        display_frame = np.zeros(shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.observable_distance * self.scale), int(self.observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (150, 0, 30), -1)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.minimum_observable_distance * self.scale), int(self.minimum_observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (0, 0, 0), -1)

        distance = 0
        theta = 0

        time = rospy.get_rostime().to_sec()
        dt = time - self.last_time
        self.last_time = time
        
        
        # Check more precisely that distance != 0
        if data.data != None and len(data.data) > 0 :
            init_distance = data.data[0]
            init_theta = data.data[1]
            distance = init_distance
            theta = init_theta
            self.distance_total += init_distance
            self.angle_total += init_theta
            self.observation_queue.append((init_distance, init_theta))
            if init_distance > 0 : self.queue_size_ex0 += 1
            if len(self.observation_queue) >= self.queue_size :
                self.distance_total -= self.observation_queue[0][0]
                self.angle_total -= self.observation_queue[0][1]
                old_observation = self.observation_queue.popleft()
                if old_observation[0] > 0 : self.queue_size_ex0 -= 1
            if self.queue_size_ex0 > 5 :
                distance = self.distance_total / self.queue_size_ex0
                theta = self.angle_total / self.queue_size_ex0
        else:
            rospy.loginfo("Ball not detected")

        (predicted_mean, predicted_covariance) = ekf_predict(self.last_dist.u, self.last_dist.S, self.motion_control, motion_model, self.motion_noise, dt)
        #rospy.loginfo(f"Predicted State: {predicted_mean}")

        if distance > self.minimum_observable_distance and distance < self.observable_distance and abs(theta) < self.observable_angle :

            dist = gauss2D_from_polar(distance, theta, self.variance())
            if self.draw_observation :
                u = (int(dist.u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(dist.u[0][0] * self.scale))
                a = dist.S[0, 0]
                b = dist.S[0, 1]
                c = dist.S[1, 1]

                l1 = (a + c) / 2 + math.sqrt(((a - c) / 2)**2 + b**2)
                l2 = (a + c) / 2 - math.sqrt(((a - c) / 2)**2 + b**2)

                angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

                cv.ellipse(display_frame, u, (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (0, 0, 255), -1)

            (corrected_mean, corrected_covariance) = ekf_correct(predicted_mean, predicted_covariance, dist.u, sensor_model, dist.S)
            self.last_dist = Gauss2D(corrected_mean, corrected_covariance)

        else :

            self.last_dist = Gauss2D(predicted_mean, predicted_covariance)
        
        # rospy.loginfo(f"u: {self.last_dist.u}")
        # rospy.loginfo(f"S: {self.last_dist.S}")
        u = (int(self.last_dist.u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(self.last_dist.u[0][0] * self.scale))
        a = self.last_dist.S[0, 0]
        b = self.last_dist.S[0, 1]
        c = self.last_dist.S[1, 1]
                  
        l1 = (a + c) / 2 + math.sqrt(((a - c) / 2)**2 + b**2)
        l2 = (a + c) / 2 - math.sqrt(((a - c) / 2)**2 + b**2)

        angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

        if self.draw_estimation :
            cv.ellipse(display_frame, u, (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (255, 255, 255), -1)
        
        marker = Marker()
        marker.header.frame_id = "map" # don't know why, all the example code i found does this
        marker.header.stamp = rospy.Time.now()
        # type for cylinder since we don't have z variance
        marker.type = 3
        # scale is allegedly in meters so z is 1 to make it tall enough to see and the / 1000 is because these are in millimeters
        # we should play with these if that seems wrong in practice
        marker.scale.x = self.last_dist.S[0, 0] / 1000
        marker.scale.y = self.last_dist.S[1, 1] / 1000
        marker.scale.z = 1
        # this should be clear, 0.5 is because i think scale is on both sides of the center
        marker.pose.position.x = self.last_dist.u[0][0]
        marker.pose.position.y = self.last_dist.u[1][0]
        marker.pose.position.z = 0.5
        # i think orientation rotates the object around the specified axis, so it should rotate around the z axis to affect yaw
        # of course, i don't know that for certain so definitely could be a point of failure
        # w is a quaternion thing, i don't know what it does but example code all makes it 1
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = angle
        marker.pose.orientation.w = 1
        # rgba on a 0-1 scale, so i think this should be white with 100% opacity
        marker.color.r = 1
        marker.color.g = 1
        marker.color.b = 1
        marker.color.a = 1

        self.publisher.publish(marker)

        cv.imshow('Space', display_frame)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")

if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
