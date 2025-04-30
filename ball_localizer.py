#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseWithCovariance
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
from collections import deque

class Gauss2D:

    def __init__(self, u: np.array, S: np.array) :
        self.u = u
        self.S = S
        
        # add epsilon so no det=0 error
        epsilon = 1e-6
        self.S += np.eye(self.S.shape[0]) * epsilon
        
        self.invS = np.linalg.inv(self.S)
        self.detS = np.linalg.det(self.S)

    def probability(self, x: np.array) -> float :
        ud = x - self.u
        return math.exp(-ud.T @ self.invS @ ud / 2) / math.sqrt(math.pow(2*math.pi, 2) * self.detS)

def rotational_matrix_2D(theta: float) -> np.array :
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], dtype=np.float64)

def gauss2D_from_polar(u_d: float, u_theta: float, S: np.array) -> Gauss2D :
    R = rotational_matrix_2D(u_theta)
    S_v = R @ S @ R.T
    return Gauss2D(np.array([[u_d * math.cos(u_theta)], [u_d * math.sin(u_theta)]], dtype=np.float64), S_v)

def local_target_pose_to_global(target_pose: np.array, sensor_translation: np.array, sensor_theta: float, robot_pose: np.array, robot_theta: float) -> np.array :
    R_R = rotational_matrix_2D(sensor_theta)
    R_G = rotational_matrix_2D(robot_theta)
    return R_G @ (R_R @ target_pose + sensor_translation) + robot_pose

def ellipse_from_gauss2D(dist) :
    a = dist.S[0, 0]
    b = dist.S[0, 1]
    c = dist.S[1, 1]

    l1 = (a + c) / 2 + math.sqrt(((a - c) / 2)**2 + b**2)
    l2 = (a + c) / 2 - math.sqrt(((a - c) / 2)**2 + b**2)

    angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

    return dist.u, l1, l2, angle

def derive_gradient(func, location, dl) :
    dimensions = location.shape[0]
    j1 = []
    j2 = []
    for i in range(dimensions) :
        dx = np.vstack(np.zeros(dimensions))
        dx[i] = dl / 2
        x1 = location - dx
        x2 = location + dx
        j1.append(func(x1).flatten())
        j2.append(func(x2).flatten())
    return np.matrix((np.array(j2) - np.array(j1)) / dl)

def ekf_predict(previous_state, previous_covariance, input, motion_model, motion_noise, dt) :
    A = derive_gradient(motion_model, input, 0.1)
    # Returns (mean, covariance)
    return (previous_state + dt * motion_model(input), A @ previous_covariance @ A.T + motion_noise)

def ekf_correct(predicted_state, predicted_covariance, observation, sensor_model, sensor_noise) :
    C = derive_gradient(sensor_model, predicted_state, 0.1)
    K = predicted_covariance @ C.T @ np.linalg.pinv(C @ predicted_covariance @ C.T + sensor_noise)
    # Returns (mean, covariance)
    return (predicted_state + K @ (observation - predicted_state), (np.eye(len(predicted_state)) - K @ C) @ predicted_covariance)

class BallLocalizer :
    def __init__(self) :
        rospy.init_node("ball_localizer", anonymous=True)
        rospy.Subscriber("/ball_data", Float32MultiArray, self.callback)
        rospy.Subscriber("/rtabmap/odom", Odometry, self.odom_callback)
        self.marker_publisher = rospy.Publisher("/ball_variance_ellipse", Marker, queue_size=10)
        self.global_ball_data_publisher = rospy.Publisher("/global_ball_data", PoseWithCovarianceStamped, queue_size=10)
        
        # checked via tape measurer
        self.observable_distance = 10000 # extended via smaller radius minimum
        # it to that results in OOB still being in the blue region, idk
        
        # checked via moving ball towards camera to find minimum computed distance
        self.minimum_observable_distance = 155
        # checked via moving ball to edge of camera FOV to check angle
        self.observable_angle = 0.48 # OLD: 0.63
        self.scale = 1 / 20
        self.frame_height = int(self.observable_distance * self.scale)
        self.frame_width = int(self.observable_distance * math.sin(self.observable_angle) * 2 * self.scale)

        # Index 0: (+) is away from camera, (-) is towards camera
        # Index 1: (+) is to the right, (-) is to the left
        # y=200 seems optimal for midterm presentation
        self.motion_control = np.array([[0], [0]], dtype=np.float64)
        
        # noise in x ... noise in y
        self.motion_noise = np.array([[10, 0.0], [0.0, 10]], dtype=np.float64)
        
        self.last_dist = gauss2D_from_polar(8000, 0, np.array([[0, 0], [0, 0]], dtype=np.float64))
        self.last_time = rospy.get_rostime().secs
        self.last_observation_time = None
        self.robot_pose = np.array([[0], [0]], dtype=np.float64)
        self.robot_orientation = 0

        # millimeters
        self.sensor_translation = np.array([[-228.6], [0]], dtype=np.float64)
        self.sensor_theta = 0

        self.draw_observation = False
        self.draw_estimation = False

        rospy.loginfo("Starting ball localizer...")

        rospy.spin()

    def motion_model(self, u) :
        return np.array([[u[0, 0] * math.cos(u[1, 0])], [u[0, 0] * math.sin(u[1, 0])]], dtype=np.float64)

    def sensor_model(self, x) :
        return x

    def transform_to_global(self, x) :
        return local_target_pose_to_global(x, self.sensor_translation, self.sensor_theta, self.robot_pose, self.robot_orientation)

    # TODO: sensor noise for angle
    def sensor_noise(self, d) :
        return np.array([[8.809328 * 1.000394 ** d, 0], [0, 0.1]], dtype=np.float64)
        
    def callback(self, data) :
        """Data 0 is distance, Data 1 is theta"""
        if self.draw_observation or self.draw_estimation :
            display_frame = np.zeros(shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)
            cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.observable_distance * self.scale), int(self.observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (150, 0, 30), -1)
            cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), (int(self.minimum_observable_distance * self.scale), int(self.minimum_observable_distance * self.scale)), 0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), (0, 0, 0), -1)

        distance = data.data[0]
        theta = data.data[1]

        time = rospy.get_rostime().to_sec()
        dt = time - self.last_time
        self.last_time = time

        (predicted_mean, predicted_covariance) = ekf_predict(self.last_dist.u, self.last_dist.S, self.motion_control, self.motion_model, self.motion_noise, dt)

        if distance > self.minimum_observable_distance and distance < self.observable_distance and abs(theta) < self.observable_angle :
            dist = gauss2D_from_polar(distance, theta, self.sensor_noise(distance))
            if self.draw_observation :
                u, l1, l2, angle = ellipse_from_gauss2D(dist)
                cv.ellipse(display_frame, (int(u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(u[0][0] * self.scale)), (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (0, 0, 255), -1)
            (corrected_mean, corrected_covariance) = ekf_correct(predicted_mean, predicted_covariance, self.transform_to_global(dist.u), self.sensor_model, dist.S)
            #delta_xy = (corrected_mean - self.last_dist.u) / dt
            #self.motion_control = np.array([[math.sqrt(delta_xy[0, 0]**2 + delta_xy[1, 0]**2)], [math.atan2(-corrected_mean[0, 0], -corrected_mean[1, 0])]])
            self.last_dist = Gauss2D(corrected_mean, corrected_covariance)

        else :

            self.last_dist = Gauss2D(predicted_mean, predicted_covariance)
        
        # rospy.loginfo(f"u: {self.last_dist.u}")
        # rospy.loginfo(f"S: {self.last_dist.S}")
        u, l1, l2, angle = ellipse_from_gauss2D(self.last_dist)

        if self.draw_estimation :
            cv.ellipse(display_frame, (int(u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(u[0][0] * self.scale)), (int(l2 * self.scale), int(l1 * self.scale)), math.degrees(angle), 0, 360, (255, 255, 255), -1)
        
        marker = Marker()
        marker.header.frame_id = "d400_aligned_depth_to_color_frame"
        marker.header.stamp = rospy.Time.now()
        # type for cylinder since we don't have z variance
        marker.type = 3
        # scale is allegedly in meters so z is 1 to make it tall enough to see and the / 1000 is because these are in millimeters
        # we should play with these if that seems wrong in practice
        marker.scale.x = self.last_dist.S[0, 0] / 1000
        marker.scale.y = self.last_dist.S[1, 1] / 1000
        marker.scale.z = 1
        # this should be clear, 0.5 is because i think scale is on both sides of the center
        marker.pose.position.x = self.last_dist.u[0][0] / 1000
        marker.pose.position.y = -self.last_dist.u[1][0] / 1000
        marker.pose.position.z = 0.5
        # convert from rpy to quaternion
        orientation = quaternion_from_euler(0, 0, -angle)
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
        # rgba on a 0-1 scale, so i think this should be white with 100% opacity
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1

        self.marker_publisher.publish(marker)

        # pose_with_covariance = PoseWithCovariance()

        # #pose_with_covariance.header.frame_id = "map"
        # #pose_with_covariance.header.stamp = rospy.Time.now()

        # pose_with_covariance.pose.position.x = self.last_dist.u[0][0] / 1000
        # pose_with_covariance.pose.position.y = self.last_dist.u[1][0] / 1000
        # pose_with_covariance.pose.position.z = 0

        # pose_with_covariance.covariance = np.array([[self.last_dist.S[0, 0], self.last_dist.S[0, 1], 0, 0, 0, 0], 
        #                                             [self.last_dist.S[1, 0], self.last_dist.S[1, 1], 0, 0, 0, 0], 
        #                                             [0, 0, 0, 0, 0, 0], 
        #                                             [0, 0, 0, 0, 0, 0], 
        #                                             [0, 0, 0, 0, 0, 0], 
        #                                             [0, 0, 0, 0, 0, 0]], dtype=np.float64)
        
        # self.global_ball_data_publisher.publish(pose_with_covariance)
        

        pose_with_covariance_stamped = PoseWithCovarianceStamped()

        pose_with_covariance_stamped.header.frame_id = "d400_aligned_depth_to_color_frame"  # or "odom", or whatever global frame you're using
        pose_with_covariance_stamped.header.stamp = rospy.Time.now()

        pose_with_covariance_stamped.pose.pose.position.x = self.last_dist.u[0][0] / 1000
        pose_with_covariance_stamped.pose.pose.position.y = self.last_dist.u[1][0] / 1000
        pose_with_covariance_stamped.pose.pose.position.z = 0

        pose_with_covariance_stamped.pose.covariance = np.array([
            [self.last_dist.S[0, 0] / 1000, self.last_dist.S[0, 1] / 1000, 0, 0, 0, 0], 
            [self.last_dist.S[1, 0] / 1000, self.last_dist.S[1, 1] / 1000, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float64).flatten().tolist()

        self.global_ball_data_publisher.publish(pose_with_covariance_stamped)

        rospy.loginfo(f"ball pose according to ball localizer (x y) {self.last_dist.u[0, 0]} {self.last_dist.u[1, 0]}")
        

        if self.draw_estimation or self.draw_estimation :
            cv.imshow('Space', display_frame)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")

    def odom_callback(self, odom) :
        orientation = odom.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.robot_pose = np.array([[odom.pose.pose.position.x * 1000], [odom.pose.pose.position.y * 1000]], dtype=np.float64)
        self.robot_orientation = yaw
        rospy.loginfo(f"robot pose according to ball localizer (x y theta) {self.robot_pose[0, 0]} {self.robot_pose[1, 0]} {self.robot_orientation}")

if __name__ == '__main__':
    try:
        BallLocalizer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
