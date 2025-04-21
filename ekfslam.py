import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
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

def motion_model(u) :
    return np.ndarray([u[0, 0] * math.cos(u[0, 1]), u[0, 0] * math.sin(u[0, 1])])

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

def ekfslam_predict(previous_state, previous_covariance, input, motion_model, motion_noise, dt) :
    A = derive_gradient(motion_model, input, 0.1)
    # Returns (mean, covariance)
    return (previous_state + dt * np.matrix(motion_model(input)).T, (A @ previous_covariance) @ A.T + motion_noise)

def ekfslam_correct(predicted_state_and_map, predicted_state_and_map_covariance, observation, sensor_model, sensor_noise) :
    C = derive_gradient(sensor_model, predicted_state_and_map, 0.1)
    W = predicted_state_and_map_covariance @ C.T @ np.linalg.pinv(C @ predicted_state_and_map_covariance @ C.T + sensor_noise)
    # Returns (mean, covariance)
    return (predicted_state_and_map + W @ (observation - sensor_model(predicted_state_and_map)), (np.eye(len(predicted_state_and_map)) - W @ C) @ predicted_state_and_map_covariance)

def match_map(map1, map2) :
    out_map1 = []
    out_map2 = []
    for i in map2 :
        out_map2.append(map2[i])
        if map1[i] is None :
            out_map1.append(map2[i])
        else :
            out_map1.append(map1[i])
    return out_map1, out_map2

class EKFSLAM :

    def __init__(self):
        rospy.init_node("EKF_SLAM", anonymous=True)

        self.previous_self_state = None
        self.previous_self_covariance = None
        self.predicted_self_state = None
        self.predicted_self_covariance = None
        self.map_table = {}
        self.previous_map_table = {}
        self.last_time = rospy.get_rostime().secs

        rospy.spin()
    
    def odometry_callback(self, odometry) :
        quat = odometry.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)
        new_self_state = np.ndarray([odometry.pose.pose.position.z, 
                                 odometry.pose.pose.position.x, 
                                 yaw])
        previous_map, new_map = match_map(self.previous_map_table, self.map_table)
        predicted_state_and_map = np.ndarray([self.predicted_self_state, previous_map])
        observation = np.ndarray([new_self_state, new_map])
        

    def map_callback(self, map) :
        self.previous_map_table = self.map_table
        for i in range(len(map.graph.poses)) :
            point = map.graph.poses[i].position
            self.map_table[map.graph.posesId[i]] = np.ndarray([point.z, point.x]).T

    def control_callback(self, controls) :
        time = rospy.get_rostime().to_sec()
        dt = time - self.last_time
        self.last_time = time

        u = controls.data # this is going to be based on however June implements the control output msg

        (self.predicted_self_state, self.predicted_self_covariance) = ekfslam_predict(self.previous_self_state, self.previous_self_covariance, u, motion_model, None, dt)