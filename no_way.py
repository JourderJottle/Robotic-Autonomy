#!/usr/bin/env python3


import numpy as np
import cv2 as cv
import math
from collections import deque
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray


# Import Numpy
import numpy as np

# Import Probabilistic Models


# Import ROS Dependencies
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

class ConstantVelocityMotionModel():
    
    def __init__(self, W=np.array([[0.0],[0.0],[0.0],[0.0]])):
        # State transition matrix for constant velocity model (4x4)
        self.Ak = np.array([[1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        # Process noise covariance (4x1)
        self.W = W
        
    def predict(self, mXkmve1Gkmve1, Uk, dk):
        '''
        Predicts the next state based on constant velocity model.

        Parameters:
                mXkmve1Gkmve1 (float []): Previous state estimate (x, y, vx, vy)
                Uk (float []): Control input (vx, vy) at time k
                dk (float): Time difference between k-1 and k

        Returns:
                mXkGkmve1 (float []): Predicted state (x, y, vx, vy)
        '''
        # Extract previous state (x, y, vx, vy)
        x_kmve1, y_kmve1, vx_kmve1, vy_kmve1 = mXkmve1Gkmve1.flatten()

        # Control input (vx, vy)
        vx_k, vy_k = Uk.flatten()

        # Predict next state using the constant velocity model
        # The state update equation for constant velocity is:
        # x_k = x_kmve1 + vx_k * dk
        # y_k = y_kmve1 + vy_k * dk
        # vx_k, vy_k remain constant
        mXkGkmve1 = np.array([[x_kmve1 + vx_k * dk],
                              [y_kmve1 + vy_k * dk],
                              [vx_k],
                              [vy_k]])

        return mXkGkmve1
        
        
    def predict_covariance(self, sXkmve1Gkmve1, dk):
        '''
        Predicts the covariance of the state.

        Parameters:
                sXkmve1Gkmve1 (float []): Previous covariance estimate
                dk (float): Time difference between k-1 and k

        Returns:
                sXkGkmve1 (float []): Predicted covariance matrix of the state
        '''
        # Predicted state covariance
        # The covariance matrix update for constant velocity will depend on the process noise and time step
        sXkGkmve1 = self.Ak @ sXkmve1Gkmve1 @ self.Ak.T + self.W * dk
        
        return sXkGkmve1
        

class SensorModelClass():
    '''
    Object representing a sensor model for 2D position measurements. Assumes direct 2D position measurement (x, y).
    Can be extended for more complex measurements.

    Parameters:
                R (float []): Measurement noise covariance matrix (2x2 for 2D position)

    Returns:
                predicted_measurement (float []): Predicted sensor measurement (x, y)
                measurement_cov (float []): Covariance of the measurement
    '''
    
    def __init__(self, R=np.array([[0.1, 0.0], [0.0, 0.1]])):
        '''
        Initialize the sensor model with measurement noise covariance.
        Default R assumes 0.1 standard deviation for both x and y measurements.
        '''
        self.R = R

    def predict_measurement(self, predicted_state):
        '''
        Predict the sensor measurement based on the predicted state.

        Parameters:
                predicted_state (float []): Predicted state (x, y, vx, vy)

        Returns:
                predicted_measurement (float []): Predicted measurement (x, y)
                measurement_cov (float []): Covariance of the predicted measurement
        '''
        # We only measure position (x, y) from the predicted state
        x_pred, y_pred, _, _ = predicted_state.flatten()

        # Predicted measurement is just the position (x, y) in the state
        predicted_measurement = np.array([[x_pred],
                                          [y_pred]])

        # The measurement covariance is the noise covariance of the sensor
        measurement_cov = self.R

        return predicted_measurement, measurement_cov

    def predict_residual_covariance(self, predicted_covariance, measurement_cov):
        '''
        Predict the covariance of the measurement residual.

        Parameters:
                predicted_covariance (float []): Predicted state covariance (4x4 matrix)
                measurement_cov (float []): Measurement covariance (2x2 matrix)

        Returns:
                residual_covariance (float []): Covariance of the measurement residual (2x2 matrix)
        '''
        # The measurement residual covariance is calculated as:
        # H * predicted_covariance * H^T + measurement_cov
        # Since we're measuring position, the measurement matrix H will be a 2x4 matrix,
        # where it extracts the position from the state (x, y).

        H = np.array([[1.0, 0.0, 0.0, 0.0],  # Only x position contributes
                      [0.0, 1.0, 0.0, 0.0]]) # Only y position contributes

        # Residual covariance
        residual_covariance = H @ predicted_covariance @ H.T + measurement_cov

        return residual_covariance

        

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

         
class ExtendedKalmanFilter():

    '''
    Object representing the Extended Kalman Filter. Assumes 3x1 State Space (To be updated later).

    VARIABLE DEFINITIONS: G: Given, m: mean, s: variance, P: covarience, mve = minus

        Parameters:
                motion_model (MotionModelObject): Object Representing Probablistic Motion Model.
                sensor_model (SensorModelObject): Object Representing Probablistic Sensor Model.

        Returns:
                mXkGk (float []): Mean Array of Estiamted State Space. (Mean of X_k given k)
                pXkGk (float []): Covariance Matrix of Estiamted State Space. (Covariance of X_k given k)
    '''
    
    def __init__(self, motion_model, sensor_model):

        # Initalize Probablistic Models
        self.motion_model = motion_model
        self.sensor_model = sensor_model

        # Initialize Predicion Spaces: ASSUMES 3X1 STATE SPACE
        self.state_estimate = np.array([[0.0],
                               [0.0],
                               [0.0]])

        self.state_covariance = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])

        # Initalize Control Space: ASSUMES 2X1 CONTROL SPACE
        self.control_input = np.array([[0.0],
                            [0.0]])

        # Initalize Covarience Matricies: ASSUMES 3X1 STATE SPACE
        self.state_covariance_matrix = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])

        # Initialize Time
        self.last_time = rospy.get_time()

    def get_dk(self):

        # Calculate dk
        dk = rospy.get_time().toSec() - self.last_time

        # Update Previous Time
        self.last_time = rospy.get_time()

        return dk

    def update(self, measurement):

        # Calculate dk
        time_diff = self.get_dk()

        # Predict State Estimate
        predicted_state, predicted_cov = self.motion_model.predict(
            self.state_estimate,
            self.control_input,
            time_diff
        )

        # Calculate the Covariance of the Predicted State Estimate
        predicted_covariance = self.motion_model.predict_covariance(
            self.state_covariance_matrix,
            time_diff
        )

        # Calcualate Measurement Residual (zk - H(mXkGkmve1, Xk)
        predicted_mesasurement, measurement_cov = self.sensor_model.predict_measurement(
            predicted_state
        )

        # Calculate Covariance of the Measurement Residual
        residual_covariance = self.sensor_model.predict_residual_covariacne(
            predicted_covariance,
            measurement_cov
        )

        # Calculate the Near-Optimal Kalman Gain
        kalman_gain = np.dot(
            predicted_covariance,
            np.linalg.pinv(residual_covariance)
        )
        
        measurement_residual = measurement - predicted_mesasurement

        # Calculate an Updated State Estimate using Near-Optimal Kalman Gain
        # self.state_estimate = predicted_state + (kalman_gain @ mYk)
        self.state_estimate = predicted_state + np.dot(kalman_gain, measurement_residual)

        # Calculate an the Covarience of the Updated State Estimate using Near-Optimal Kalman Gain
        self.state_covariance_matrix = np.dot(
            np.eye(3) - np.dot(kalman_gain, self.sensor_model.H),
            predicted_covariance
        )

        return self.state_estimate, self.state_covariance_matrix

class EKFVisualizer:
    def __init__(self):
        self.frame_width = 800
        self.frame_height = 600
        self.observable_distance = 50  # Define max observable distance (in some units)
        self.observable_angle = math.pi / 4  # Define observable angle (in radians)
        self.minimum_observable_distance = 5  # Min observable distance
        self.scale = 100  # Scale factor to map to pixels
        self.queue_size = 10  # Number of observations to average
        self.observation_queue = deque(maxlen=self.queue_size)
        self.distance_total = 0
        self.angle_total = 0

        # Initialize ROS publisher
        self.marker_pub = rospy.Publisher('/ekf/estimated_state', Odometry, queue_size=10)

        # Initialize EKF (Assuming the EKF class is already created)
        MotionModel = ConstantVelocityMotionModel()
        SensorModel = SensorModelClass()
        self.EKF = ExtendedKalmanFilter(MotionModel, SensorModel)

    def callback(self, data):
        """Data 0 is distance, Data 1 is theta"""
        # rospy.loginfo(f'Received data: {data.data}')
        display_frame = np.zeros(shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Add visual representation of the observable area
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height), 
                   (int(self.observable_distance * self.scale), int(self.observable_distance * self.scale)),
                   0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), 
                   (150, 0, 30), -1)
        cv.ellipse(display_frame, (int(self.frame_width / 2), self.frame_height),
                   (int(self.minimum_observable_distance * self.scale), int(self.minimum_observable_distance * self.scale)),
                   0, 270 - math.degrees(self.observable_angle), 270 + math.degrees(self.observable_angle), 
                   (0, 0, 0), -1)

        # Get the current observation
        init_distance = data.data[0]
        init_theta = data.data[1]

        self.distance_total += init_distance
        self.angle_total += init_theta
        self.observation_queue.append((init_distance, init_theta))

        # Average the observations
        if len(self.observation_queue) >= self.queue_size:
            self.distance_total -= self.observation_queue[0][0]
            self.angle_total -= self.observation_queue[0][1]
            self.observation_queue.popleft()
        distance = self.distance_total / self.queue_size
        theta = self.angle_total / self.queue_size
        
        rospy.loginfo(f'distance = {distance}, theta = {theta}\n min disntace = {self.minimum_observable_distance}, max distance = {self.observable_distance}, min angle = {-self.observable_angle}, max angle = {self.observable_angle}')

        # Process only if the distance and angle are within the observable range
        if self.minimum_observable_distance < distance < self.observable_distance and abs(theta) < self.observable_angle:
            # rospy.loginfo(f'Processing observation: distance={distance}, theta={theta}')
            dist = gauss2D_from_polar(distance, theta, self.variance_of_likelihood())

            # Convert from polar to Cartesian coordinates
            u = (int(dist.u[1][0] * self.scale + self.frame_width / 2), self.frame_height - int(dist.u[0][0] * self.scale))

            # Ellipse parameters for uncertainty
            a = dist.S[0, 0]
            b = dist.S[0, 1]
            c = dist.S[1, 1]

            # Calculate eigenvalues to define the ellipse
            l1 = (a + c) / 2 + math.sqrt(((a - c) / 2) ** 2 + b ** 2)
            l2 = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b ** 2)

            # Angle of the ellipse
            angle = 0 if b == 0 and a >= c else math.pi / 2 if b == 0 and a < c else math.atan2(l1 - a, b)

            # Draw the ellipse
            cv.ellipse(display_frame, u, (int(l2 * self.scale), int(l1 * self.scale)),
                       math.degrees(angle), 0, 360, (255, 255, 255), -1)

            # EKF state update
            predicted_state, predicted_covariance = self.EKF.update(np.array([distance, theta]))  # Assuming measurement is distance, theta
            estimated_position = predicted_state[:2]  # Get the estimated position (x, y)

            # Draw the estimated position from EKF
            estimated_pos_x = int(estimated_position[0] * self.scale + self.frame_width / 2)
            estimated_pos_y = int(self.frame_height - estimated_position[1] * self.scale)
            cv.circle(display_frame, (estimated_pos_x, estimated_pos_y), 5, (0, 255, 0), -1)

            # Optionally, publish the estimated state as Odometry message
            odom_msg = Odometry()
            odom_msg.pose.pose.position = Point(estimated_position[0], estimated_position[1], 0)
            self.marker_pub.publish(odom_msg)
        else:
            # rospy.loginfo("Observation out of range")
            blank = 1

        # Show the frame with visualizations
        cv.imshow('EKF Tracking Visualization', display_frame)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")

if __name__ == '__main__':
    try:
        # Initialize ROS Node
        rospy.init_node('ekf', anonymous=True)

        # Initialize visualization class and start the loop
        visualizer = EKFVisualizer()

        # Initialize ROS Subscriber
        rospy.Subscriber('/ball_data', Float32MultiArray, visualizer.callback)

        while not rospy.is_shutdown():
            rospy.spin()

    except rospy.ROSInterruptException:
        pass
