#!/usr/bin/env python

# Import Numpy
import numpy as np

# Import Probabilistic Models

# from diff_motion_professor import MotionModel as MotionModel
from RGBDSensorModel2d import RGBDSensorModel2d as SensorModel

# Import ROS Dependencies
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Vector3, Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Float32MultiArray

class ConstantVelocityMotionModel(Object):
    
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
        

class SensorModel(Object):
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

        
        
         
class ExtendedKalmanFilter(Object):

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

        self.state_covariance = np.array([[0.0, 0.0, 0.0]
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])

        # Initalize Control Space: ASSUMES 2X1 CONTROL SPACE
        self.control_input = np.array([[0.0],
                            [0.0]])

        # Initalize Covarience Matricies: ASSUMES 3X1 STATE SPACE
        self.state_covariance_matrix = np.array([[0.0, 0.0, 0.0]
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])

        # Initialize Time
        self.last_time = rospy.get_time().toSec()

    def get_dk(self):

        # Calculate dk
        dk = rospy.get_time().toSec() - self.last_time

        # Update Previous Time
        self.last_time = rospy.get_time().toSec()

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

if __name__ == '__main__':

    try:
        # Initalize ROS Node
        rospy.init_node('ekf', anonymous=True)
        marker_pub = rospy.Publisher('/ekf/estimated_state', Odometry, queue_size=10)

        # Initalize Models
        MotionModel = ConstantVelocityMotionModel()
        SensorModel = SensorModel()
        EKF = ExtendedKalmanFilter(MotionModel, SensorModel)

        # Initalize ROS Subscriber
        # YOUR CODE HERE

        while not rospy.is_shutdown():
            rospy.spin()

    except rospy.ROSInterruptException:
        pass