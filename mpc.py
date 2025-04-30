import rospy
import numpy as np
import math
import scipy
from collections import deque
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

def rotational_matrix_2D(theta: float) -> np.array :
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], dtype=np.float64)

class MPC() :
    def __init__(self) :
        rospy.init_node("mpc", anonymous=True)
        rospy.Subscriber("/waypoint", Pose, self.waypoint_callback)
        rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, self.map_callback)

        self.waypoint = np.array([[0], [0]], dtype=np.float64)
        self.state = np.array([[0], [0], [0]], dtype=np.float64)
        self.M = None
        self.map_resolution = None
        self.map_height = None
        self.map_width = None
        self.max_linear_speed = 0.2
        self.max_angular_speed = 0.5
        self.dt = 0.2
        self.nk = 10
        self.last_time = rospy.get_rostime().secs
        self.controls_queue = deque()

        self.drive_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Timer(rospy.Duration(self.nk * self.dt), self.compute_controls)
        rospy.Timer(rospy.Duration(self.dt), self.drive)

        rospy.spin()

    def occupancy_probability(self, x) :
        return self.M.data[math.floor(x[1, 0] / self.map_resolution) * self.map_width + math.floor(x[0, 0] / self.map_resolution)]
    def motion_model(self, u, x) :
        return np.array([[x[0, 0] + u[0, 0] * math.cos(x[2, 0]) * self.dt], [x[1, 0] + u[0, 0] * math.sin(x[2, 0]) * self.dt], [x[2, 0] + u[1, 0] * self.dt]], dtype=np.float64)
    def integral_objective_function(self, x) :
        return 1 + self.occupancy_probability(x)
    def terminal_objective_function(self, x) :
        return math.sqrt((x[0, 0] - self.waypoint[0, 0])**2 + (x[1, 0] - self.waypoint[1, 0])**2) + self.occupancy_probability(x)
    def objective_function(self, controls) :
        integral_cost = 0
        predicted_state = self.state
        for i in range(len(controls), step=2) :
            u = np.array([[controls[i]], [controls[i+1]]], dtype=np.float64)
            predicted_state = self.motion_model(u, predicted_state)
            integral_cost += self.integral_objective_function(predicted_state)
        return integral_cost + self.terminal_objective_function(predicted_state)
    def waypoint_callback(self, x) :
        self.waypoint = np.array([[x.position.x], [x.position.y]], dtype=np.float64)
    def map_callback(self, M) :
        self.M = M
        self.map_resolution = M.info.resolution
        self.map_height = M.info.height
        self.map_width = M.info.width
        rospy.loginfo("map callback")
    def odom_callback(self, odom) :
        orientation = odom.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        robot_pose = np.array([[odom.pose.pose.position.x], [odom.pose.pose.position.y]], dtype=np.float64)
        robot_orientation = yaw
        (trans, rot) = self.transform_listener.lookupTransform("/map", "/odom", rospy.get_rostime())
        (roll, pitch, yaw) = euler_from_quaternion(rot)
        R_M = rotational_matrix_2D(yaw)
        robot_pose = R_M @ robot_pose + np.vstack(trans[0:2])
        robot_orientation += yaw
        self.state = np.array([[robot_pose[0, 0]], [robot_pose[1, 0]], [robot_orientation]], dtype=np.float64)
    def compute_controls(self, timer) :
        rospy.loginfo("calling compute_controls")
        if self.M is not None :
            rospy.loginfo("calling compute_controls and have M")
            guess = [1, 0] * self.nk
            optimized = scipy.optimize.minimize(self.objective_function, guess, bounds=[(-self.max_linear_speed, self.max_linear_speed), (-self.max_angular_speed, self.max_angular_speed)] * self.nk)
            for i in range(len(optimized), step=2) :
                twist = Twist()
                twist.linear.x = optimized[i]
                twist.angular.z = optimized[i+1]
                self.controls_queue.append(twist)
    def drive(self, timer) :
        if self.M is not None and self.controls_queue :
            self.drive_publisher.publish(self.controls_queue.popleft())

if __name__ == '__main__':
    try:
        MPC()
    except rospy.ROSInterruptException:
        rospy.loginfo("Model Predictive Control terminated.")
