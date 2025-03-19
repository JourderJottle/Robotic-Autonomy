import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import math

class MotionModel :
    def __init__(self) :
        rospy.init_node("motion_model", anonymous=True)
        rospy.Subscriber("/adversary_pose", Float32MultiArray, self.callback)
        self.publisher = rospy.Publisher("/expected_adversary_pose", Float32MultiArray, queue_size=10)

        self.wheel_radius = 76.2 # millimeters, by measuring tape
        self.virtual_motor_radius = 49 # https://www.superdroidrobots.com/store/robot-parts/electrical-parts/encoders-accessories/encoder-motors/product=849
        self.track_width = 444.5 # millimeters, by measuring tape
        self.gear_ratio = self.wheel_radius / self.virtual_motor_radius

        rospy.spin()

    def robot_speed_from_wheel_speeds(self, u) :
        wl = u[0, 0]
        wr = u[1, 0]
        linear_velocity = self.gear_ratio * (wl + wr) / 2
        angular_velocity = self.gear_ratio * (wr - wl) / self.track_width
        return np.matrix([linear_velocity * math.cos(angular_velocity), linear_velocity * math.sin(angular_velocity), angular_velocity]).T

    # data.data is an array with entries: [robot x, robot y, robot theta, left wheel speed, right wheel speed, delta_time]
    def callback(self, data) :
        x_k1 = np.matrix(data.data[0:3]).T
        xdot = self.robot_speed_from_wheel_speeds(np.matrix(data.data[3:5]).T)
        x_k = x_k1 + data.data[5] * xdot
        self.publisher.publish(Float32MultiArray(data=[x_k[0, 0], x_k[1, 0], x_k[2, 0]]))

