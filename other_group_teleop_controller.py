#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import UInt16MultiArray

class controller():

    def __init__(self): 

        # Initialize Button Definitions; MODIFY FOR YOUR CONTROLLER.
        self.linear_axis = 1
        self.angular_axis = 3
        self.horizontal_axis = 6
        self.vertical_axis = 7
        self.safety_button = 5
        self.safety_scale = 0.7
        
        # Setup ROS Namespace.
        self.path = rospy.get_namespace()

        # Setup ROS Publisher For Servos.
        self.camera_pub = rospy.Publisher(str(self.path) + 'servo_position', UInt16MultiArray, queue_size=1)

        # Setup ROS Publisher For Base.
        self.base_pub = rospy.Publisher(str(self.path) + 'cmd_vel', Twist, queue_size=1)

        # Setup ROS Subscriber.
        rospy.Subscriber(str(self.path) + 'joy', Joy, self.callback)

        # Initalize Servo Position.
        self.servo_x = 90
        self.servo_y = 90

    def bound(self, low, high, value):

        return max(low, min(high, value))

    def iterate(self, value, step):

        value = self.bound(0, 180, value+step)

        return value

    def callback(self, data):

        # Initialize Messages.
        servo = UInt16MultiArray()
        twist = Twist()

        # Initalize Servo Data.
        servo.data = [self.servo_x,self.servo_y]

        # Check Button Press.	
        if data.buttons[self.safety_button]:

            # Scale Base Axis.
            twist.linear.x = self.safety_scale*(data.axes[self.linear_axis])**3
            twist.angular.z = self.safety_scale*(data.axes[self.angular_axis])**3

            # Scale Horizontal Axis.
   #         if data.axes[self.horizontal_axis] == 1:
#
 #               servo.data[0] = self.iterate(self.servo_x, 1)
#
 #           if data.axes[self.horizontal_axis] == -1:
#
 #               servo.data[0] = self.iterate(self.servo_x, -1)
#
 #           # Scale Vertical Axis.
  #          if data.axes[self.vertical_axis] == 1:
#
 #               servo.data[1] = self.iterate(self.servo_y, 1)
#
 #           if data.axes[self.vertical_axis] == -1:
#
 #               servo.data[1] = self.iterate(self.servo_y, -1) 

        else:
            twist.linear.x = 0
            twist.angular.z = 0
        rospy.loginfo("Linear: %s, Angular: %s, Servo X: %s, Servo Y: %s",
                      twist.linear.x, twist.angular.z, servo.data[0], servo.data[1])
        # Publish New Data.
        self.base_pub.publish(twist)
        self.camera_pub.publish(servo)

if __name__ == "__main__":

    # Initalize ROS Node.
    rospy.init_node("controller")  

    # Initalize Python Class.      
    controller = controller()

    # Start ROS Node.
    rospy.spin()
