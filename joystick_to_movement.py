#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class PublishMovementFromController():

    def __init__(self): 
        # Set Speed Scaler:
        self.safety_scale = 0.4
        
        # Initialize Button Definitions; MODIFY FOR YOUR CONTROLLER.
        self.linear_axis = 1
        self.angular_axis = 3
        self.horizontal_axis = 6
        self.vertical_axis = 7
        self.safety_button = 5
    

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/joy', Joy, self.callback)

    def callback(self, data):

        twist = Twist()

        # Check if safety button is being held down
        if data.buttons[self.safety_button]:

            #  Joystick value is cubed to allow for finer control at lower speeds.
            twist.linear.x = self.safety_scale*(data.axes[self.linear_axis])**3
            twist.angular.z = self.safety_scale*(data.axes[self.angular_axis])**3

        else:
            twist.linear.x = 0
            twist.angular.z = 0

        # Publish Data
        self.pub.publish(twist)

if __name__ == "__main__":

    rospy.init_node("EnableMovementFromController", anonymous=True)  

    controller = PublishMovementFromController()

    rospy.spin()
