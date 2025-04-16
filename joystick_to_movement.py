#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class PublishMovementFromController():

    def __init__(self): 
        # Set Speed Scaler:
        self.safety_scale = 0.2
        
        # Initialize Button Definitions; MODIFY FOR YOUR CONTROLLER.
        self.linear_axis = 1
        self.angular_axis = 3
        self.horizontal_axis = 6
        self.vertical_axis = 7
        self.safety_button = 5
    
        # This topic subscribed to by the robot's movement node.
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Take in joystick input and eventually publish to cmd_vel topic
        rospy.Subscriber('/joy', Joy, self.callback)
        
        rospy.loginfo('PublishMovementFromController Node Initialized\n',
                      'Hold right bumper to allow for input\n',
                      'Left stick controls forward/backward movement\n',
                      'Right stick controls left/right rotation\n')

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
