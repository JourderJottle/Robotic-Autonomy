#!/usr/bin/python3

import rospy
from geometry_msgs.msg import Twist


class Controller:
    '''Sends messages to cmd_vel through functions'''

    def __init__(self, safety_scale=0.5): 
        rospy.init_node("Controller", anonymous=True)  

        # Set Speed Scaler:
        self.safety_scale = safety_scale
    
        # This topic subscribed to by the robot's movement node.
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        
    def send_movement(self, linear_x, angular_z):
        '''Sends a Twist message to the cmd_vel topic'''
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub.publish(twist)


