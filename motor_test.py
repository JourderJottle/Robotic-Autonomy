#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import numpy as np
import time

class MotorTest:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('motor_test', anonymous=True)
        
        # Publisher for motor commands
        self.motor_pub = rospy.Publisher('/motor_commands', Float32MultiArray, queue_size=10)
        
        # Subscribe to joystick input
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        # Motor parameters
        self.max_speed = 1.0  # Maximum speed (adjust based on your robot)
        self.current_speed = 0.0
        
        # Controller mapping (adjust based on your controller)
        self.LEFT_STICK_VERTICAL = 1    # Left stick vertical axis
        self.LEFT_STICK_HORIZONTAL = 0  # Left stick horizontal axis
        self.RIGHT_TRIGGER = 5          # Right trigger
        self.LEFT_TRIGGER = 2           # Left trigger
        self.A_BUTTON = 0               # A button
        self.B_BUTTON = 1               # B button
        self.X_BUTTON = 2               # X button
        self.Y_BUTTON = 3               # Y button
        
        print("Motor Test Node Initialized")
        print("Controller Controls:")
        print("  Left Stick: Forward/Backward")
        print("  Right Stick: Left/Right")
        print("  A Button: Emergency Stop")
        print("  B Button: Reset Emergency Stop")
        print("  Right Trigger: Increase Speed")
        print("  Left Trigger: Decrease Speed")

    def joy_callback(self, data):
        # Get controller inputs
        left_y = data.axes[self.LEFT_STICK_VERTICAL]
        left_x = data.axes[self.LEFT_STICK_HORIZONTAL]
        right_trigger = data.axes[self.RIGHT_TRIGGER]
        left_trigger = data.axes[self.LEFT_TRIGGER]
        
        # Handle emergency stop
        if data.buttons[self.A_BUTTON]:  # Emergency stop
            self.emergency_stop = True
            print("EMERGENCY STOP ACTIVATED")
        elif data.buttons[self.B_BUTTON]:  # Reset emergency stop
            self.emergency_stop = False
            print("Emergency stop reset")
        
        # Adjust speed with triggers
        if right_trigger > 0.1:  # Right trigger pressed
            self.current_speed = min(self.current_speed + 0.1, self.max_speed)
            print(f"Speed increased to: {self.current_speed}")
        elif left_trigger > 0.1:  # Left trigger pressed
            self.current_speed = max(self.current_speed - 0.1, 0.0)
            print(f"Speed decreased to: {self.current_speed}")
        
        # Calculate motor speeds based on controller input
        if not self.emergency_stop:
            # Convert joystick input to motor speeds
            # Using differential drive kinematics
            # left_y controls forward/backward
            # left_x controls turning
            left_speed = self.current_speed * (left_y - left_x)
            right_speed = self.current_speed * (left_y + left_x)
            
            # Send motor command
            self.send_motor_command(left_speed, right_speed)
        else:
            self.send_motor_command(0.0, 0.0)

    def send_motor_command(self, left_speed, right_speed):
        # Create and publish motor command message
        msg = Float32MultiArray()
        msg.data = [left_speed, right_speed]
        self.motor_pub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        motor_test = MotorTest()
        motor_test.run()
    except rospy.ROSInterruptException:
        pass 