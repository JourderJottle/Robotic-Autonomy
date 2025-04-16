#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import time
import sys
import select
import termios
import tty

class MotorTest:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('motor_test', anonymous=True)
        
        # Publisher for motor commands
        self.motor_pub = rospy.Publisher('/motor_commands', Float32MultiArray, queue_size=10)
        
        # Motor parameters
        self.max_speed = 1.0  # Maximum speed (adjust based on your robot)
        self.speed_increment = 0.1
        self.current_speed = 0.0
        
        # Safety parameters
        self.emergency_stop = False
        self.command_timeout = 0.5  # seconds
        self.last_command_time = time.time()
        
        # Initialize terminal settings for keyboard input
        self.settings = termios.tcgetattr(sys.stdin)
        
        print("Motor Test Node Initialized")
        print("Controls:")
        print("  w: Forward")
        print("  s: Backward")
        print("  a: Turn Left")
        print("  d: Turn Right")
        print("  q: Increase Speed")
        print("  e: Decrease Speed")
        print("  x: Emergency Stop")
        print("  c: Exit")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def send_motor_command(self, left_speed, right_speed):
        if self.emergency_stop:
            left_speed = 0.0
            right_speed = 0.0
        
        # Create and publish motor command message
        msg = Float32MultiArray()
        msg.data = [left_speed, right_speed]
        self.motor_pub.publish(msg)
        self.last_command_time = time.time()

    def run(self):
        try:
            while not rospy.is_shutdown():
                # Check for command timeout
                if time.time() - self.last_command_time > self.command_timeout:
                    self.send_motor_command(0.0, 0.0)
                
                key = self.get_key()
                
                if key == 'w':  # Forward
                    self.send_motor_command(self.current_speed, self.current_speed)
                elif key == 's':  # Backward
                    self.send_motor_command(-self.current_speed, -self.current_speed)
                elif key == 'a':  # Turn Left
                    self.send_motor_command(-self.current_speed, self.current_speed)
                elif key == 'd':  # Turn Right
                    self.send_motor_command(self.current_speed, -self.current_speed)
                elif key == 'q':  # Increase Speed
                    self.current_speed = min(self.current_speed + self.speed_increment, self.max_speed)
                    print(f"Speed increased to: {self.current_speed}")
                elif key == 'e':  # Decrease Speed
                    self.current_speed = max(self.current_speed - self.speed_increment, 0.0)
                    print(f"Speed decreased to: {self.current_speed}")
                elif key == 'x':  # Emergency Stop
                    self.emergency_stop = True
                    self.send_motor_command(0.0, 0.0)
                    print("EMERGENCY STOP ACTIVATED")
                elif key == 'c':  # Exit
                    self.send_motor_command(0.0, 0.0)
                    break
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up
            self.send_motor_command(0.0, 0.0)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

if __name__ == '__main__':
    try:
        motor_test = MotorTest()
        motor_test.run()
    except rospy.ROSInterruptException:
        pass 