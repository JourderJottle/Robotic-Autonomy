#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import imutils

class BallTracker:
    
    def __init__(self):
        rospy.init_node('ball_tracker', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to the camera feed
        rospy.Subscriber("video_topic", Image, self.image_callback)
        
        rospy.loginfo("Ball Tracker Node Initialized")
        
        # spin instead of while true
        rospy.spin()
        
    def image_callback(self, data):
        try:
            # Convert the ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Process frame
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Define the color range for the ball (yellow in this case)
        lower_color = np.array([100, 30, 30])
        upper_color = np.array([140, 255, 255])
        
        mask = cv.inRange(hsv, lower_color, upper_color)
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if len(contours) != 0:
            # Find largest because that's probably the ball
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw the circle
                cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv.drawContours(frame, contours, -1, 255, 3)
            
            
        # Display the resulting frame
        cv.imshow('Video', frame)
        cv.imshow('Mask', mask)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")

if __name__ == '__main__':
    try:
        BallTracker()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
    finally:
        cv.destroyAllWindows()