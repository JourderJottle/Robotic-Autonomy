#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import imutils

class BallTracker:
    
    def __init__(self):
        rospy.init_node('ball_tracker', anonymous=True)
        self.bridge = CvBridge()
        
        # Publish center of ball eventually
        self.ball_center_pub = rospy.Publisher("\ball_center", Int32MultiArray, queue_size=10)
        
        # Subscribe to the camera feed
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
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
        
        frame_with_contours = frame.copy()
        frame_with_largest_circle = frame.copy()
        
        # Process frame
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Define the color range for the ball (blue in this case)
        lower_color = np.array([70, 100, 100])
        upper_color = np.array([140, 255, 255])
        
        mask = cv.inRange(hsv, lower_color, upper_color)
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        cv.drawContours(frame_with_contours, contours, -1, (0, 0, 255), 2)

        circles = None
        cv.HoughCircles(mask, circles, cv.CV_HOUGH_GRADIENT, 1, 50, 50, 0.9, 10, -1)
        if len(circles) > 0 and len(circles[0]) == 3 :
            largest_circle = circles[0][0]
            cv.circle(frame_with_largest_circle, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 0, 255), 2)
        
        if len(contours) != 0:
            # Find largest because that's probably the ball
            largest_contour = max(contours, key=cv.contourArea)
            
            (x, y), r = cv.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            r = int(r)
            
            if r > 0:
                cv.circle(frame, center, r, (0, 255, 0), 2)
                cv.circle(frame, center, 5, (0,0,255), -1)
                cv.drawContours(frame, [largest_contour], -1, (255, 255, 255), 2)
                
                # Publish center data
                ball_center_msg = Int32MultiArray()
                ball_center_msg.data = [center[0], center[1]]
                self.ball_center_pub.publish(ball_center_msg)
                rospy.loginfo(f"Published ball center at: {center}")
            
            
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