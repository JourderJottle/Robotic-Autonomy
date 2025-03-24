#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import math
import imutils

class BallTracker:
    
    def __init__(self):
        rospy.init_node('ball_tracker', anonymous=True)
        self.bridge = CvBridge()
        
        # Publish ball data
        self.ball_data_pub = rospy.Publisher("/ball_data", Float32MultiArray, queue_size=10)

        # Subscribe to the depth camera feed
        
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.d_and_theta_callback)
        
        # Subscribe to the color camera feed
        self.ball_2d_data = None
        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        
        # Subscribe to depth camera info for focal length
        self.focal_length = None
        self.image_width = None
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        
        
        rospy.loginfo("Ball Tracker Node Initialized")
        
        # spin instead of while true
        rospy.spin()

    def d_and_theta_callback(self, data) :
        if self.ball_2d_data != None :
            try :
                cv_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return
            
        # depth_at_center = cv_img[self.ball_2d_data[1]][self.ball_2d_data[0]]
        x, y = self.ball_2d_data[0], self.ball_2d_data[1]
        region = cv_img[max(0, y - 2):y + 3, max(0, x - 2):x + 3]  # 5x5 region

        # Use nanmean to automatically handle NaNs
        depth_at_center = np.nanmean(region)

        self.ball_data_pub.publish(Float32MultiArray(data=[depth_at_center, self.ball_2d_data[2]]))
        rospy.loginfo(f'Published d = {depth_at_center} and theta = {self.ball_2d_data[2]}')

    def camera_info_callback(self, data):
        if self.focal_length == None or self.image_width == None :
            self.focal_length = data.K[0]
            self.image_width = data.K[2]

    def color_callback(self, data) :
        if self.image_width != None and self.focal_length != None :
            try:
                # Convert the ROS Image message to OpenCV format
                frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return
            
            frame_with_contours = frame.copy()
            #frame_with_largest_circle = frame.copy()
            
            # Process frame
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Define the color range for the ball (blue in this case)
            lower_color = np.array([70, 100, 100])
            upper_color = np.array([140, 255, 255])
            
            mask = cv.inRange(hsv, lower_color, upper_color)
            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            cv.drawContours(frame_with_contours, contours, -1, (0, 0, 255), 2)

            # circles = None
            # cv.HoughCircles(mask, circles, cv.CV_HOUGH_GRADIENT, 1, 50, 50, 0.9, 10, -1)
            # if len(circles) > 0 and len(circles[0]) == 3 :
            #     largest_circle = circles[0][0]
            #     cv.circle(frame_with_largest_circle, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 0, 255), 2)
            
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

                    theta = math.atan((center[0] - self.image_width) / self.focal_length)
                    
                    self.ball_2d_data = [center[0], center[1], theta]
            
            
        # Display the resulting frame
        cv.imshow('Video', frame)
        #cv.imshow('Mask', mask)
        if cv.waitKey(10) & 0xFF == ord('b'):
            rospy.signal_shutdown("Shutting down")
        


if __name__ == '__main__':
    try:
        BallTracker()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node terminated.")
    finally:
        cv.destroyAllWindows()
