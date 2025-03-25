#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from realsense2_camera.msg import Extrinsics
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
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.depth_camera_info_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.color_camera_info_callback)
        rospy.Subscriber("/camera/extrinsics/depth_to_color", Extrinsics, self.depth_to_color_extrinsics_callback)
        
        # Camera frame transformations
        self.KcI = None
        self.Kd = None
        self.cdRotation = None
        self.cdTranslation = None

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
            x = self.ball_2d_data[0]
            y = self.ball_2d_data[1]
            coords = np.array([[x], [y]])
            new_coords = self.Kd @ (self.cdRotation @ (self.KcI @ coords) + self.cdTranslation)
            rospy.loginfo(f"New Coords: {new_coords}")
            cv.circle(cv_img, (x, y), 5, (0,0,255), -1)
            # Display the resulting frame
            cv.imshow('Depth Video', cv_img)
            #cv.imshow('Mask', mask)
            if cv.waitKey(10) & 0xFF == ord('b'):
                rospy.signal_shutdown("Shutting down")
            
            depth_at_center = cv_img[y][x]

            self.ball_data_pub.publish(Float32MultiArray(data=[depth_at_center, self.ball_2d_data[2]]))
            rospy.loginfo(f'Published d = {depth_at_center} and theta = {self.ball_2d_data[2]}')
        else :
            self.ball_data_pub.publish(Float32MultiArray(data=None))

    def color_camera_info_callback(self, data):
        if self.focal_length == None or self.image_width == None or self.KcI == None :
            self.focal_length = data.K[0]
            self.image_width = data.K[2]
            self.KcI = np.pinv(np.matrix(data.K[0:3], data.K[3:6], data.K[6:9]))

    def depth_camera_info_callback(self, data):
        if self.Kd == None :
            self.Kd = np.matrix(data.K[0:3], data.K[3:6], data.K[6:9])

    def depth_to_color_extrinsics_callback(self, data) :
        if self.cdRotation == None or self.cdTranslation == None :
            self.cdRotation = np.pinv(np.matrix([data.rotation[0:3], data.rotation[3:6], data.rotation[6:9]]))
            self.cdTranslation = -np.array(data.translation)

    def color_callback(self, data) :
        if self.image_width != None and self.focal_length != None :
            try:
                # Convert the ROS Image message to OpenCV format
                frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return
            
            frame_with_contours = frame.copy()
            
            # Process frame
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Define the color range for the ball (blue in this case)
            lower_color = np.array([70, 100, 100])
            upper_color = np.array([140, 255, 255])
            
            mask = cv.inRange(hsv, lower_color, upper_color)
            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            cv.drawContours(frame_with_contours, contours, -1, (0, 0, 255), 2)

            """
            frame_with_largest_circle = frame.copy()
            circles = None
            cv.HoughCircles(mask, circles, cv.CV_HOUGH_GRADIENT, 1, 50, 50, 0.9, 10, -1)
            if len(circles) > 0 and len(circles[0]) == 3 :
                largest_circle = circles[0][0]
                cv.circle(frame_with_largest_circle, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 0, 255), 2)
            """

            largest_contour = None
            x = 0
            y = 0
            r = 0
            center = (0, 0)
            
            if len(contours) != 0:
                # Find largest because that's probably the ball
                largest_contour = max(contours, key=cv.contourArea)
                
                (x, y), r = cv.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                r = int(r)
                # minimum circle radius; it tends to see bits of the environment currently.
                # also do a minimum area of the circle which the contour takes up? brainstorming ways to avoid seeing the box lids
            if r > 20 and cv.contourArea(largest_contour) / (math.pi * r**2) > 0.3 :
                cv.circle(frame, center, r, (0, 255, 0), 2)
                cv.circle(frame, center, 5, (0,0,255), -1)
                cv.drawContours(frame, [largest_contour], -1, (255, 255, 255), 2)

                theta = math.atan((center[0] - self.image_width) / self.focal_length)
                
                self.ball_2d_data = [center[0], center[1], theta]
            else :
                self.ball_2d_data = None
            
            
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
