import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import imutils

rospy.init_node("camera_publisher", anonymous=True)

publisher = rospy.Publisher("video_topic", Image, queue_size=60)

rate = rospy.Rate(30)

videoCaptureObject = cv.VideoCapture(0)

bridgeObject = CvBridge()

while not rospy.is_shutdown() :
    returnValue, capturedFrame = videoCaptureObject.read()
    if returnValue == True :
        rospy.loginfo("Video frame captured and published")
        imageToTransmit = bridgeObject.cv2_to_imgmsg(capturedFrame, encoding="bgr8")
        publisher.publish(imageToTransmit)
    rate.sleep()