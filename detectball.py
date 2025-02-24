import cv2 as cv
import numpy as np
import imutils
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([100,30,30])
    upper_yellow = np.array([140,255,255])
 
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) != 0 :

        largest_contour = max(contours, key = cv.contourArea)

        M = cv.moments(largest_contour)
        if M["m00"] != 0 :
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv.drawContours(frame, contours, -1, 255, 3)

    cv.imshow('Video', frame)
    cv.imshow('Mask', mask)

    if(cv.waitKey(10) & 0xFF == ord('b')):
        break
    