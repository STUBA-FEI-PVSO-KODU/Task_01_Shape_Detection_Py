# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:03:07 2022

@author: Michal Kovac
"""

import cv2
import numpy as np

frameWidth = 640
frameHigh = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHigh)

def empty(a):
    pass
    

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold 1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold 2", "Parameters", 255, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    high = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((high, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContorous(img, imgContour):
     
    
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin: #Filter na velkost   
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            epsilon = 0.02*peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(approx) == 3:
                cv2.putText(imgContour, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 4:
                cv2.putText(imgContour, 'Stvorec', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 5:
                cv2.putText(imgContour, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 6:
                cv2.putText(imgContour, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            else:
                cv2.putText(imgContour, 'Kruh', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            
    
while True:
    success, img = cap.read()
    imgContour = img.copy()
    
    threshold1 = cv2.getTrackbarPos("Threshold 1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold 2", "Parameters")
    
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGary = cv2.cvtColor(imgBlur,  cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGary, threshold1, threshold2)
    
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)
    
    getContorous(imgDil, imgContour)
    
    imgStack = stackImages(0.8, ([img, imgGary, imgCanny], [imgDil, imgContour, imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break