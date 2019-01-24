import cv2 as cv
import numpy as np
#from cscore import CameraServer
import time

import imprep

startTime = time.time()

#Vision Processing Settings
imgWidth = 320
imgHeight = 240

# #Get access to the camera server
# cameraServer = CameraServer.getInstance()
# cameraServer.enableLogging()
# #Setup a camera object for LifeCam
# lifeCam = cameraServer.startAutomaticCapture()
# lifeCam.setResolution(320,240)

#Get access to the video stream from LifeCam
# vidStream = cameraServer.getVideo()

#Pre allocating space for the frames to be stored
frame = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
cap = cv.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)

x = 0
while True:
    timeElapsed = time.time() - startTime
    # #Capture each frame for processing
    # _, frame = vidStream.grabFrame(frame)
    _, frame = cap.read()
    #Convert frames to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Break the images down into smaller Regions of Interest(ROI)
    ROI = imprep.segmentImg(gray, 32, 32)
    #Subsample the ROI to half their original size
    newROI = imprep.subsampleRegions(ROI, 0.5)
    #Calculate a histogram for each ROI and return them in an array
    hist = imprep.generateHistograms(newROI)
    #For debugging purposes
    print(hist)
    x += 1
    #print(x/timeElapsed)
    continue