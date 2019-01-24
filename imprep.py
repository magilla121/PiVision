import numpy as np
import cv2 as cv

#Splits an image into smaller sub-images of given pixel dimensions and returns them all in an array
def segmentImg(image, segmentWidth, segmentHeight):
    sw = segmentWidth
    sh = segmentHeight
    imgWidth = image.shape[1]
    imgHeight = image.shape[0]
    horizontalSegments = int(imgWidth/sw)
    verticalSegments = int(imgHeight/sh)
    #Regions of interest are simply returned as an array
    ROI = []

    for y in range(0,verticalSegments):
        for x in range(0,horizontalSegments):
            region = image[sh * y: (sh * y) + sh, sw * x: (sw * x) + sw]
            ROI.append(region)
    return ROI

#Resizes an image by a given scale. Optionally applies a Gaussian blur
def downsample(_image, scale, blur=False):
    image = _image
    if(blur):
        cv.gaussianBlur(image, (5,5), 0)
    res = cv.resize(image, (0,0), fx=scale, fy=scale)
    return res

def subsampleRegions(regions, _scale):
    scaledROI = []
    for r in regions:
        res = downsample(r,_scale)
        scaledROI.append(res)
    return scaledROI

#Takes a set of images and returns an array of their histograms. Regions must be an array
def generateHistograms(regions):
    histograms = []
    for r in regions:
        newHist = np.histogram(r.ravel(), 256, [0,256])
        histograms.append(newHist)
    return histograms


#Debugging functions
def showROI(roi):
    for x in range(0,len(roi)):
        for y in range(0,len(roi)):
            roi[x][y] = 255