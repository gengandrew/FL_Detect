from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils import face_utils
from scipy.spatial import distance
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_landmarks.dat")

def getInitialCal():
    vs = VideoStream(0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        rects = detector(frame, 0)
        if len(rects) == 0:
            print("Face detection error")

        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            xlist = []
            ylist = []
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=1)
                xlist.append(float(x))
                ylist.append(float(y))

            cv2.line(frame, (int(xlist[2]), int(ylist[2])), (int(xlist[14]), int(ylist[14])), (255, 0, 255), thickness=2)
            cv2.line(frame, (int(xlist[8]), int(ylist[8])), (int(xlist[27]), int(ylist[27])), (255, 0, 255), thickness=2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("o"):
                distance.euclidean((xlist[2], ylist[2]), (xlist[14], ylist[14]))
                ret = {
                    "hline": distance.euclidean((xlist[2], ylist[2]), (xlist[14], ylist[14])),
                    "vline": distance.euclidean((xlist[8], ylist[8]), (xlist[27], ylist[27]))
                }
                cv2.destroyAllWindows()
                vs.stop()
                return ret


initCal = getInitialCal()
vs = VideoStream(0).start()
time.sleep(2.0)

while True:
    shouldExit = False
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 0)
    # print("rects1 = " + str(len(rects)))
    rects = detector(frame, 0)
    if len(rects) == 0:
        print("Face detection error")

    for rect in rects:
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        xlist = []
        ylist = []
        index = 1
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=2)
            xlist.append(float(x))
            ylist.append(float(y))
            index = index+1

        # Start frame streaming
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Some normalizing calculations
        p32 = (xlist[31], ylist[31])
        p36 = (xlist[35], ylist[35])

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        print("At new index 49 value of ["+str(xcentral[48])+","+str(ycentral[48])+"]")
        print("At new index 55 value of ["+str(xcentral[54])+","+str(ycentral[54])+"]")
        
        landmarks_vectorised = []
        indexer = 1
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
            if indexer == 49 or indexer == 55:
                print("At index " + str(indexer))
                print(dist)
                print((math.atan2(y, x)*360)/(2*math.pi))
            
            indexer = indexer+1
        # for i in range(1, 68):
        #     if(i == 49 or i == 55):
        #         print("At new index "+str(i)+" value of ["+str(xcentral[i])+","+str(ycentral[i])+"]")

        # Upon q key press end session
        if key == ord("q"):
            shouldExit = True
            break
    
    if shouldExit:
        break


# Cleanup video stream and etc.
cv2.destroyAllWindows()
vs.stop()
print("Cleanup complete")