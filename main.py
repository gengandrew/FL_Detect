from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils import face_utils
from scipy.spatial import distance
from datetime import datetime
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_landmarks.dat")
deviation = 40

def solveLinearEq(p1, p2):
    points = [p1, p2]
    xCoords, yCoords = zip(*points)
    matrixA = vstack([xCoords,ones(len(xCoords))]).T
    m, c = lstsq(matrixA, yCoords)[0]
    # print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    return m, c


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def getInitialCal():
    vs = VideoStream(0).start()
    time.sleep(2.0)

    print("Starting initial calibration")
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
                ret = {
                    "hline": distance.euclidean((xlist[2], ylist[2]), (xlist[14], ylist[14])),
                    "vline": distance.euclidean((xlist[8], ylist[8]), (xlist[27], ylist[27]))
                }
                cv2.destroyAllWindows()
                vs.stop()
                return ret


def dataCollection(calibration):
    vs = VideoStream(0).start()
    time.sleep(2.0)
    data = []

    print("Starting data collection")
    while True:
        shouldExit = False
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rects = detector(gray, 0)
        # print("rects1 = " + str(len(rects)))
        rects = detector(frame, 0)
        currTime = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        if len(rects) == 0:
            print("Face detection error")

        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            xylist = []
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=2)
                xylist.append((float(x), float(y)))

            nhline = distance.euclidean((xylist[2][0], xylist[2][1]), (xylist[14][0], xylist[14][1]))
            nvline = distance.euclidean((xylist[8][0], xylist[8][1]), (xylist[27][0], xylist[27][1]))
            if abs(float(calibration["hline"]) - float(nhline)) < deviation and abs(float(calibration["vline"]) - float(nvline)) < deviation:
                m, c = solveLinearEq((xylist[2][0],xylist[2][1]),(xylist[14][0],xylist[14][1]))
                m, c = solveLinearEq((xylist[8][0],xylist[8][1]),(xylist[27][0],xylist[27][1]))
                L1 = line((xylist[2][0],xylist[2][1]),(xylist[14][0],xylist[14][1]))
                L2 = line((xylist[8][0],xylist[8][1]),(xylist[27][0],xylist[27][1]))
                R = intersection(L1, L2)
                distList = []
                for (x, y) in xylist:
                    distList.append((distance.euclidean((x,y),R), currTime))
                
                data.append(distList)
            else:
                print("Failed deviation check")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(data[1])
                cv2.destroyAllWindows()
                vs.stop()
                print("Cleanup complete")
                return data


initCal = getInitialCal()
data = dataCollection(initCal)


# Some normalizing calculations
# p32 = (xlist[31], ylist[31])
# p36 = (xlist[35], ylist[35])

# xmean = np.mean(xlist)
# ymean = np.mean(ylist)
# xcentral = [(x-xmean) for x in xlist]
# ycentral = [(y-ymean) for y in ylist]
# print("At new index 49 value of ["+str(xcentral[48])+","+str(ycentral[48])+"]")
# print("At new index 55 value of ["+str(xcentral[54])+","+str(ycentral[54])+"]")

# landmarks_vectorised = []
# indexer = 1
# for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
#     landmarks_vectorised.append(w)
#     landmarks_vectorised.append(z)
#     meannp = np.asarray((ymean,xmean))
#     coornp = np.asarray((z,w))
#     dist = np.linalg.norm(coornp-meannp)
#     landmarks_vectorised.append(dist)
#     landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
#     if indexer == 49 or indexer == 55:
#         print("At index " + str(indexer))
#         print(dist)
#         print((math.atan2(y, x)*360)/(2*math.pi))
    
#     indexer = indexer+1