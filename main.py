from imutils.video import VideoStream
from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
import numpy as np
import imutils
import time
import dlib
import math
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_landmarks.dat")
deviation = 40

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
            xylist = []
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=1)
                xylist.append((float(x), float(y)))

            cv2.line(frame, (int(xylist[2][0]), int(xylist[2][1])), (int(xylist[14][0]), int(xylist[14][1])), (255, 0, 255), thickness=2)
            cv2.line(frame, (int(xylist[8][0]), int(xylist[8][1])), (int(xylist[27][0]), int(xylist[27][1])), (255, 0, 255), thickness=2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("o"):
                L1 = line((xylist[2][0], xylist[2][1]), (xylist[14][0], xylist[14][1]))
                L2 = line((xylist[8][0], xylist[8][1]), (xylist[27][0], xylist[27][1]))
                R = intersection(L1, L2)
                normalize = []
                for (x, y) in xylist:
                    normalize.append(distance.euclidean((x,y),R))
                
                ret = {
                    "hline": distance.euclidean((xylist[2][0], xylist[2][1]), (xylist[14][0], xylist[14][1])),
                    "vline": distance.euclidean((xylist[8][0], xylist[8][1]), (xylist[27][0], xylist[27][1])),
                    "normalize": normalize
                }
                cv2.destroyAllWindows()
                vs.stop()
                return ret


def dataCollection(calibration):
    vs = VideoStream(0).start()
    time.sleep(2.0)
    normalize = calibration["normalize"]
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
        currTime = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
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
                L1 = line((xylist[2][0],xylist[2][1]),(xylist[14][0],xylist[14][1]))
                L2 = line((xylist[8][0],xylist[8][1]),(xylist[27][0],xylist[27][1]))
                R = intersection(L1, L2)
                distList = []
                index = 0
                for (x, y) in xylist:
                    # TODO: Potentially parse out all the not needed points
                    distList.append(distance.euclidean((x,y),R)/normalize[index])
                    print("Unnormalize is " + str(distance.euclidean((x,y),R)) + " normalize is " + str(normalize[index]) + " resulting value is " + str(distance.euclidean((x,y),R)/normalize[index]))
                    index = index + 1
                
                data.append((distList, currTime))
            else:
                print("Failed deviation check")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                vs.stop()
                print("Cleanup complete")
                return data


initCal = getInitialCal()
data = dataCollection(initCal)
print(data)
