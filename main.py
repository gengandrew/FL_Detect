from imutils.video import VideoStream
from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import dlib
import math
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_landmarks.dat")
face_size_deviation = 40 # Deviation check for the expected size of face
face_landmark_deviation_x = 38 # Deviation check for expected position of landmark
face_landmark_deviation_y = 25 # Deviation check for expected position of landmark
time_format = '%m-%d-%Y %H:%M:%S.%f'


def graph(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for each in data:
        distList = each[0]
        index = 0
        for distance in distList:
            ax.scatter(distance, index, alpha=0.8, c="red", edgecolors='none', s=30)
            index = index+1

    plt.title('Facial landmarks scatter plot')
    plt.legend(loc=2)
    plt.show()


##
# Fitting a linear line between two specified points
##
def getLine(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

##
# Taking the intersection between two specified points
##
def getIntersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x),int(y)
    else:
        return False

##
# Takes a line and point and finds the closest distance
# between the line and point
##
def getLinePointDistance(line, point):
    # distance = (Ax + By + C) / sqrt(A^2 + B^2)
    A = line[0]
    B = line[1]
    C = -line[2]
    numerator = (A*point[0])+(B*point[1])+C
    denominator = math.sqrt(math.pow(A,2)+math.pow(B,2))
    return numerator / denominator


def getInitialCal():
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        current_time = datetime.utcnow().strftime(time_format)[:-3]
        rects = detector(frame, 0)

        if(len(rects) == 0):
            print("Failed to recognize face")

        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            xylist = [] # Datastructure for holding in raw facial features

            for (x, y) in shape:
                xylist.append((float(x), float(y)))

            temp_intersect = getIntersection(getLine(xylist[0], xylist[16]), getLine(xylist[8], xylist[27]))
            scalar = (int(xylist[8][0]-temp_intersect[0]), int(xylist[8][1]-temp_intersect[1]))
            
            top_right = getIntersection(getLine((xylist[19][0], xylist[19][1]), (xylist[24][0], xylist[24][1])), getLine((xylist[15][0], xylist[15][1]), (xylist[16][0], xylist[16][1])))
            top_left = getIntersection(getLine((xylist[19][0], xylist[19][1]), (xylist[24][0], xylist[24][1])), getLine((xylist[0][0], xylist[0][1]), (xylist[1][0], xylist[1][1])))
            bot_right = (int(xylist[16][0]+scalar[0]), int(xylist[16][1]+scalar[1]))
            bot_left = (int(xylist[0][0]+scalar[0]), int(xylist[0][1]+scalar[1]))
            
            cv2.line(frame, (int(xylist[2][0]), int(xylist[2][1])), (int(xylist[14][0]), int(xylist[14][1])), (255, 0, 255), thickness=2)
            cv2.line(frame, (int(xylist[8][0]), int(xylist[8][1])), (int(xylist[27][0]), int(xylist[27][1])), (255, 0, 255), thickness=2)
            cv2.line(frame, top_left, top_right, (255, 0, 255), thickness=2)
            cv2.line(frame, bot_left, bot_right, (255, 0, 255), thickness=2)
            cv2.line(frame, top_left, bot_left, (255, 0, 255), thickness=2)
            cv2.line(frame, top_right, bot_right, (255, 0, 255), thickness=2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("o"):
                x_axis = getLine(top_left, bot_left)
                y_axis = getLine(bot_left, bot_right)

                normalize = []
                for (x,y) in xylist:
                    newX = getLinePointDistance(x_axis, (x,y))
                    newY = getLinePointDistance(y_axis, (x,y))
                    normalize.append((newX, newY))
                
                ret = {
                    "normalize": normalize,
                    "hline": distance.euclidean((xylist[2][0], xylist[2][1]), (xylist[14][0], xylist[14][1])),
                    "vline": distance.euclidean((xylist[8][0], xylist[8][1]), (xylist[27][0], xylist[27][1]))
                }

                cv2.destroyAllWindows()
                return ret


def dataCollection(calibration):
    data = []
    camera = cv2.VideoCapture(0)
    data_iteration = 0
    frame_window = 1

    while True:
        _, frame = camera.read()
        data_iteration = data_iteration + 1
        current_time = datetime.utcnow().strftime(time_format)[:-3]
        rects = detector(frame, 0)

        if(len(rects) == 0):
            print("Failed to recognize face " + str(data_iteration))

        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            xylist = [] # Datastructure for holding in raw facial features

            for (x, y) in shape:
                xylist.append((float(x), float(y)))

            temp_intersect = getIntersection(getLine(xylist[0], xylist[16]), getLine(xylist[8], xylist[27]))
            scalar = (int(xylist[8][0]-temp_intersect[0]), int(xylist[8][1]-temp_intersect[1]))
            
            top_right = getIntersection(getLine((xylist[19][0], xylist[19][1]), (xylist[24][0], xylist[24][1])), getLine((xylist[15][0], xylist[15][1]), (xylist[16][0], xylist[16][1])))
            top_left = getIntersection(getLine((xylist[19][0], xylist[19][1]), (xylist[24][0], xylist[24][1])), getLine((xylist[0][0], xylist[0][1]), (xylist[1][0], xylist[1][1])))
            bot_right = (int(xylist[16][0]+scalar[0]), int(xylist[16][1]+scalar[1]))
            bot_left = (int(xylist[0][0]+scalar[0]), int(xylist[0][1]+scalar[1]))
            
            cv2.line(frame, (int(xylist[2][0]), int(xylist[2][1])), (int(xylist[14][0]), int(xylist[14][1])), (255, 0, 255), thickness=2)
            cv2.line(frame, (int(xylist[8][0]), int(xylist[8][1])), (int(xylist[27][0]), int(xylist[27][1])), (255, 0, 255), thickness=2)
            cv2.line(frame, top_left, top_right, (255, 0, 255), thickness=2)
            cv2.line(frame, bot_left, bot_right, (255, 0, 255), thickness=2)
            cv2.line(frame, top_left, bot_left, (255, 0, 255), thickness=2)
            cv2.line(frame, top_right, bot_right, (255, 0, 255), thickness=2)

            cv2.imshow("Frame", frame)

            x_axis = getLine(top_left, bot_left)
            y_axis = getLine(bot_left, bot_right)
            nhline = distance.euclidean((xylist[2][0], xylist[2][1]), (xylist[14][0], xylist[14][1]))
            nvline = distance.euclidean((xylist[8][0], xylist[8][1]), (xylist[27][0], xylist[27][1]))

            if abs(float(calibration['hline']) - float(nhline)) < face_size_deviation and abs(float(calibration['vline']) - float(nvline)) < face_size_deviation:
                index = 0
                isToss = False
                parsed_xylist = []
                for (x,y) in xylist:
                    newX = getLinePointDistance(x_axis, (x,y))
                    newY = getLinePointDistance(y_axis, (x,y))
                    parsed_xylist.append((newX, newY))
                    if index in range(27,36):
                        if abs(newX - calibration['normalize'][index][0]) > face_landmark_deviation_x:
                            isToss = True
                        elif abs(newY - calibration['normalize'][index][1]) > face_landmark_deviation_y:
                            isToss = True
                
                    index = index + 1

                # Secondary deviation check for the coordinates of the landmarks
                if not isToss and data_iteration%frame_window == 0:
                    data.append((current_time, parsed_xylist))
                elif data_iteration%frame_window == 0:
                    print("Failed face landmark deviation check " + str(data_iteration))
            
            else:
                print("Failed face width deviation check " + str(data_iteration))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return data


initCal = getInitialCal()
data = dataCollection(initCal)

print(data)
