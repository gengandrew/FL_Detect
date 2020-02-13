from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True)
args = vars(ap.parse_args())

print("Got here 1")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("Got here 2")
vs = VideoStream(0).start()
time.sleep(2.0)
print("Got here 3")

while True:
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
        index = 1
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            if index == 49 or index == 55:
                print("At index "+str(index)+" value of ["+str(x)+","+str(y)+"]")
            
            index = index+1

        # Start frame streaming
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Upon q key press end session
        if key == ord("q"):
            break


# Cleanup video stream and etc.
cv2.destroyAllWindows()
vs.stop()
print("Cleanup complete")
