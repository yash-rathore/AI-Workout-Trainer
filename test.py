'''
AI workout trainer
1> load a video/live feed
2> detect body landmarks and draw on frame with tracking
3> body landmark of elbow and hand necessary , both hands=>13,15,14,16
4> if lm[elbow] is above lm[hand] => rep progress=0
    if lm[elbow] is below lm[hand] => rep progress=100
    and thus it fluctuates bw in bw
5> draw a progress bar that fills up according to distance bw lm[elbow] and lm[hand]
6> counting reps :)
'''

import poseestimationmodule as pem
import cv2
import numpy as np

cap = cv2.VideoCapture('dbcurls.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
detector = pem.posedetector()
text = "FPS {}".format(fps)

reps=0
t=0

while True:
    _, img = cap.read()
    img = detector.findpose(img, draw=False)
    lmlist = detector.findposition(img, draw=False)

    lelbowx, lelbowy = lmlist[13][1], lmlist[13][2]
    relbowx, relbowy = lmlist[14][1], lmlist[14][2]
    lwristx, lwristy = lmlist[15][1], lmlist[15][2]
    rwristx, rwristy = lmlist[16][1], lmlist[16][2]

    lmpoints = [(lelbowx, lelbowy),
                (lwristx, lwristy),
                (relbowx, relbowy),
                (rwristx, rwristy)]

    for i in range(0, len(lmpoints)):
        cv2.circle(img, lmpoints[i], 5, (255, 0, 0), 3)

    # build progress bar acc to reps in exercise:

    cv2.rectangle(img, (1000, 100), (1050, 600), (0, 0, 0), 5)
    cv2.rectangle(img, (990, 90), (1060, 610), (0, 0, 0), 5)
    cv2.putText(img, "Progress Bar : ", (920, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)

    if len(lmlist) != 0:
        color = (255, 0, 0)
        dist = int(lmpoints[1][1]) - int(lmpoints[0][1])
        if dist < 0:
            dist = 0
            color = (0, 255, 0)

        max = 87
        per = (max - dist) / max
        cv2.putText(img, "Percentage: {} %".format(int(per * 100)), (930, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)

        cv2.rectangle(img, (1000, 600-(int(500*per))), (1050, 600), color,cv2.FILLED)

        #how to count reps
        if per==1 and t==1:
            reps+=1
            t=0
        elif per!=1:
            t=1
        cv2.putText(img, "Total Reps: {}".format(reps), (930, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)

    cv2.line(img, (lelbowx, lelbowy),(lwristx, lwristy) ,(255,255,255), 3)
    cv2.line(img, (relbowx, relbowy),(rwristx, rwristy) ,(255,255,255), 3)

    cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
    cv2.putText(img, "Personal AI Trainer", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
