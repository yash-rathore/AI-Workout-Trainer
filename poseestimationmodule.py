import cv2
import mediapipe as mp

class posedetector():
    def __init__(self,mode=False,upbody=False,
                 smooth=True,detectionconfidence=0.5,
                 trackingconfidence=0.5):
        self.mode=mode
        self.upbody=upbody
        self.smooth=smooth
        self.detectionconfidence=detectionconfidence
        self.trackingconfidence=trackingconfidence

        self.mpdraw = mp.solutions.drawing_utils
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose(self.mode,self.upbody,self.smooth,self.detectionconfidence,self.trackingconfidence)

    def findpose(self,img,draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgrgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpdraw.draw_landmarks(img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)
        return img

    def findposition(self,img,draw=True):
        lmlist=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy,cz,visibility = int(lm.x * w), int(lm.y * h),int(lm.z),lm.visibility
                lmlist.append([id,cx,cy,cz,visibility])
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
        return lmlist

def main():
    cap=cv2.VideoCapture('pose.mp4')
    fps=cap.get(cv2.CAP_PROP_FPS)
    detector=posedetector()
    text="fps => {}".format(fps)
    while True:
        _, img = cap.read()
        img=detector.findpose(img,draw=True)
        lmlist=detector.findposition(img,draw=False)
        print(lmlist)
        cv2.putText(img,text,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()