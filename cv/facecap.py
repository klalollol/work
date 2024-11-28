import cv2 as cv
import time
cap = cv.VideoCapture(0)
face_cascde = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascde = cv.CascadeClassifier("haarcascade_smile.xml")
eye_cascde = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
count = 0
countdown = None
while True:
    ret,frame =  cap.read()
    if ret:
        frame_copy = frame.copy()
        gray_cap = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        eye_dectect = eye_cascde.detectMultiScale(gray_cap)
        face_detect = face_cascde.detectMultiScale(gray_cap)
        smile_dectect = smile_cascde.detectMultiScale(gray_cap)
        if len(smile_dectect)> 20:
            for (x1,y1,w1,h1) in face_detect:
                radius =min(w1,h1)//2
                center_x = x1 + w1 // 2
                center_y = y1 + h1 // 2
                cv.circle(frame_copy,(center_x,center_y),radius,(0,255,0),5)
            for (x2,y2,w2,h2) in eye_dectect:
                radius =min(w2,h2)//2
                center_x = x2 + w2 // 2
                center_y = y2 + h2 // 2
                cv.circle(frame_copy,(center_x,center_y),radius,(255,0,0),5)
            if countdown == None:
                countdown = time.time()
            if countdown !=None:
                t = time.time()- countdown
                cv.putText(frame_copy,str(t),(50,50),cv.FONT_HERSHEY_SIMPLEX ,1,(0,255,255),4)
                
                if t >= 3:
                    count +=1
                    cv.imwrite(f"gg{count}.jpg",frame)
                    countdown =None
                    t = None
        else:        
            for (x,y,w,h) in face_detect:
                cv.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),5)
            for (x,y,w,h) in eye_dectect:
                cv.rectangle(frame_copy,(x,y),(x+w,y+h),(255,0,0),5)
            countdown= None
            
        cv.imshow("frame",frame_copy)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cv.destroyAllWindows()