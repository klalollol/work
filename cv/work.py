import cv2 as cv
import numpy as np
import datetime

img = cv.imread("whi.jfif")

# cv.rectangle(img,(0,0),(474,316),(0,0,255),10)


# cv.circle(img,(237,158),10,(0,0,255),10)

# cv.ellipse(img,(237,158),(100,50),0,0,360,(0,0,255),-1)

# p = np.array([[50,50],[59,100],[200,100]])
# cv.fillConvexPoly(img,points=[p],color= (0,0,255),lineType=10)
dt =str(datetime.datetime.now())

box = cv.rectangle(img,(0,316),(474,255),(0,0,0),-1)
font =cv.FONT_HERSHEY_PLAIN
text = cv.putText(box,"I'm white but inside I'm black",(200,300),font,1,(255,255,255),1)
time = cv.putText(text,dt,(100,300),font,1,(255,255,255),1)
cv.imshow("e",text)

cv.waitKey(0)
cv.destroyAllWindows()