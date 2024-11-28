import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

font =cv.FONT_HERSHEY_PLAIN

img =np.full((600,600,3),255,dtype="uint8")
blue = cv.rectangle(img,(200,250),(400,350),(255,50,50),-1)
green = cv.circle(img,(300,300),50,(25,25,255),-1)
line = cv.line(img,(0,0),(600,600),(50,255,50),10)
text = cv.putText(img,"OpenCV Homework",(100,500),font,3,(0,0,0),1)


cv.imshow("1",blue)
cv.waitKey(0)
cv.destroyAllWindows()