import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
font =cv.FONT_HERSHEY_PLAIN



vid = cv.VideoCapture(0) 
while(True): 
    ret, frame = vid.read() 
    r =  frame.shape
    y = r[0]
    x = r[1]
    # Display the resulting frame 
    cv.circle(frame,(x//2,y//2),50,(50,255,25),10)
    cv.rectangle(frame,(0,0),(x,y),(255,50,50),20)
    cv.line(frame,(0,0),(x,y),(50,50,255),10)
    cv.putText(frame,"Live Video",(x//20,y//13),font,3,(255,255,255),3)

    

    
    cv.imshow('work2', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break