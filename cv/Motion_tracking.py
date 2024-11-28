import cv2 as cv
import numpy as np
a = False
kernel = np.ones((3,3),np.uint8)
cap  = cv.VideoCapture(0)
while (cap.isOpened()):
    ret , frame = cap.read()
    if a == True:
        grey_copy = grey_img.copy()
    
    
    
    
    if ret ==True:
        grey_img = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        tresh,result1 = cv.threshold(grey_img,128,255,cv.THRESH_BINARY_INV)
        result2 = cv.dilate(result1,kernel,iterations=5)
        # contours,hier = cv.findContours(result2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        # draw_con = cv.drawContours(frame,contours,-1,(255,0,0),2)
        x,y= grey_img.shape

        a = not a
        if a ==False:
            for i in range(x):
                for j in range(y):
                    if grey_img[i,j] != grey_copy[i,j]:
                        grey_img[i,j] = 0
                    else:
                        grey_img[i,j] =255
        else:
            continue

        cv.imshow("Frame",grey_img)
        cv.imshow("NIG",result2)
        if cv.waitKey(1) == ord("q"):
            break
cap.release()
cv.destroyAllWindows()