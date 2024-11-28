import cv2 as cv
import numpy as np
from random import randint
import math

ix,iy = -1,-1
img =  np.zeros([300,300,3],np.uint8)
draw =False

def draw_rectangle(event, x, y, flags, param):
    global ix,iy,draw
    r = randint(0,255)
    g = randint(0,255)
    b = randint(0,255)
    color = (b,g,r)
    if event == cv.EVENT_LBUTTONDOWN:
        draw = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if draw:
            img2 = img.copy()
            cv.rectangle(img2, (ix, iy), (x, y), color, 2)
            cv.imshow('image', img2)
            
    elif event == cv.EVENT_LBUTTONUP:
        draw = False
        cv.rectangle(img, (ix, iy), (x, y), color, 2)
        cv.imshow('image', img)


def draw_circle(event, x, y, flags, param):
    global ix,iy,draw
    r = randint(0,255)
    g = randint(0,255)
    b = randint(0,255)
    color = (b,g,r)
    rad = ((x-ix)**2+(y-iy)**2)
    r = math.ceil(math.sqrt(rad))
    if event == cv.EVENT_LBUTTONDOWN:
        draw = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if draw:
            img2 = img.copy()
            cv.circle(img2, (ix, iy), r, color, 2)
            cv.line(img2,(ix,iy),(x,y),color,2)
            cv.putText(img2,f"{r}",(ix,iy),1,3,(255,255,255),3)
            cv.imshow('image', img2)
            
    elif event == cv.EVENT_LBUTTONUP:
        draw = False
        cv.circle(img, (ix, iy), r, color, 2)
        cv.imshow('image', img)



D = draw_circle
drawing =False
while True:
    if cv.waitKey(0)== ord('t'):
        drawing = not drawing
        if drawing == False:
            D = draw_circle
        else:
            D = draw_rectangle
    cv.imshow("image",img)
    cv.setMouseCallback("image",D)
    if cv.waitKey(0) == ord("q"):
        cv.destroyAllWindows()
        break