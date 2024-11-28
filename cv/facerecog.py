import cv2
import datetime
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img_id=0

def save_image_with_filename(img,filename):
    filepath = f"path/{filename}" 
    cv2.imwrite(filepath, img)


def draw_boundary(img,clf):
      current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      face_detect=face_cascade.detectMultiScale(gray_img,1.1,5) 
      xywh = []
      for (x,y,w,h) in face_detect :
            cv2.rectangle(img, (x,y), (x+w, y+ h), (0,255,0),5) 
            cv2.rectangle(img,(x,y-50),(x+w,y),(0,0,255), -1)
            id, con = clf.predict(gray_img[y:y+h,x:x+w])
            if id ==1 and con <= 50 :
                  cv2.putText(img, "year", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.nameyear.time{current_time}.jpg"
                  save_image_with_filename(frame,filename) 
            elif id ==2 and con <= 50 :
                  cv2.putText(img, "tawan", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.nametawan.time{current_time}.jpg"
                  save_image_with_filename(frame,filename)                   
            elif id ==3 and con <= 50 :
                  cv2.putText(img, "nam", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.namenam.time{current_time}.jpg"
                  save_image_with_filename(frame,filename)   
            if id ==4 and con <= 50 :
                  cv2.putText(img, "dd", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.nameprim.time{current_time}.jpg"
                  save_image_with_filename(frame,filename) 
            if id ==5 and con <= 50 :
                  cv2.putText(img, "prim", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.nameprim.time{current_time}.jpg"
                  save_image_with_filename(frame,filename)          
            elif id ==6 and con <= 50 :
                  cv2.putText(img, "kla", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                  filename = f"id{id}.namekla.time{current_time}.jpg"
                  save_image_with_filename(frame,filename)
                                   
            else :
                  cv2.putText(img, "unknown", (x+10, y-10), cv2. FONT_HERSHEY_SIMPLEX, 2, (255,255, 255), 3)
            show_con ="{0}%".format(round(100-con))
            cv2.rectangle(img, (x+10,y+h+10), (x+w,y+h+50), (255,0,255), -1)
            cv2.putText(img, show_con, (x+10,y+h+40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 2) 
            xywh=[x,y,w,h]
      return img,xywh
def detect(img,img_id,clf) :
      img,xywh=draw_boundary(img, clf) 
      if len(xywh) == 4 :
            result=img[xywh[1]:xywh[1 ]+xywh[3],xywh[0]:xywh[0]+xywh[2]]
      return img

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("Year_Tawan_Nam_DD_Kla_Preme_classifier.xml")


while (True):
      check, frame=cap.read()
      frame=detect(frame,img_id,clf)
      cv2.imshow("output camera",frame)
      img_id +=1
      if cv2.waitKey(1) & 0xFF == ord("g"):
            break
cap.release()
cv2.destoryAllWindows()