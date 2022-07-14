import cv2
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
c=0
while True:
    ret,frame=video.read()
    faces=facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        c=c+1
        name='./images/1/'+str(c)+'.jpg' #(dataset of images with mask)
        # name='./images/1/'+str(c)+'.jpg' (dataset of images without mask)
        print("Creating Image..."+name)
        cv2.imwrite(name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow("Data Window",frame)
    k=cv2.waitKey(1)
    if c>500:
        break
video.release()
cv2.destroyAllWindows()
