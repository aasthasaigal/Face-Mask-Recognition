import numpy as np
import cv2
from keras.models import load_model
model = load_model('MyTrainingModel.h5')
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.5
vid=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX

def preprocessing(img):
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

def get_className(classNo):
	if classNo==0:
		return "MASK DETECTED"
	elif classNo==1:
		return "MASK NOT DETECTED"

while True:
	success, frame=vid.read()
	faces = facedetect.detectMultiScale(frame,1.3,5)

	for x,y,w,h in faces:
		crop_img=frame[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (32,32))
		img=preprocessing(img)
		img=img.reshape(1,32,32,1)
		prediction=model.predict(img)
		classIndex=np.argmax(prediction,axis=1)
		probabilityValue=np.amax(prediction)

		if probabilityValue>threshold:
			if classIndex==0:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.putText(frame, str(get_className(classIndex)),(x-10,y-10),font,1,(0,255,0),2)

			elif classIndex==1:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
				cv2.putText(frame, str(get_className(classIndex)),(x-25,y-10),font,1,(0,0,255),2)
	cv2.imshow("Face Mask Recognition",frame)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
vid.release()
cv2.destroyAllWindows()