#run command: python face_detection_image.py
import cv2


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#face detector method
def detect_face(img):

    face_image = img.copy()

    face_rects = face_cascade.detectMultiScale(face_image)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_image, (x,y), (x+w, y+h), (0,255,0), 5)

    return face_image

#load image read the image
image = cv2.imread('images/girl.jpg')
detect_face_image = detect_face(image)
#show image
cv2.imshow("Detect-face",detect_face_image)
cv2.waitKey(0)
