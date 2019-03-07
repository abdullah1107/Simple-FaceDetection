#Run command: python face_detection_camera.py
import cv2
#face cascades path
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#face detector method
def detect_face(img):

    face_image = img.copy()

    face_rects = face_cascade.detectMultiScale(face_image)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_image, (x,y), (x+w, y+h), (0,255,0), 3)

    return face_image

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)

    frame = detect_face(frame)

    cv2.imshow('Video Face Detection', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
