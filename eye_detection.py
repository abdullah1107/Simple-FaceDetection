#run command: python eye_detection.py
import cv2

#face cascades path
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

#face detector method
def detect_eye(img):

    eye_image = img.copy()

    eye_rects = eye_cascade.detectMultiScale(eye_image)

    for (x,y,w,h) in eye_rects:
        cv2.rectangle(eye_image, (x,y), (x+w, y+h), (255,255,0), 3)

    return eye_image

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)

    frame = detect_eye(frame)

    cv2.imshow('Eye', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
