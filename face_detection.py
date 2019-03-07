from FaceDetectorClass import FaceDetector
import cv2

# Define paths
image_path = 'images/girl_face.jpg'
cascade_path = 'cascades/haarcascade_frontalface_default.xml'


#comment section is comment for read image from image_path
'''
# Load the image and convert it to greyscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find faces in the image
detector = FaceDetector(cascade_path)
face_boxes = detector.detect(gray, 1.2, 5)
print("{} face(s) found".format(len(face_boxes)))

# Loop over the faces and draw a rectangle around each
for (x, y, w, h) in face_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 5)

# Show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
'''

#real time face Detection
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = FaceDetector(cascade_path)
    face_boxes = detector.detect(gray, 1.2, 5)
    #print("{} face(s) found".format(len(face_boxes)))

    # Loop over the faces and draw a rectangle around each
    for (x, y, w, h) in face_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)


    #frame = detect_face(frame)

    cv2.imshow('Video Face Detection', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
