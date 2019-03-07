#run:python face_detection_eye.py
import cv2

face_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
eye_cascade_path = 'cascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


#face detector method
def detect_face_with_eye(img):

    face_image = img.copy()

    face_rects = face_cascade.detectMultiScale(face_image)

    boxes = []

    for (f_x,f_y,f_w,f_h) in face_rects:

        face_roi = img[f_y:f_y + f_h, f_x:f_x + f_w]
        boxes.append((f_x, f_y, f_x + f_w, f_y + f_h))
        #cv2.rectangle(face_image, (x,y), (x+w, y+h), (0,255,0), 3)
        eye_boxes = eye_cascade.detectMultiScale(face_roi)

        for (e_x, e_y, e_w, e_h) in eye_boxes:
            # Update the list of bounding boxes
            boxes.append((f_x + e_x, f_y + e_y, f_x + e_x + e_w, f_y + e_y + e_h))

    # Return the bounding boxes around the faces and eyes
    return boxes
# Load the video
camera = cv2.VideoCapture(0)

while True:
    # Grab the current frame
    _,frame = camera.read()


    # Resize the frame and convert it to greyscale
    #frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes in the image
    boxes = detect_face_with_eye(gray)

    # Draw the face bounding boxes
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Show the tracked eyes and face
    cv2.imshow("Tracking", frame)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
