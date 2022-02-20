import cv2

#Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# grab webcam feed
webcam = cv2.VideoCapture(0)

# show the current frame
while True:

    #read the current frame from webcam video stream
    successful_frame_read, frame = webcam.read()

    # if tere's an error, abort
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    # run face detection within each of those face
    for (x, y, w, h) in faces:
        
        # draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 2)

        the_face = (x, y, w, h)
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # find all smiles in the face
        for (x_, y_, w_, h_) in smiles:

            # draw a rectangles around the smile
            cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 2)

    # show the current frame
    cv2.imshow('Smile Detector', frame)

    # display
    cv2.waitKey(1)

# cleanup
webcam.release()
cv2.destroyAllWindows()


print('ok')




"""
import cv2

# Load some pre-trained data on face frontals from povencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_face_data = cv2.CascadeClassifier('haarcascade_smile.xml')

# choose an image to detect face in
#img = cv2.imread('ME.jpg')
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    # stop if key is pressed
    if key == 76 or key == 108:
        break


#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#print(face_coordinates)

cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print("Code Completed!")


"""